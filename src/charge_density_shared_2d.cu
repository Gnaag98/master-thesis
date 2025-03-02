#include "charge_density_shared_2d.cuh"

#include <cub/device/device_radix_sort.cuh>

namespace {
    /// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
    constexpr
    auto ceil_pow2(const int number) -> int {
        auto v = static_cast<uint32_t>(number);
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        v += v == 0;
        return static_cast<int>(v);
    }
};

__global__
void thesis::shared_2d::initialize_particle_indices(
    const size_t particle_count, int *indices
) {
    // Grid-stride loop. Equivalent to regular if-statement if grid is large
    // enough to cover all iterations of the loop.
    for (
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < particle_count;
        index += blockDim.x * gridDim.x
    ) {
        indices[index] = index;
    }
}

__global__
void thesis::shared_2d::associate_particles_with_cells(
    const FP *pos_x, const FP *pos_y, const size_t particle_count,
    const int3 grid_dimensions, const int cell_size, int *cell_indices
) {
    // Grid-stride loop. Equivalent to regular if-statement if grid is large
    // enough to cover all iterations of the loop.
    for (
        auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
        particle_index < particle_count;
        particle_index += blockDim.x * gridDim.x
    ) {
        // Position in world coordinates.
        const auto position = FP3{
            pos_x[particle_index], pos_y[particle_index], 0
        };
        // Position in grid coordinates, with origin at the first cell center.
        const auto [ u, v, w ] = cell_coordinates(position, cell_size);
        // 2D indices of first enclosing cell, by first meaning the one with
        // lowest index, i.e., closest to the origin.
        const auto i = static_cast<int>(floor(u));
        const auto j = static_cast<int>(floor(v));
        // Store 1D index.
        cell_indices[particle_index] = i + j * grid_dimensions.x;
    }
}

void thesis::shared_2d::sort_particles_by_cell(
    void *sort_storage, size_t &sort_storage_size,
    const int *associated_cells_in, int *associated_cells_out,
    const int *particle_indices_in, int *particle_indices_out,
    const size_t particle_count
) {
    cub::DeviceRadixSort::SortPairs(
        sort_storage, sort_storage_size,
        associated_cells_in, associated_cells_out,
        particle_indices_in, particle_indices_out,
        particle_count
    );
}

__global__
void thesis::shared_2d::contextualize_cell_associations(
    const size_t particle_count, const int *associated_cell_indices,
    int *particle_indices_rel_cell, int *particle_count_per_cell
) {
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto global_index = threadIdx.x + blockIdx.x * blockDim.x;
        global_index < particle_count;
        global_index += blockDim.x * gridDim.x
    ) {
        extern __shared__ int s_memory[];
        auto s_ptr = s_memory;
        // Shared memory used as working memory for the single thread. Might
        // speed up the algirithm by not having to access global memory, or
        // unnecessary since the data would be cached after the first read.
        auto s_cell_indices = s_ptr;
        s_ptr += blockDim.x;
        s_cell_indices[threadIdx.x] = associated_cell_indices[global_index];
        __syncthreads();

        // Shared memory populated by the single thread. Used by all threads
        // to populate global memory.
        auto s_particle_indices_rel_cell = s_ptr;
        s_ptr += blockDim.x;
        // Incrementing 0, 1, 2, ..., for each cell in block. Not to be
        // confused with cell index.
        auto s_cell_ids = s_ptr;
        s_ptr += blockDim.x;
        // Indexed with cell id, not index. Sized for worst case scenario with
        // one cell per particle.
        auto s_particle_count_per_cell_id = s_ptr;
        s_ptr += blockDim.x;

        // Linear algorithm using only one thread per block.
        if (threadIdx.x == 0) {
            // Unroll first iteration.
            s_particle_indices_rel_cell[0] = 0;
            s_cell_ids[0] = 0;

            // Loop memory.
            // Start on 1 since we already set the value for the first particle.
            auto particle_index_rell_cell = 1;
            auto cell_particle_count = 1;
            auto cell_id = 0;
            auto previous_cell_index = s_cell_indices[0];
            // Don't iterate too far in the last block.
            const auto block_particle_count = min(
                blockDim.x,
                static_cast<uint>(particle_count) - blockIdx.x * blockDim.x
            );
            for (auto i = 1u; i < block_particle_count; ++i) {
                const auto cell_index = s_cell_indices[i];
                if (cell_index == previous_cell_index) {
                    ++cell_particle_count;
                } else {
                    // Store accumulated particle count for last cell.
                    s_particle_count_per_cell_id[cell_id] = cell_particle_count;
                    // Reset loop memory.
                    particle_index_rell_cell = 0;
                    cell_particle_count = 1;
                    ++cell_id;
                    previous_cell_index = cell_index;
                }
                // Store data for this particle.
                s_particle_indices_rel_cell[i] = particle_index_rell_cell++;
                s_cell_ids[i] = cell_id;
            }
            // Store data for the last cell.
            s_particle_count_per_cell_id[cell_id] = cell_particle_count;
        }
        // Wait for shared memory to be filled before reading.
        __syncthreads();
        particle_indices_rel_cell[global_index]
            = s_particle_indices_rel_cell[threadIdx.x];
        const auto cell_id = s_cell_ids[threadIdx.x];
        particle_count_per_cell[global_index]
            = s_particle_count_per_cell_id[cell_id];
    }
}

__global__
void thesis::shared_2d::charge_density(
    const FP *pos_x, const FP *pos_y, const size_t particle_count,
    const int3 grid_dimensions, const int cell_size,
    const int *particle_indices, const int *associated_cells,
    const int *indices_rel_cell, const int *particle_count_per_cell,
    FP *densities
) {
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto global_index = threadIdx.x + blockIdx.x * blockDim.x;
        global_index < particle_count;
        global_index += blockDim.x * gridDim.x
    ) {
        const auto particle_index = particle_indices[global_index];
        const auto cell_index = associated_cells[global_index];
        const auto index_rel_cell = indices_rel_cell[global_index];
        const auto cell_particle_count = particle_count_per_cell[global_index];

        // Position in world coordinates.
        const auto x = pos_x[particle_index];
        const auto y = pos_y[particle_index];
        const auto position = FP3{
            x, y, 0
        };
        // Position in grid coordinates with first cell center as origin.
        const auto [u, v, w] = cell_coordinates(position, cell_size);

        // 2D index of associated cell, i.e, surrounding cell closest to origin.
        const auto i = static_cast<int>(u);
        const auto j = static_cast<int>(v);

        // uvw-position relative to associated cell.
        const auto associated_cell_center = int3{ i, j, 0 };
        const auto pos_rel_cell = FP3{
            u - associated_cell_center.x,
            v - associated_cell_center.y,
            w - associated_cell_center.z
        };

        // Cell weights based on the distance to the particle.
        const auto cell_00_weight = (1 - pos_rel_cell.x) * (1 - pos_rel_cell.y);
        const auto cell_10_weight =      pos_rel_cell.x  * (1 - pos_rel_cell.y);
        const auto cell_01_weight = (1 - pos_rel_cell.x) *      pos_rel_cell.y;
        const auto cell_11_weight =      pos_rel_cell.x  *      pos_rel_cell.y;

        // Assumed size: 4 * blockDim.x.
        extern __shared__ FP s_densities[];
        // Helper to access 1D array using 2D index.
        const auto density = [&](const int row, const int column) -> FP & {
            return s_densities[column + row * blockDim.x];
        };

        // Weighted sum of the particle's charge for this particle.
        density(0, threadIdx.x) = cell_00_weight;
        density(1, threadIdx.x) = cell_10_weight;
        density(2, threadIdx.x) = cell_01_weight;
        density(3, threadIdx.x) = cell_11_weight;
        // Wait until the shared memory is filled.
        __syncthreads();

        // In-place reduction in shared memory.
        for (
            auto stride = ceil_pow2(cell_particle_count) / 2;
            stride > 0;
            stride /= 2
        ) {
            const auto i = index_rel_cell;
            // Make sure not to stride outside of the cell range. Crucial when
            // the number of particles in a cell isn't a power of two.
            if (i < stride && i + stride < cell_particle_count) {
                density(0, threadIdx.x) += density(0, threadIdx.x + stride);
                density(1, threadIdx.x) += density(1, threadIdx.x + stride);
                density(2, threadIdx.x) += density(2, threadIdx.x + stride);
                density(3, threadIdx.x) += density(3, threadIdx.x + stride);
            }
            __syncthreads();
        }
        
        // Linear cell indices.
        const auto cell_00_index = cell_index;
        const auto cell_10_index = cell_index + 1;
        const auto cell_01_index = cell_index + grid_dimensions.x;
        const auto cell_11_index = cell_index + grid_dimensions.x + 1;

        // Store reduction to global memory.
        if (index_rel_cell == 0) {
            atomicAdd(&densities[cell_00_index], density(0, threadIdx.x));
            atomicAdd(&densities[cell_10_index], density(1, threadIdx.x));
            atomicAdd(&densities[cell_01_index], density(2, threadIdx.x));
            atomicAdd(&densities[cell_11_index], density(3, threadIdx.x));
        }
    }
}
