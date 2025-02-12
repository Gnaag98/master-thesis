#include "charge_density_shared_2d.cuh"

#include <cub/device/device_radix_sort.cuh>

#include "common.cuh"

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
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < particle_count;
        index += blockDim.x * gridDim.x
    ) {
        indices[index] = index;
    }
}

__global__
void thesis::shared_2d::initialize_particle_cell_indices(
    const float *pos_x, const float *pos_y, const size_t particle_count,
    const int3 grid_dimensions, const int cell_size, int *cell_indices
) {
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
        particle_index < particle_count;
        particle_index += blockDim.x * gridDim.x
    ) {
        // Position in world coordinates.
        const auto position = float3{
            pos_x[particle_index], pos_y[particle_index], 0
        };
        // Position in grid coordinates.
        const auto [ u, v, w ] = cell_coordinates(position, cell_size);
        // 2D indices of first enclosing cell, by first meaning the one with
        // lowest index, i.e., closest to the origin.
        const auto i = static_cast<int>(floor(u));
        const auto j = static_cast<int>(floor(v));

        cell_indices[particle_index] = i + j * grid_dimensions.x;
    }
}

void thesis::shared_2d::sort_particles_by_cell(
    void *sort_storage, size_t &sort_storage_size,
    int *particle_cell_indices_before, int *particle_cell_indices_after,
    int *particle_indices_before, int *particle_indices_after,
    const size_t particle_count
) {
    cub::DeviceRadixSort::SortPairs(
        sort_storage, sort_storage_size,
        particle_cell_indices_before, particle_cell_indices_after,
        particle_indices_before, particle_indices_after,
        particle_count
    );
}

__global__
void thesis::shared_2d::initialize_particle_occupancy(
    const size_t particle_count, const int *cell_indices,
    int *particle_indices_rel_cell, int *particle_count_per_cell
) {
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < particle_count;
        index += blockDim.x * gridDim.x
    ) {
        __shared__ int s_cell_indices[block_size];
        __shared__ int s_particle_indices_rel_cell[block_size];
        // Incrementing 0, 1, 2, ..., for each new cell.
        __shared__ uint s_cell_ids[block_size];
        // Indexed with cell id, not cell index.
        __shared__ uint s_particle_count_per_cell_id[block_size];

        s_cell_indices[threadIdx.x] = cell_indices[index];
        __syncthreads();

        // TODO: Parallelize.
        if (threadIdx.x == 0) {
            s_particle_indices_rel_cell[0] = 0;
            s_cell_ids[0] = 0;
            // Start on 1 since we already set the value for i = 0.
            auto particle_index_rel_cell = 1;
            auto cell_id = 0;
            auto cell_particle_count = 0;
            auto previous_cell_index = s_cell_indices[0];
            // Don't iterate too far in the last block.
            const auto block_particle_count = min(
                particle_count - index, static_cast<size_t>(block_size)
            );
            for (auto i = 1uz; i < block_particle_count; ++i) {
                const auto cell_index = s_cell_indices[i];
                ++cell_particle_count;
                if (cell_index > previous_cell_index) {
                    s_particle_count_per_cell_id[cell_id] = cell_particle_count;
                    particle_index_rel_cell = 0;
                    ++cell_id;
                    cell_particle_count = 0;
                    previous_cell_index = cell_index;
                }
                s_particle_indices_rel_cell[i] = particle_index_rel_cell++;
                s_cell_ids[i] = cell_id;
            }
            s_particle_count_per_cell_id[cell_id] = cell_particle_count + 1;
        }
        __syncthreads();
        particle_indices_rel_cell[index] = s_particle_indices_rel_cell[threadIdx.x];
        const auto cell_index = s_cell_ids[threadIdx.x];
        particle_count_per_cell[index] = s_particle_count_per_cell_id[cell_index];
    }
}

__global__
void thesis::shared_2d::charge_density(
    const float *pos_x, const float *pos_y, const size_t particle_count,
    const float particle_charge, const int3 grid_dimensions,
    const int cell_size, const int *particle_indices,
    const int *particle_cell_indices, const int *particle_indices_rel_cell,
    const int *particle_count_per_cell, float *densities
) {
    // Grid-stride loop. Equivalent to regular if-statement grid is large enough
    // to cover all iterations of the loop.
    for (
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < particle_count;
        index += blockDim.x * gridDim.x
    ) {
        // Each particle will contribute to its 4 surrounding cells.
        __shared__ float s_densities[4][block_size];

        // 1D index of first enclosing cell, i.e., with lowest index.
        const auto first_cell_index = particle_cell_indices[index];
        const auto particle_index = particle_indices[index];
        const auto particle_index_rel_cell = particle_indices_rel_cell[index];
        const auto cell_particle_count = particle_count_per_cell[index];

        // Convert 1D index to 2D.
        const auto i = first_cell_index % grid_dimensions.x;
        const auto j = first_cell_index / grid_dimensions.x;

        // Centers of all surrounding cells, named relative the indices (i,j,k)
        // of the surrounding cell closest to the origin (cell_000).
        const auto cell_000_center = int3{ i,     j    , 0 };
        const auto cell_100_center = int3{ i + 1, j    , 0 };
        const auto cell_010_center = int3{ i,     j + 1, 0 };
        const auto cell_110_center = int3{ i + 1, j + 1, 0 };

        const auto position = float3{
            pos_x[particle_index], pos_y[particle_index], 0
        };
        const auto [ u, v, w ] = cell_coordinates(position, cell_size);
        // uvw-position relative to cell_000.
        const auto pos_rel_cell = float3{
            u - cell_000_center.x,
            v - cell_000_center.y,
            w - cell_000_center.z
        };
        // Cell weights based on the distance to the particle.
        const auto cell_000_weight = (1 - pos_rel_cell.x) * (1 - pos_rel_cell.y);
        const auto cell_100_weight =      pos_rel_cell.x  * (1 - pos_rel_cell.y);
        const auto cell_010_weight = (1 - pos_rel_cell.x) *      pos_rel_cell.y;
        const auto cell_110_weight =      pos_rel_cell.x  *      pos_rel_cell.y;

        // Linear cell indices.
        const auto cell_000_index = cell_index(cell_000_center, grid_dimensions);
        const auto cell_100_index = cell_index(cell_100_center, grid_dimensions);
        const auto cell_010_index = cell_index(cell_010_center, grid_dimensions);
        const auto cell_110_index = cell_index(cell_110_center, grid_dimensions);

        // Weighted sum of the particle's charge.
        s_densities[0][threadIdx.x] = particle_charge * cell_000_weight;
        s_densities[1][threadIdx.x] = particle_charge * cell_100_weight;
        s_densities[2][threadIdx.x] = particle_charge * cell_010_weight;
        s_densities[3][threadIdx.x] = particle_charge * cell_110_weight;
        // Wait until the shared memory is filled.
        __syncthreads();

        // In-place reduction in shared memory.
        for (
            auto stride = ceil_pow2(cell_particle_count) / 2;
            stride > 0;
            stride /= 2
        ) {
            // Shorthand notation.
            const auto i = particle_index_rel_cell;
            // Make sure not to stride outside of the cell range. Crucial when
            // the number of particles in a cell isn't a power of two.
            if (i < stride && i + stride < cell_particle_count) {
                s_densities[0][threadIdx.x] += s_densities[0][threadIdx.x + stride];
                s_densities[1][threadIdx.x] += s_densities[1][threadIdx.x + stride];
                s_densities[2][threadIdx.x] += s_densities[2][threadIdx.x + stride];
                s_densities[3][threadIdx.x] += s_densities[3][threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Store reduction to global memory.
        if (particle_index_rel_cell == 0) {
            atomicAdd(&densities[cell_000_index], s_densities[0][threadIdx.x]);
            atomicAdd(&densities[cell_100_index], s_densities[1][threadIdx.x]);
            atomicAdd(&densities[cell_010_index], s_densities[2][threadIdx.x]);
            atomicAdd(&densities[cell_110_index], s_densities[3][threadIdx.x]);
        }
    }
}
