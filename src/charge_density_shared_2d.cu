#include "charge_density_shared_2d.cuh"

#include <cub/device/device_radix_sort.cuh>

#include "common.cuh"

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
    const float *pos_x, const float *pos_y, const size_t particle_count,
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
        const auto position = float3{
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
