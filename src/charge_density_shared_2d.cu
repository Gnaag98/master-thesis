#include "charge_density_shared_2d.cuh"

#include "common.cuh"

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
