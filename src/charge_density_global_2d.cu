#include "charge_density_global_2d.cuh"

#include "common.cuh"

__global__
void thesis::global_2d::charge_density(
    const float *pos_x, const float *pos_y, const size_t particle_count,
    const float particle_charge, const int3 grid_dimensions,
    const int cell_size, float *densities
) {
    // Grid-stride loop. Equivalent to regular if-statement if grid is large
    // enough to cover all iterations of the loop.
    for (
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < particle_count;
        index += blockDim.x * gridDim.x
    ) {
        const auto position = float3{ pos_x[index], pos_y[index], 0 };
        const auto [ u, v, w ] = cell_coordinates(position, cell_size);

        // 2D index, or center of surrounding cell closest to the origin.
        const auto i = static_cast<int>(floor(u));
        const auto j = static_cast<int>(floor(v));

        // Centers of all surrounding cells, named relative the indices
        // (i,j,k) of the surrounding cell closest to the origin (cell_000).
        const auto cell_000_center = int3{ i,     j    , 0 };
        const auto cell_100_center = int3{ i + 1, j    , 0 };
        const auto cell_010_center = int3{ i,     j + 1, 0 };
        const auto cell_110_center = int3{ i + 1, j + 1, 0 };

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
        atomicAdd(&densities[cell_000_index], particle_charge * cell_000_weight);
        atomicAdd(&densities[cell_100_index], particle_charge * cell_100_weight);
        atomicAdd(&densities[cell_010_index], particle_charge * cell_010_weight);
        atomicAdd(&densities[cell_110_index], particle_charge * cell_110_weight);
    }
}
