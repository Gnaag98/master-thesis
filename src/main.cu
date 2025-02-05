#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "grid.cuh"
#include "particles.cuh"

const auto random_seed = 1u;
const auto block_size = 128;

// Unit charge.
const auto particle_charge = 1.0f;
// Number of outside layers of ghost cells.
const auto ghost_layer_count = 1;

auto generate_particles(
    const int3 simulation_dimensions, const int cell_size,
    const int particles_per_cell, const float particle_charge
) -> amitis::HostParticles {
    auto random_engine = std::default_random_engine(random_seed);
    auto distribution_x = std::uniform_real_distribution<float>(
        0, cell_size
    );
    auto distribution_y = std::uniform_real_distribution<float>(
        0, cell_size
    );
    auto distribution_z = std::uniform_real_distribution<float>(
        0, cell_size
    );

    const auto particle_count = particles_per_cell * simulation_dimensions.x
                                                   * simulation_dimensions.y
                                                   * simulation_dimensions.z;
    auto particles = amitis::HostParticles{ particle_count, particle_charge };

    auto particle_index = 0;
    for (auto k = 0; k < simulation_dimensions.z; ++k) {
        const auto z_offset = k * cell_size;
        for (auto j = 0; j < simulation_dimensions.y; ++j) {
            const auto y_offset = j * cell_size;
            for (auto i = 0; i < simulation_dimensions.x; ++i) {
                const auto x_offset = i * cell_size;
                for (auto p = 0; p < particles_per_cell; ++p) {
                    particles.pos_x[particle_index] = x_offset
                        + distribution_x(random_engine);
                    particles.pos_y[particle_index] = y_offset
                        + distribution_y(random_engine);
                    particles.pos_z[particle_index] = z_offset
                        + distribution_z(random_engine);
                    ++particle_index;
                }
            }
        }
    }

    return particles;
}

constexpr auto cell_coordinates(const float3 position, const int cell_size) {
    // XXX: Hardcoded half-cell shift due to one layer of ghost cells.
    return float3{
        position.x / cell_size + 0.5f,
        position.y / cell_size + 0.5f,
        position.z / cell_size + 0.5f
    };
}

constexpr auto cell_index(const int3 cell_center,
        const int3 grid_dimensions) {
    const auto i = cell_center.x;
    const auto j = cell_center.y;
    const auto k = cell_center.z;
    return i + (j * grid_dimensions.x)
             + (k * grid_dimensions.x * grid_dimensions.y);
}

__global__
void charge_density_global_2d(const float *pos_x, const float *pos_y,
        const uint particle_count, float particle_charge, float *densities,
        const int3 grid_dimensions, const int cell_size) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    const auto position = float3{ pos_x[index], pos_y[index], 0 };
    const auto [ u, v, w ] = cell_coordinates(position, cell_size);

    // Center of surrounding cell closest to the origin.
    const auto i = static_cast<int>(floor(u));
    const auto j = static_cast<int>(floor(v));
    const auto k = static_cast<int>(floor(w));

    // Centers of all surrounding cells, named relative the indices (i,j,k) of
    // the surrounding cell closest to the origin (cell_000).
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

int main(int argc, char *argv[]) {
    using namespace amitis;

    if (argc < 7) {
        std::cerr << "Usage: master_thesis dim_x dim_y dim_z cell_size"
            " particles/cell output_directory\n";
        return 1;
    }

    const auto simulation_dimensions = int3{
        std::stoi(argv[1]),
        std::stoi(argv[2]),
        std::stoi(argv[3])
    };
    const auto cell_size = std::stoi(argv[4]);
    const auto particles_per_cell = std::stoi(argv[5]);
    const auto output_directory_name = argv[6];

    const auto particle_count = particles_per_cell * simulation_dimensions.x
                                                   * simulation_dimensions.y
                                                   * simulation_dimensions.z;
    // The complete grid includes ghost layers around the simulation grid.
    const auto grid_dimensions = int3{
        simulation_dimensions.x + 2 * ghost_layer_count,
        simulation_dimensions.y + 2 * ghost_layer_count,
        simulation_dimensions.z + 2 * ghost_layer_count
    };

    // Initialize particles.
    auto h_particles = generate_particles(
        simulation_dimensions, cell_size, particles_per_cell, particle_charge
    );
    auto d_particles = DeviceParticles{ h_particles };

    d_particles.copy(h_particles);
    // Initialize grid.
    auto h_charge_densities = HostGrid{ grid_dimensions };
    auto d_charge_densities = DeviceGrid{ grid_dimensions };

    // Run kernel.
    const auto block_count = (particle_count + block_size - 1) / block_size;
    charge_density_global_2d<<<block_count, block_size>>>(
        d_particles.pos_x, d_particles.pos_y, particle_count, particle_charge,
        d_charge_densities.cells, grid_dimensions, cell_size
    );

    // Copy data from the device to the host.
    h_particles.copy(d_particles);
    h_charge_densities.copy(d_charge_densities);

    // Save data to disk.
    const auto output_directory = std::filesystem::path(output_directory_name);
    std::filesystem::create_directory(output_directory);
    h_particles.save_positions(output_directory / "positions.csv");
    h_charge_densities.save(output_directory / "charge_densities.csv");
}
