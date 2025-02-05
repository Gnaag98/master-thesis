#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "grid.cuh"
#include "particles.cuh"

const auto random_seed = 1u;

auto generate_particles_from_2d_pattern(
    const int3 simulation_dimensions, const int cell_size,
    const int particles_per_cell, const float particle_charge
) {
    /* 
     * The simulation box will be split into four zones that determine the
     * particle density from low to high density, all relative to the
     * user-specified number of particles per cell:
     * ┌────────────────────────────────┐
     * │ 3.            mid              │
     * ├────────┬───────────────────────┤
     * │ 1. low │ 2.   high->low        │
     * ├────────┴───────────────────────┤
     * │ 0.            mid              │
     * └────────────────────────────────┘
     * The following diagram shows the size of each zone using equally sized
     * boxes:
     * ┌───┬───┬───┬───┐
     * │ 3 │ 3 │ 3 │ 3 │
     * ├───┼───┼───┼───┤
     * │ 1 │ 2 │ 2 │ 2 │
     * ├───┼───┼───┼───┤
     * │ 1 │ 2 │ 2 │ 2 │
     * ├───┼───┼───┼───┤
     * │ 0 │ 0 │ 0 │ 0 │
     * └───┴───┴───┴───┘
     */
    auto random_engine = std::default_random_engine(random_seed);
    // Particle density distributions.
    auto low_density_distribution = std::uniform_int_distribution<int>(0, 4);
    auto mid_density_distribution = std::uniform_int_distribution<int>(
        0.5 * particles_per_cell, 1.5 * particles_per_cell
    );
    auto high_density_distribution = std::uniform_int_distribution<int>(
        1.5 * particles_per_cell, 2 * particles_per_cell
    );

    // Shorthand notations.
    const auto I = simulation_dimensions.x;
    const auto J = simulation_dimensions.y;

    auto particle_densities = std::vector<int>(I * J);
    auto particle_count = 0;
    
    // Mid density distribution (zone 0 and 3).
    for (auto j = 0; j < J; ++j) {
        for (auto i = 0; i < I; ++i) {
            const auto cell_index = i + j * I;
            // Skip zone 1 and 2.
            if (j >= J/4 && j < J * 3/4) {
                continue;
            }
            const auto cell_particle_count = mid_density_distribution(
                random_engine
            );
            particle_densities[cell_index] = cell_particle_count;
            particle_count += cell_particle_count;
        }
    }
    // Low distribution (zone 1).
    for (auto j = J / 4; j < J * 3/4; ++j) {
        for (auto i = 0; i < I / 4; ++i) {
            const auto cell_index = i + j * I;
            const auto cell_particle_count = low_density_distribution(
                random_engine
            );
            particle_densities[cell_index] = cell_particle_count;
            particle_count += cell_particle_count;
        }
    }
    // Linear gradient distribution (zone 2).
    for (auto j = J / 4; j < J * 3/4; ++j) {
        for (auto i = I / 4; i < I; ++i) {
            const auto cell_index = i + j * I;
            
            const auto mid_density = mid_density_distribution(random_engine);
            const auto high_density = high_density_distribution(random_engine);
            // Linear gradient from high (left) to low (right).
            const auto zone_start = I / 4;
            const auto zone_width = I * 3.0f/4.0f;
            const auto t = (i - zone_start) / zone_width;
            const auto cell_particle_count = static_cast<int>(
                t * mid_density + (1 - t) * high_density
            );

            particle_densities[cell_index] = cell_particle_count;
            particle_count += cell_particle_count;
        }
    }

    // Generate particles from particle densities.
    auto position_distribution = std::uniform_real_distribution<float>(
        0, cell_size
    );
    auto particles = amitis::HostParticles{ particle_count, particle_charge };
    auto particle_index = 0;
    for (auto j = 0; j < J; ++j) {
        const auto y_offset = j * cell_size;
        for (auto i = 0; i < I; ++i) {
            const auto x_offset = i * cell_size;
            const auto cell_index = i + j * I;
            const auto cell_particle_count = particle_densities[cell_index];
            for (auto p = 0; p < cell_particle_count; ++p) {
                particles.pos_x[particle_index] = x_offset
                    + position_distribution(random_engine);
                particles.pos_y[particle_index] = y_offset
                    + position_distribution(random_engine);
                ++particle_index;
            }
        }
    }

    return particles;
}

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

    // Unit charge.
    const auto particle_charge = 1.0f;
    // Number of outside layers of ghost cells.
    const auto ghost_layer_count = 1;

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

    // The complete grid includes ghost layers around the simulation grid.
    const auto grid_dimensions = int3{
        simulation_dimensions.x + 2 * ghost_layer_count,
        simulation_dimensions.y + 2 * ghost_layer_count,
        simulation_dimensions.z + 2 * ghost_layer_count
    };

    // Initialize particles.
    /* auto h_particles = generate_particles( */
    auto h_particles = generate_particles_from_2d_pattern(
        simulation_dimensions, cell_size, particles_per_cell, particle_charge
    );
    auto d_particles = DeviceParticles{ h_particles };

    std::cout << h_particles.pos_x.size() << " particles generated.\n";

    d_particles.copy(h_particles);
    // Initialize grid.
    auto h_charge_densities = HostGrid{ grid_dimensions };
    auto d_charge_densities = DeviceGrid{ grid_dimensions };

    // Run kernel.
    const auto particle_count = h_particles.pos_x.size();
    // XXX: Hardcoded block_size.
    const auto block_size = 128;
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
