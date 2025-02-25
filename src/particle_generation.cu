#include "particle_generation.cuh"

#include <algorithm>
#include <numeric>
#include <random>

namespace {
    /// Returns all indices in the range [0, count - 1) in random order.
    auto get_shuffled_indices(const int count, const int random_seed) {
        auto random_engine = std::default_random_engine(random_seed);
        auto indices = std::vector<int>(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), random_engine);
        return indices;
    }

    /// Randomly generates a specified number of particels per cell.
    auto generate_particles_uniformly(
        const int3 simulation_dimensions, const int cell_size,
        const int particles_per_cell, const int random_seed
    ) -> thesis::HostParticles {
        auto random_engine = std::default_random_engine(random_seed);
        auto position_distribution = std::uniform_real_distribution<float>(
            0, cell_size
        );

        const auto particle_count = particles_per_cell
            * simulation_dimensions.x
            * simulation_dimensions.y
            * simulation_dimensions.z;
        auto particles = thesis::HostParticles{ particle_count };
        const auto particle_indices = get_shuffled_indices(
            particle_count, random_seed
        );
        auto indirect_particle_index = 0;
        for (auto k = 0; k < simulation_dimensions.z; ++k) {
            const auto z_offset = k * cell_size;
            for (auto j = 0; j < simulation_dimensions.y; ++j) {
                const auto y_offset = j * cell_size;
                for (auto i = 0; i < simulation_dimensions.x; ++i) {
                    const auto x_offset = i * cell_size;
                    for (auto p = 0; p < particles_per_cell; ++p) {
                        const auto particle_index = particle_indices[
                            indirect_particle_index
                        ];
                        particles.pos_x[particle_index] = x_offset
                            + position_distribution(random_engine);
                        particles.pos_y[particle_index] = y_offset
                            + position_distribution(random_engine);
                        particles.pos_z[particle_index] = z_offset
                            + position_distribution(random_engine);
                        ++indirect_particle_index;
                    }
                }
            }
        }

        return particles;
    }

    /// Randomly generates particles based on a density pattern.
    auto generate_particles_from_2d_pattern(
        const int3 simulation_dimensions, const int cell_size,
        const int particles_per_cell, const int random_seed
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
        auto low_density_distribution = std::uniform_int_distribution<int>(
            0, 4
        );
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
                
                const auto mid_density = mid_density_distribution(
                    random_engine
                );
                const auto high_density = high_density_distribution(
                    random_engine
                );
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
        auto particles = thesis::HostParticles{ particle_count };
        const auto particle_indices = get_shuffled_indices(
            particle_count, random_seed
        );
        auto indirect_particle_index = 0;
        for (auto j = 0; j < J; ++j) {
            const auto y_offset = j * cell_size;
            for (auto i = 0; i < I; ++i) {
                const auto x_offset = i * cell_size;
                const auto cell_index = i + j * I;
                const auto cell_particle_count = particle_densities[cell_index];
                for (auto p = 0; p < cell_particle_count; ++p) {
                    const auto particle_index = particle_indices[
                        indirect_particle_index
                    ];
                    particles.pos_x[particle_index] = x_offset
                        + position_distribution(random_engine);
                    particles.pos_y[particle_index] = y_offset
                        + position_distribution(random_engine);
                    ++indirect_particle_index;
                }
            }
        }

        return particles;
    }

    /// Randomly generates a specified number of particels per cell.
    auto generate_particles_from_file(
        const std::filesystem::path positions_filepath
    ) -> thesis::HostParticles {
        auto particles = thesis::HostParticles{ positions_filepath };
        return particles;
    }
};

auto thesis::generate_particles (
    const int3 simulation_dimensions, const int cell_size,
    const int particles_per_cell, const int random_seed,
    const ParticleDistribution distribution,
    const std::optional<std::filesystem::path> positions_filepath
) -> thesis::HostParticles {
    switch (distribution) {
    case ParticleDistribution::uniform:
        return generate_particles_uniformly(
            simulation_dimensions, cell_size, particles_per_cell, random_seed
        );
    case ParticleDistribution::pattern_2d:
        return generate_particles_from_2d_pattern(
            simulation_dimensions, cell_size, particles_per_cell, random_seed
        );
    case ParticleDistribution::file:
        return generate_particles_from_file(*positions_filepath);
    
    default:
        throw std::runtime_error("Unhandled particle distribution enum option");
    }
}
