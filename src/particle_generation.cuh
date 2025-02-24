#ifndef THESIS_PARTICLE_GENERATION_CUH
#define THESIS_PARTICLE_GENERATION_CUH

#include <filesystem>
#include <optional>

#include "particles.cuh"

namespace thesis {
    /// Methods of particle generation.
    enum class ParticleDistribution {
        uniform = 0,
        pattern_2d,
        file
    };

    /// Generates particles based on a specified distribution.
    auto generate_particles(
        const int3 simulation_dimensions, const int cell_size,
        const int particles_per_cell, const float particle_charge,
        const int random_seed, const ParticleDistribution distribution,
        std::optional<std::filesystem::path> positions_filepath
    ) -> HostParticles;
};

#endif
