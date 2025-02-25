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
        int3 simulation_dimensions, int cell_size, int particles_per_cell,
        int random_seed, ParticleDistribution distribution,
        std::optional<std::filesystem::path> positions_filepath
    ) -> HostParticles;
};

#endif
