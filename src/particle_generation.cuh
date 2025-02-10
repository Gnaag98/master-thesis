#ifndef THESIS_PARTICLE_GENERATION_HPP
#define THESIS_PARTICLE_GENERATION_HPP

#include "particles.cuh"

namespace thesis {
    /// Methods of particle generation.
    enum class ParticleDistribution {
        uniform,
        pattern_2d
    };

    /// Generates particles based on a specified distribution.
    auto generate_particles(
        const int3 simulation_dimensions, const int cell_size,
        const int particles_per_cell, const float particle_charge,
        const int random_seed, const ParticleDistribution distribution
    ) -> HostParticles;
};

#endif
