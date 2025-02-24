#ifndef THESIS_PROGRAM_OPTIONS_CUH
#define THESIS_PROGRAM_OPTIONS_CUH

#include <filesystem>
#include <optional>

#include "particle_generation.cuh"

enum class Version {
    global,
    shared
};

namespace thesis::program_options {
    void parse(int argc, char *argv[]);

    auto simulation_dimensions() -> int3;
    auto cell_size() -> std::optional<int>;
    auto distribution() -> std::optional<ParticleDistribution>;
    auto distribution_filepath() -> std::optional<std::filesystem::path>;
    auto output_directory() -> std::optional<std::filesystem::path>;
    auto particles_per_cell() -> std::optional<int>;
    auto random_seed() -> std::optional<int>;
    auto version() -> std::optional<Version>;
};

#endif
