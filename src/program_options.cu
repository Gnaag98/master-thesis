#include "program_options.cuh"

#include <charconv>
#include <sstream>
#include <string_view>
#include <system_error>
#include <vector>

#include <iostream>

namespace {
    enum class NextOptional {
        none,
        cell_size,
        distribution,
        output_directory,
        particles_per_cell,
        random_seed,
        version
    };
    enum class NextPositional {
        dim_x,
        dim_y,
        dim_z,
        none
    };
    // Helper to increment the enum until all positional arguments are parsed.
    NextPositional operator++(NextPositional &next) {
        if (next != NextPositional::none) {
            next = static_cast<NextPositional>(static_cast<int>(next) + 1);
        }
        return next;
    }

    auto _simulation_dimensions = int3{};
    auto _cell_size = std::optional<int>{};
    auto _distribution = std::optional<thesis::ParticleDistribution>{};
    // Distribution filepath set at the same time as the distribution.
    auto _distribution_filepath = std::optional<std::filesystem::path>{};
    auto _output_directory = std::optional<std::filesystem::path>{};
    auto _particles_per_cell = std::optional<int>{};
    auto _random_seed = std::optional<int>{};
    auto _version = std::optional<Version>{};

    auto to_int(std::string_view arg) {
        auto value = int{};
        const auto result = std::from_chars(
            arg.data(), arg.data() + arg.size(), value
        );
        if (result.ec == std::errc{}) {
            return value;
        } else {
            throw std::runtime_error{ "Error parsing int." };
        }
    };

    void parse_optional(std::string_view arg, NextOptional &next) {
        switch (next) {

        case NextOptional::none:
            break;

        case NextOptional::cell_size:
            try {
                _cell_size = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("cell_size must be an int.");
            }
            next = NextOptional::none;
            break;

        case NextOptional::distribution:
            if (arg == "uniform") {
                _distribution = thesis::ParticleDistribution::uniform;
            } else if (arg == "pattern_2d") {
                _distribution = thesis::ParticleDistribution::pattern_2d;
            } else if (
                const auto path = std::filesystem::path{ arg };
                std::filesystem::is_regular_file(path)
            ) {
                _distribution = thesis::ParticleDistribution::file;
                _distribution_filepath = path;
            } else {
                auto message = std::stringstream{};
                message << "Invalid distribution.";
                message << " Specify \"uniform\", \"pattern_2d\" or the path";
                message << " to a file containing particle starting positions.";
                throw std::runtime_error(message.str());
            }
            next = NextOptional::none;
            break;

        case NextOptional::output_directory:
            if (
                const auto path = std::filesystem::path{ arg };
                std::filesystem::is_directory(path)
            ) {
                _output_directory = path;
            } else {
                throw std::runtime_error("Invalid output directory.");
            }
            next = NextOptional::none;
            break;

        case NextOptional::particles_per_cell:
            try {
                _particles_per_cell = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("particles_per_cell must be an int.");
            }
            next = NextOptional::none;
            break;

        case NextOptional::random_seed:
            try {
                _random_seed = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("random_seed must be an int.");
            }
            next = NextOptional::none;
            break;

        case NextOptional::version:
            if (arg == "global") {
                _version = Version::global;
            } else if (arg == "shared") {
                _version = Version::shared;
            } else {
                throw std::runtime_error(
                    "Invalid version. Specify either \"global\" or \"shared\"."
                );
            }
            next = NextOptional::none;
            break;
        
        default: {
            auto message = std::stringstream{};
            message << "Unhandled optional parameter in switch. arg: " << arg;
            message << ". enum: " << static_cast<int>(next) << '.';
            throw std::runtime_error(message.str());
        }
        }
    }
};

void thesis::program_options::parse(int argc, char *argv[]) {
    const auto args = std::vector<std::string_view>(argv + 1, argv + argc);

    auto next_optional = NextOptional::none;
    auto next_positional = NextPositional{};

    for (auto arg : args) {
        using namespace std::string_literals;

        // If a flag has been parsed, parse the following argument.
        if (next_optional != NextOptional::none) {
            parse_optional(arg, next_optional);
            continue;
        }

        // Parse flags.
        if (arg == "-c") {
            next_optional = NextOptional::cell_size;
            continue;
        } else if (arg == "-d") {
            next_optional = NextOptional::distribution;
            continue;
        } else if (arg == "-o") {
            next_optional = NextOptional::output_directory;
            continue;
        } else if (arg == "-p") {
            next_optional = NextOptional::particles_per_cell;
            continue;
        } else if (arg == "-r") {
            next_optional = NextOptional::random_seed;
            continue;
        } else if (arg == "-v") {
            next_optional = NextOptional::version;
            continue;
        }

        // Parse positional arguments.
        switch (next_positional) {

        case NextPositional::dim_x:
            try {
                _simulation_dimensions.x = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("dim_x must be an int.");
            }
            ++next_positional;
            break;

        case NextPositional::dim_y:
            try {
                _simulation_dimensions.y = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("dim_y must be an int.");
            }
            ++next_positional;
            break;

        case NextPositional::dim_z:
            try {
                _simulation_dimensions.z = to_int(arg);
            } catch (const std::runtime_error &exception) {
                throw std::runtime_error("dim_z must be an int.");
            }
            ++next_positional;
            break;

        case NextPositional::none: {
            auto message = std::stringstream{};
            message << "Unknown flag: " << arg;
            throw std::runtime_error(message.str());
        }

        default: {
            auto message = std::stringstream{};
            message << "Unhandled positional argument: ";
            message << static_cast<int>(next_positional) << '.';
            throw std::runtime_error(message.str());
        }
        }
    }

    // Make sure that all positional (mandatory) arguments were parsed.
    if (next_positional != NextPositional::none) {
        auto message = std::stringstream{};
        message << "Positional arguments missing. Expected ";
        switch (next_positional) {

        case NextPositional::dim_x:
            message << "dim_x.";
            break;
            
        case NextPositional::dim_y:
            message << "dim_y.";
            break;
            
        case NextPositional::dim_z:
            message << "dim_z.";
            break;
        
        default:
            message << "argument is not handled in switch: ";
            message << static_cast<int>(next_positional) << '.';
            break;
        }
        throw std::runtime_error(message.str());
    }

    if (next_optional != NextOptional::none) {
        throw std::runtime_error("Flag specified without succeeding argument.");
    }
}

auto thesis::program_options::simulation_dimensions() -> int3 {
    return _simulation_dimensions;
}

auto thesis::program_options::cell_size() -> std::optional<int> {
    return _cell_size;
}

auto thesis::program_options::distribution()
        -> std::optional<ParticleDistribution> {
    return _distribution;
}

auto thesis::program_options::distribution_filepath()
        -> std::optional<std::filesystem::path> {
    return _distribution_filepath;
}

auto thesis::program_options::output_directory()
        -> std::optional<std::filesystem::path> {
    return _output_directory;
}

auto thesis::program_options::particles_per_cell() -> std::optional<int> {
    return _particles_per_cell;
}

auto thesis::program_options::random_seed() -> std::optional<int> {
    return _random_seed;
}

auto thesis::program_options::version() -> std::optional<Version> {
    return _version;
}
