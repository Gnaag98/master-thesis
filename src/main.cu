#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "charge_density_global_2d.cuh"
#include "charge_density_shared_2d.cuh"
#include "common.cuh"
#include "grid.cuh"
#include "int_array.cuh"
#include "particles.cuh"
#include "particle_generation.cuh"
#include "timer.cuh"

enum class Version {
    global = 0,
    shared
};

int main(int argc, char *argv[]) {
    using namespace thesis;

    // Unit charge.
    const auto particle_charge = 1.0f;
    // Number of outside layers of ghost cells.
    const auto ghost_layer_count = 1;
    // XXX: Back to being hardcoded.
    const auto random_seed = 1;

    if (argc < 8) {
        std::cerr << "Usage: "<< argv[0] <<
            " dim_x dim_y dim_z cell_size particles/cell version"
            " output_directory particle_distribution should_save"
            " [positions_filename]\n";
        return 1;
    }

    const auto simulation_dimensions = int3{
        std::stoi(argv[1]),
        std::stoi(argv[2]),
        std::stoi(argv[3])
    };
    const auto cell_size = std::stoi(argv[4]);
    const auto particles_per_cell = std::stoi(argv[5]);
    const auto selected_version = Version{ std::stoi(argv[6]) };
    const auto output_directory_name = argv[7];
    const auto particle_distribution = ParticleDistribution{
        std::stoi(argv[8])
    };
    const auto should_save = std::stoi(argv[9]);

    const auto positions_filepath = (
        argc > 10 ? std::filesystem::path{ argv[10] } : ""
    );
    if (particle_distribution == ParticleDistribution::file
        && positions_filepath.empty()
    ) {
        std::cerr << "Filename needed when generating from file.\n";
        return 1;
    }

    const auto version_name = (
        (selected_version == Version::global) ? "Global" : "Shared"
    );
    std::cout << version_name << '\n';

    // The complete grid includes ghost layers around the simulation grid.
    const auto grid_dimensions = int3{
        simulation_dimensions.x + 2 * ghost_layer_count,
        simulation_dimensions.y + 2 * ghost_layer_count,
        simulation_dimensions.z + 2 * ghost_layer_count
    };

    // Initialize particles.
    auto h_particles = generate_particles(
        simulation_dimensions, cell_size, particles_per_cell, particle_charge,
        random_seed, particle_distribution, positions_filepath
    );
    auto d_particles = DeviceParticles{ h_particles };
    d_particles.copy(h_particles);
    const auto particle_count = h_particles.pos_x.size();

    // Kernel block settings.
#ifndef DEBUG
    const auto block_count = (particle_count + block_size - 1) / block_size;
#else
    const auto block_count = 1;
#endif
    std::cout << " <<<" << block_count << ", " << block_size << ">>>\n";

    std::cout << h_particles.pos_x.size() << " particles generated.\n";

    // Initialize grid.
    auto h_charge_densities = HostGrid{ grid_dimensions };
    auto d_charge_densities = DeviceGrid{ grid_dimensions };

    // Prepare shared version, even if it is not used.
    auto h_particle_indices_before = HostIntArray{ particle_count };
    auto h_particle_indices_after = HostIntArray{ particle_count };
    auto h_particle_cell_indices_before = HostIntArray{ particle_count };
    auto h_particle_cell_indices_after = HostIntArray{ particle_count };
    auto h_particle_indices_rel_cell = HostIntArray{ particle_count };
    auto h_particle_count_per_cell = HostIntArray{ particle_count };
    
    auto d_particle_indices_before = DeviceIntArray{ h_particle_indices_before };
    auto d_particle_indices_after = DeviceIntArray{ h_particle_indices_after };
    auto d_particle_cell_indices_before = DeviceIntArray{ h_particle_cell_indices_before };
    auto d_particle_cell_indices_after = DeviceIntArray{ h_particle_cell_indices_after };
    auto d_particle_indices_rel_cell = DeviceIntArray{ h_particle_indices_rel_cell };
    auto d_particle_count_per_cell = DeviceIntArray{ h_particle_count_per_cell };
    
    void *sort_storage = nullptr;
    auto sort_storage_size = size_t{};
    
    // Particle indices [0, 1, 2, ...] only need to be initialized once.
    shared_2d::initialize_particle_indices<<<block_count, block_size>>>(
        particle_count, d_particle_indices_before.i
    );
    // Run the sorting with uninitialized sort storage to compute the
    // required temporary storage size.
    shared_2d::sort_particles_by_cell(
        sort_storage, sort_storage_size,
        d_particle_cell_indices_before.i, d_particle_cell_indices_after.i,
        d_particle_indices_before.i, d_particle_indices_after.i, particle_count
    );
    // Allocate sort storage.
    cudaMalloc(&sort_storage, sort_storage_size);
    

    // Limit the lifetime of the timer using a scope.
    {
        auto kernel_timer = thesis::Timer{ version_name };
        // Run kernel.
        switch (selected_version) {
        case Version::global:
            using namespace global_2d;
            charge_density<<<block_count, block_size>>>(
                d_particles.pos_x, d_particles.pos_y, particle_count,
                particle_charge, grid_dimensions, cell_size,
                d_charge_densities.cells
            );
            break;
        case Version::shared: {
            using namespace shared_2d;
            initialize_particle_cell_indices<<<block_count, block_size>>>(
                d_particles.pos_x, d_particles.pos_y, particle_count,
                grid_dimensions, cell_size, d_particle_cell_indices_before.i
            );
            sort_particles_by_cell(
                sort_storage, sort_storage_size,
                d_particle_cell_indices_before.i,
                d_particle_cell_indices_after.i,
                d_particle_indices_before.i,
                d_particle_indices_after.i, particle_count
            );
            initialize_particle_occupancy<<<block_count, block_size>>>(
                particle_count, d_particle_cell_indices_after.i,
                d_particle_indices_rel_cell.i, d_particle_count_per_cell.i
            );
            charge_density<<<block_count, block_size>>>(
                d_particles.pos_x, d_particles.pos_y, particle_count,
                particle_charge, grid_dimensions, cell_size,
                d_particle_indices_after.i, d_particle_cell_indices_after.i,
                d_particle_indices_rel_cell.i, d_particle_count_per_cell.i,
                d_charge_densities.cells
            );
            break;
        }
        default:
            std::cerr << "Unsupported version number.\n";
            return 1;
        }

        // Wait for the kernel to finish.
        cudaDeviceSynchronize();
    }

    // Deallocate the sort storage.
    cudaFree(sort_storage);

    // Save data to disk.
    if (should_save) {
        h_particles.copy(d_particles);
        h_charge_densities.copy(d_charge_densities);

        std::cout << "Saving to disk.\n";
        const auto output_directory = std::filesystem::path(output_directory_name);
        std::filesystem::create_directory(output_directory);
        h_particles.save_positions(output_directory / "positions.npz");
        const auto densities_filename = ([selected_version](){
            auto filename = std::string("charge_densities");
            switch (selected_version) {
            case Version::global:
                filename += "_global";
                break;
            case Version::shared:
                filename += "_shared";
                break;
            }
            filename += ".npy";
            return filename;
        })();
        h_charge_densities.save(output_directory / densities_filename);
    }
    std::cout << "Done\n";
}
