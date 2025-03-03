#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <time.h>

#include "charge_density_global_2d.cuh"
#include "charge_density_shared_2d.cuh"
#include "common.cuh"
#include "grid.cuh"
#include "int_array.cuh"
#include "particles.cuh"
#include "particle_generation.cuh"
#include "program_options.cuh"
#include "timer.cuh"

int main(int argc, char *argv[]) {
    using namespace thesis;

    // Number of outside layers of ghost cells.
    const auto ghost_layer_count = 1;

    // Default values for program arguments.
    const auto default_cell_size = 64;
    const auto default_particle_distribution = ParticleDistribution::uniform;
    const auto default_particles_per_cell = 16;
    const auto default_seed = static_cast<int>(time(nullptr));
    const auto default_version = Version::global;

    try {
        program_options::parse(argc, argv);
    } catch (const std::runtime_error &exception) {
        std::cerr << exception.what() << '\n';
        std::cerr << "usage " << argv[0] << " dim_x dim_y dim_z";
        std::cerr << " [-c cell_size] [-d distribution ] [-o output_directory]";
        std::cerr << " [-p particles_per_cell] [-r random_seed] [-v version]\n";
        return 1;
    }

    const auto simulation_dimensions = program_options::simulation_dimensions();
    const auto cell_size = (
        program_options::cell_size().value_or(default_cell_size)
    );
    const auto particle_distribution = (
        program_options::distribution().value_or(default_particle_distribution)
    );
    const auto positions_filepath = program_options::distribution_filepath();
    const auto output_directory = program_options::output_directory();
    const auto particles_per_cell = (
        program_options::particles_per_cell().value_or(
            default_particles_per_cell
        )
    );
    const auto random_seed = (
        program_options::random_seed().value_or(default_seed)
    );
    const auto version = (
        program_options::version().value_or(default_version)
    );

    const auto version_name = (
        (version == Version::global) ? "Global" : "Shared"
    );
    std::cout << version_name << " version with seed " << random_seed << '\n';

    // The complete grid includes ghost layers around the simulation grid.
    const auto grid_dimensions = int3{
        simulation_dimensions.x + 2 * ghost_layer_count,
        simulation_dimensions.y + 2 * ghost_layer_count,
        simulation_dimensions.z + 2 * ghost_layer_count
    };

    // Initialize particles.
    auto h_particles = generate_particles(
        simulation_dimensions, cell_size, particles_per_cell,
        random_seed, particle_distribution, positions_filepath
    );
    auto d_particles = DeviceParticles{ h_particles };
    d_particles.copy(h_particles);
    const auto particle_count = h_particles.pos_x.size();

    // Kernel block settings.
#ifndef DEBUG
    const auto block_size = 128;
    const auto block_count = (particle_count + block_size - 1) / block_size;
#else
    const auto block_size = 1;
    const auto block_count = 1;
#endif
    std::cout << " <<<" << block_count << ", " << block_size << ">>>\n";

    std::cout << h_particles.pos_x.size() << " particles generated.\n";

    // Initialize grid.
    auto h_charge_densities = HostGrid{ grid_dimensions };
    auto d_charge_densities = DeviceGrid{ grid_dimensions };

    // Shared: initialize particle indices.
    auto h_particle_indices = HostIntArray{ particle_count };
    auto d_particle_indices = DeviceIntArray{
        h_particle_indices
    };
    shared_2d::initialize_particle_indices<<<block_count, block_size>>>(
        particle_count, d_particle_indices.i
    );

    // Shared: Associated cell index for each particle.
    auto h_associated_cells = HostIntArray{ particle_count };
    auto d_associated_cells = DeviceIntArray{
        h_associated_cells
    };

    // Shared: initialize sorting.
    auto h_particle_indices_sorted = HostIntArray{ particle_count };
    auto d_particle_indices_sorted = DeviceIntArray{
        h_particle_indices_sorted
    };
    auto h_associated_cells_sorted = HostIntArray{ particle_count };
    auto d_associated_cells_sorted = DeviceIntArray{
        h_associated_cells_sorted
    };
    void *sort_storage = nullptr;
    auto sort_storage_size = size_t{};
    shared_2d::sort_particles_by_cell(
        sort_storage, sort_storage_size,
        d_associated_cells.i, d_associated_cells_sorted.i,
        d_particle_indices.i, d_particle_indices_sorted.i,
        particle_count
    );
    cudaMalloc(&sort_storage, sort_storage_size);

    // Shared: Block data for the density kernel.
    auto h_particle_indices_rel_cell = HostIntArray{ particle_count };
    auto d_particle_indices_rel_cell = DeviceIntArray{
        h_particle_indices_rel_cell
    };
    auto h_particle_count_per_cell = HostIntArray{ particle_count };
    auto d_particle_count_per_cell = DeviceIntArray{
        h_particle_count_per_cell
    };

    // Limit the lifetime of the timer using a scope.
    {
        // Wait for any previous kernel to finish before starting the timer.
        cudaDeviceSynchronize();
        const auto kernel_timer = Timer{ version_name };
        // Run kernel.
        switch (version) {
        case Version::global: {
            using namespace global_2d;
            charge_density<<<block_count, block_size>>>(
                d_particles.pos_x, d_particles.pos_y, particle_count,
                grid_dimensions, cell_size,
                d_charge_densities.cells
            );
            break;
        }
        case Version::shared: {
            using namespace shared_2d;
            {
                //const auto timer = Timer{ "Associate" };
                associate_particles_with_cells<<<block_count, block_size>>>(
                    d_particles.pos_x, d_particles.pos_y, particle_count, grid_dimensions,
                    cell_size, d_associated_cells.i
                );
                //cudaDeviceSynchronize();
            }
            {
                //const auto timer = Timer{ "Sort" };
                sort_particles_by_cell(
                    sort_storage, sort_storage_size,
                    d_associated_cells.i, d_associated_cells_sorted.i,
                    d_particle_indices.i, d_particle_indices_sorted.i,
                    particle_count
                );
                //cudaDeviceSynchronize();
            }
            {
                //const auto timer = Timer{ "Contextualize" };
                contextualize_cell_associations<<<
                    block_count,
                    block_size,
                    4 * block_size * sizeof(int)
                >>>(
                    particle_count, d_associated_cells_sorted.i,
                    d_particle_indices_rel_cell.i, d_particle_count_per_cell.i
                );
                //cudaDeviceSynchronize();
            }
            {
                //const auto timer = Timer{ "Density" };
                charge_density<<<
                    block_count,
                    block_size,
                    4 * block_size * sizeof(float)
                >>>(
                    d_particles.pos_x, d_particles.pos_y, particle_count,
                    grid_dimensions, cell_size, d_particle_indices_sorted.i,
                    d_associated_cells_sorted.i, d_particle_indices_rel_cell.i,
                    d_particle_count_per_cell.i, d_charge_densities.cells
                );
                //cudaDeviceSynchronize();
            }
            break;
        }
        default:
            std::cerr << "Unsupported version number.\n";
            return 1;
        }

        // Wait for the kernel(s) to finish before destructing the timer.
        cudaDeviceSynchronize();
    }

    cudaFree(sort_storage);

    // Save data to disk.
    if (output_directory) {
        using namespace std::filesystem;
        h_particles.copy(d_particles);

        h_particle_indices.copy(d_particle_indices);
        h_associated_cells.copy(d_associated_cells);
        h_particle_indices_sorted.copy(d_particle_indices_sorted);
        h_associated_cells_sorted.copy(d_associated_cells_sorted);

        h_particle_indices_rel_cell.copy(d_particle_indices_rel_cell);
        h_particle_count_per_cell.copy(d_particle_count_per_cell);

        h_charge_densities.copy(d_charge_densities);

        std::cout << "Saving to disk.\n";
        h_particles.save(*output_directory / "positions.npz");
        

        h_particle_indices.save(*output_directory / "particle_indices.npy");
        h_associated_cells.save(*output_directory / "associated_cells.npy");
        h_particle_indices_sorted.save(*output_directory / "particle_indices_sorted.npy");
        h_associated_cells_sorted.save(*output_directory / "associated_cells_sorted.npy");

        h_particle_indices_rel_cell.save(*output_directory / "particle_indices_rel_cell.npy");
        h_particle_count_per_cell.save(*output_directory / "particle_count_per_cell.npy");

        const auto densities_filename = ([version](){
            auto filename = std::string("charge_densities");
            switch (version) {
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
        h_charge_densities.save(*output_directory / densities_filename);
    }
    std::cout << "Done\n";
}
