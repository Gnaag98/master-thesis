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
// XXX: Hardcoded dimensions, and one layer of ghost cells added manually.
constexpr auto grid_dimensions = dim3{ 4+2, 2+2, 1+2 };
// Side length of cubic cell.
constexpr auto cell_size = 64;
// XXX: Hardcoded to account for one layer of ghost cells.
constexpr auto space_dimensions = dim3{
    (grid_dimensions.x - 2) * cell_size,
    (grid_dimensions.y - 2) * cell_size,
    (grid_dimensions.z - 2) * cell_size
};

void distribute(amitis::HostParticles &particles) {
    auto random_engine = std::default_random_engine(random_seed);

    auto distribution_x = std::uniform_real_distribution<float>(
        0, space_dimensions.x
    );
    auto distribution_y = std::uniform_real_distribution<float>(
        0, space_dimensions.y
    );
    auto distribution_z = std::uniform_real_distribution<float>(
        0, space_dimensions.z
    );

    for (auto i = 0; i < particles.pos_x.size(); ++i) {
        particles.pos_x[i] = distribution_x(random_engine);
        particles.pos_y[i] = distribution_y(random_engine);
        particles.pos_z[i] = distribution_z(random_engine);
    }
}

constexpr auto to_cell_coordinates(const float3 position) {
    // XXX: Hardcoded half-cell shift due to one layer of ghost cells.
    return float3{
        position.x / cell_size + 0.5f,
        position.y / cell_size + 0.5f,
        position.z / cell_size + 0.5f
    };
}

constexpr auto get_cell_index(const int3 cell_center) {
    const auto i = cell_center.x;
    const auto j = cell_center.y;
    const auto k = cell_center.z;
    return i + (j * grid_dimensions.x)
             + (k * grid_dimensions.x * grid_dimensions.y);
}

__global__
void charge_density_global_2d(const float *pos_x, const float *pos_y,
        const uint particle_count, float particle_charge, float *densities) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    const auto position = float3{ pos_x[index], pos_y[index], 0 };
    const auto [ u, v, w ] = to_cell_coordinates(position);

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
    const auto cell_000_index = get_cell_index(cell_000_center);
    const auto cell_100_index = get_cell_index(cell_100_center);
    const auto cell_010_index = get_cell_index(cell_010_center);
    const auto cell_110_index = get_cell_index(cell_110_center);

    // Weighted sum of the particle's charge.
    atomicAdd(&densities[cell_000_index], particle_charge * cell_000_weight);
    atomicAdd(&densities[cell_100_index], particle_charge * cell_100_weight);
    atomicAdd(&densities[cell_010_index], particle_charge * cell_010_weight);
    atomicAdd(&densities[cell_110_index], particle_charge * cell_110_weight);
}

int main(int argc, char *argv[]) {
    using namespace amitis;

    if (argc < 3) {
        std::cerr << "Usage: master_thesis particle_count output_directory\n";
        return 1;
    }

    const auto particle_count = std::stoi(argv[1]);
    const auto output_directory_name = argv[2];
    
    // Initialize particles.
    auto h_particles = HostParticles{ particle_count, particle_charge };
    auto d_particles = DeviceParticles{ h_particles };
    distribute(h_particles);
    d_particles.copy(h_particles);
    // Initialize grid.
    auto h_charge_densities = HostGrid{ grid_dimensions };
    auto d_charge_densities = DeviceGrid{ grid_dimensions };

    // Run kernel.
    const auto block_count = (particle_count + block_size - 1) / block_size;
    charge_density_global_2d<<<block_count, block_size>>>(
        d_particles.pos_x, d_particles.pos_y, particle_count, particle_charge,
        d_charge_densities.cells
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
