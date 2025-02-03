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
// XXX: Hardcoded dimensions.
constexpr auto grid_dimensions = dim3{ 2, 4, 8 };
// Side length of cubic cell.
constexpr auto cell_size = 64;
constexpr auto space_dimensions = dim3{
    grid_dimensions.x * cell_size,
    grid_dimensions.x * cell_size,
    grid_dimensions.x * cell_size
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

constexpr auto x_to_u(const float x) {
    return x * (grid_dimensions.x / space_dimensions.x);
}
constexpr auto y_to_v(const float y) {
    return y * (grid_dimensions.y / space_dimensions.y);
}
constexpr auto z_to_w(const float z) {
    return z * (grid_dimensions.z / space_dimensions.z);
}

constexpr auto get_cell_index(const float u, const float v, const float w) {
    const auto i = static_cast<int>(floor(u));
    const auto j = static_cast<int>(floor(v));
    const auto k = static_cast<int>(floor(w));
    return i + (j * grid_dimensions.x)
             + (k * grid_dimensions.x * grid_dimensions.y);
}

__global__
void count_particles(const float *pos_x, const float *pos_y, const float *pos_z,
        const uint particle_count, float *cells) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    const auto x = pos_x[index];
    const auto y = pos_y[index];
    const auto z = pos_z[index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);
    const auto w = z_to_w(z);

    const auto cell_index = get_cell_index(u, v, w);

    atomicAdd(&cells[cell_index], 1);
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
    auto h_particle_counts = HostGrid{ grid_dimensions };
    auto d_particle_counts = DeviceGrid{ grid_dimensions };

    const auto block_count = (particle_count + block_size - 1) / block_size;
    count_particles<<<block_count, block_size>>>(
        d_particles.pos_x, d_particles.pos_y, d_particles.pos_z, particle_count,
        d_particle_counts.cells
    );

    // Save particle positions to disk.
    h_particles.copy(d_particles);
    const auto output_directory = std::filesystem::path(output_directory_name);
    std::filesystem::create_directory(output_directory);
    h_particles.save_positions(output_directory / "positions.csv");
    // Save particle counts to disk.
    h_particle_counts.copy(d_particle_counts);
    h_particle_counts.save(output_directory / "particle_counts.csv");
}
