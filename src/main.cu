#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "particles.cuh"

const auto random_seed = 1u;

void distribute(amitis::HostParticles &particles) {
    auto random_engine = std::default_random_engine(random_seed);

    // XXX: Magic numbers for the size of the space.
    auto distribution_x = std::uniform_real_distribution<float>(0, 128);
    auto distribution_y = std::uniform_real_distribution<float>(0, 256);
    auto distribution_z = std::uniform_real_distribution<float>(0, 512);

    for (auto i = 0; i < particles.pos_x.size(); ++i) {
        particles.pos_x[i] = distribution_x(random_engine);
        particles.pos_y[i] = distribution_y(random_engine);
        particles.pos_z[i] = distribution_z(random_engine);
    }
}

int main(int argc, char *argv[]) {
    using namespace amitis;

    if (argc < 3) {
        std::cerr << "Usage: master_thesis particle_count output_directory\n";
        return 1;
    }

    const auto particle_count = std::stoi(argv[1]);
    const auto output_directory_name = argv[2];

    const float particle_charge = 1;
    
    // Initialize particles.
    auto h_particles = HostParticles{ particle_count, particle_charge };
    auto d_particles = DeviceParticles{ h_particles };
    distribute(h_particles);
    d_particles.copy(h_particles);

    // Save particle positions to disk.
    h_particles.copy(d_particles);
    const auto output_directory = std::filesystem::path(output_directory_name);
    std::filesystem::create_directory(output_directory);
    h_particles.save_positions(output_directory / "positions.csv");
}
