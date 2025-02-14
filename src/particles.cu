#include "particles.cuh"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "cnpy.h"

namespace {
    void parse_line(std::ifstream &file, std::vector<float> &positions) {
        auto line_x = std::string{};
        getline(file, line_x);
        auto stream_x = std::stringstream{ line_x };
        auto coordinate_x = std::string{};
        while (getline(stream_x, coordinate_x, ',')) {
            if (!coordinate_x.empty()) {
                positions.push_back(std::stof(coordinate_x));
            }
        }
    }
}

thesis::HostParticles::HostParticles(const int count, const float charge)
: pos_x(count), pos_y(count), pos_z(count), charge{ charge } {}

thesis::HostParticles::HostParticles(
    const std::filesystem::path positions_filepath, const float charge
) : charge{ charge } {
    auto file = std::ifstream{ positions_filepath };
    if (!file.is_open()) {
        throw std::runtime_error("Could not open particle positions file.");
    }
    parse_line(file, pos_x);
    parse_line(file, pos_y);
    parse_line(file, pos_z);

    if (pos_x.size() != pos_y.size() && pos_y.size() != pos_z.size()) {
        throw std::runtime_error("x,y and z data not of same length.");
    }
}

void thesis::HostParticles::copy(const DeviceParticles &particles) {
    const auto size = pos_x.size() * sizeof(float);
    cudaMemcpy(pos_x.data(), particles.pos_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_y.data(), particles.pos_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_z.data(), particles.pos_z, size, cudaMemcpyDeviceToHost);
}

void thesis::HostParticles::save_positions(std::filesystem::path filepath) {
    const auto shape = std::vector{ 1, pos_x.size() };
    cnpy::npz_save(filepath, "pos_x", pos_x.data(), shape);
    cnpy::npz_save(filepath, "pos_y", pos_y.data(), shape, "a");
    cnpy::npz_save(filepath, "pos_z", pos_z.data(), shape, "a");
}

thesis::DeviceParticles::DeviceParticles(const HostParticles &particles) {
    const auto size = particles.pos_x.size();
    cudaMalloc(&pos_x, size * sizeof(float));
    cudaMalloc(&pos_y, size * sizeof(float));
    cudaMalloc(&pos_z, size * sizeof(float));
}

thesis::DeviceParticles::~DeviceParticles() {
    cudaFree(pos_x);
    cudaFree(pos_y);
    cudaFree(pos_z);
}

void thesis::DeviceParticles::copy(const HostParticles &particles) {
    const auto size = particles.pos_x.size() * sizeof(float);
    cudaMemcpy(pos_x, particles.pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(pos_y, particles.pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(pos_z, particles.pos_z.data(), size, cudaMemcpyHostToDevice);
}
