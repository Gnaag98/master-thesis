#include "particles.cuh"

#include <fstream>

amitis::HostParticles::HostParticles(const int count, const float charge)
: pos_x(count), pos_y(count), pos_z(count), charge{ charge } {}

void amitis::HostParticles::copy(const DeviceParticles &particles) {
    const auto size = pos_x.size() * sizeof(float);
    cudaMemcpy(pos_x.data(), particles.pos_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_y.data(), particles.pos_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_z.data(), particles.pos_z, size, cudaMemcpyDeviceToHost);
}

void amitis::HostParticles::save_positions(std::filesystem::path filepath) {
    auto file = std::ofstream{ filepath };
    for (const auto x : pos_x) { file << x << ','; }
    file << '\n';
    for (const auto y : pos_y) { file << y << ','; }
    file << '\n';
    for (const auto z : pos_z) { file << z << ','; }
    file << '\n';
}

amitis::DeviceParticles::DeviceParticles(const HostParticles &particles) {
    cudaMalloc(&pos_x, particles.pos_x.size() * sizeof(float));
    cudaMalloc(&pos_y, particles.pos_y.size() * sizeof(float));
    cudaMalloc(&pos_z, particles.pos_z.size() * sizeof(float));
}

amitis::DeviceParticles::~DeviceParticles() {
    cudaFree(pos_x);
    cudaFree(pos_y);
    cudaFree(pos_z);
}

void amitis::DeviceParticles::copy(const HostParticles &particles) {
    const auto size = particles.pos_x.size() * sizeof(float);
    cudaMemcpy(pos_x, particles.pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(pos_y, particles.pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(pos_z, particles.pos_z.data(), size, cudaMemcpyHostToDevice);
}
