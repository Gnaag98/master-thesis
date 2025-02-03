#include "grid.cuh"

#include <fstream>

amitis::HostGrid::HostGrid(const dim3 dimensions)
    : cells(dimensions.x * dimensions.y * dimensions.z),
      dimensions{ dimensions } {}

void amitis::HostGrid::copy(const DeviceGrid &grid) {
    const auto size = cells.size() * sizeof(float);
    cudaMemcpy(cells.data(), grid.cells, size, cudaMemcpyDeviceToHost);
}

void amitis::HostGrid::save(std::filesystem::path filepath) {
    auto file = std::ofstream{ filepath };
    for (const auto cell : cells) { file << cell << ','; }
    file << '\n';
}

amitis::DeviceGrid::DeviceGrid(const dim3 dimensions)
    : dimensions{ dimensions } {
    const auto cell_count = dimensions.x * dimensions.y * dimensions.z;
    cudaMalloc(&cells, cell_count * sizeof(float));
}

amitis::DeviceGrid::~DeviceGrid() {
    cudaFree(cells);
}

void amitis::DeviceGrid::copy(const HostGrid &grid) {
    const auto size = grid.cells.size() * sizeof(float);
    cudaMemcpy(cells, grid.cells.data(), size, cudaMemcpyHostToDevice);
}
