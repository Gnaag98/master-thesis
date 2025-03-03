#include "grid.cuh"

#include "cnpy.h"

thesis::HostGrid::HostGrid(const int3 dimensions)
    : cells(dimensions.x * dimensions.y * dimensions.z),
      dimensions{ dimensions } {}

void thesis::HostGrid::copy(const DeviceGrid &grid) {
    const auto size = cells.size() * sizeof(FP);
    cudaMemcpy(cells.data(), grid.cells, size, cudaMemcpyDeviceToHost);
}

void thesis::HostGrid::save(std::filesystem::path filepath) {
    const auto shape = std::vector{ cells.size() };
    cnpy::npy_save(filepath, cells.data(), shape);
}

thesis::DeviceGrid::DeviceGrid(const int3 dimensions)
    : dimensions{ dimensions } {
    const auto cell_count = dimensions.x * dimensions.y * dimensions.z;
    cudaMalloc(&cells, cell_count * sizeof(FP));
}

thesis::DeviceGrid::~DeviceGrid() {
    cudaFree(cells);
}

void thesis::DeviceGrid::copy(const HostGrid &grid) {
    const auto size = grid.cells.size() * sizeof(FP);
    cudaMemcpy(cells, grid.cells.data(), size, cudaMemcpyHostToDevice);
}
