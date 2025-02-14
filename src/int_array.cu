#include "int_array.cuh"

#include "cnpy.h"

namespace {
    using namespace thesis;
    constexpr auto type_size = sizeof(decltype(HostIntArray::i)::value_type);
};

thesis::HostIntArray::HostIntArray(const size_t count) : i(count) {}

void thesis::HostIntArray::copy(const DeviceIntArray &indices) {
    const auto size = i.size() * type_size;
    cudaMemcpy(i.data(), indices.i, size, cudaMemcpyDeviceToHost);
}

void thesis::HostIntArray::save(std::filesystem::path filepath) {
    const auto shape = std::vector{ i.size() };
    cnpy::npy_save(filepath, i.data(), shape);
}

thesis::DeviceIntArray::DeviceIntArray(const HostIntArray &indices) {
    const auto size = indices.i.size() * type_size;
    cudaMalloc(&i, size);
}

thesis::DeviceIntArray::~DeviceIntArray() {
    cudaFree(i);
}

void thesis::DeviceIntArray::copy(const HostIntArray &indices) {
    const auto size = indices.i.size() * type_size;
    cudaMemcpy(i, indices.i.data(), size, cudaMemcpyHostToDevice);
}
