#include "int_array.cuh"

#include <fstream>

namespace {
    using namespace amitis;
    constexpr auto type_size = sizeof(decltype(HostIntArray::i)::value_type);
};

amitis::HostIntArray::HostIntArray(const size_t count) : i(count) {}

void amitis::HostIntArray::copy(const DeviceIntArray &indices) {
    const auto size = i.size() * type_size;
    cudaMemcpy(i.data(), indices.i, size, cudaMemcpyDeviceToHost);
}

void amitis::HostIntArray::save(std::filesystem::path filepath) {
    auto file = std::ofstream{ filepath };
    for (const auto index : i) { file << index << ','; }
    file << '\n';
}

amitis::DeviceIntArray::DeviceIntArray(const HostIntArray &indices) {
    const auto size = indices.i.size() * type_size;
    cudaMalloc(&i, size);
}

amitis::DeviceIntArray::~DeviceIntArray() {
    cudaFree(i);
}

void amitis::DeviceIntArray::copy(const HostIntArray &indices) {
    const auto size = indices.i.size() * type_size;
    cudaMemcpy(i, indices.i.data(), size, cudaMemcpyHostToDevice);
}
