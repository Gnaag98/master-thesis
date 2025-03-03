#ifndef AMITIS_COMMON_CUH
#define AMITIS_COMMON_CUH

#include <type_traits>

/// Select float or double. Affects FP2, FP3 and FP4.
using FP = float;
/// Either float2 or double2 depending on type of FP.
using FP2 = std::conditional_t<
    std::is_same_v<FP, float>, float2, double2
>;
/// Either float3 or double3 depending on type of FP.
using FP3 = std::conditional_t<
    std::is_same_v<FP, float>, float3, double3
>;
/// Either float4 or double4 depending on type of FP.
using FP4 = std::conditional_t<
    std::is_same_v<FP, float>, float4, double4
>;

constexpr auto cell_coordinates(const FP3 position, const int cell_size) {
    // Optimization validated in Nsight Compute.
    const auto cell_size_inverse = 1.0f / cell_size;
    // XXX: Hardcoded half-cell shift due to one layer of ghost cells.
    return FP3{
        position.x * cell_size_inverse + 0.5f,
        position.y * cell_size_inverse + 0.5f,
        position.z * cell_size_inverse + 0.5f
    };
}

constexpr auto cell_index(const int3 cell_center,
        const int3 grid_dimensions) {
    const auto i = cell_center.x;
    const auto j = cell_center.y;
    const auto k = cell_center.z;
    return i + (j * grid_dimensions.x)
             + (k * grid_dimensions.x * grid_dimensions.y);
}

#endif
