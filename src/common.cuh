#ifndef AMITIS_COMMON_CUH
#define AMITIS_COMMON_CUH

constexpr auto cell_coordinates(const float3 position, const int cell_size) {
    // Optimization validated in Nsight Compute.
    const auto cell_size_inverse = 1.0f / cell_size;
    // XXX: Hardcoded half-cell shift due to one layer of ghost cells.
    return float3{
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
