#ifndef THESIS_CHARGE_DENSITY_GLOBAL_2D_CUH
#define THESIS_CHARGE_DENSITY_GLOBAL_2D_CUH

namespace thesis {
    __global__
    /// Computes 2D charge density using only global memory.
    void charge_density_global_2d(
        const float *pos_x, const float *pos_y, size_t particle_count,
        float particle_charge, int3 grid_dimensions, int cell_size,
        float *densities
    );
};

#endif
