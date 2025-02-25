#ifndef THESIS_CHARGE_DENSITY_GLOBAL_2D_CUH
#define THESIS_CHARGE_DENSITY_GLOBAL_2D_CUH

namespace thesis::global_2d {
    __global__
    /// Computes 2D charge density using only global memory.
    void charge_density(
        const float *pos_x, const float *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, float *densities
    );
};

#endif
