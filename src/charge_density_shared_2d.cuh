#ifndef THESIS_CHARGE_DENSITY_SHARED_2D_CUH
#define THESIS_CHARGE_DENSITY_SHARED_2D_CUH

#include "int_array.cuh"

namespace thesis::shared_2d {
    __global__
    void associate_particles_with_cells(
        const float *pos_x, const float *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, int *cell_indices
    );
};

#endif
