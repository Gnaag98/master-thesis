#ifndef THESIS_CHARGE_DENSITY_SHARED_2D_CUH
#define THESIS_CHARGE_DENSITY_SHARED_2D_CUH

#include "int_array.cuh"

namespace thesis::shared_2d {
    __global__
    void initialize_particle_indices(
        size_t particle_count, int *indices
    );

    __global__
    void associate_particles_with_cells(
        const float *pos_x, const float *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, int *cell_indices
    );

    void sort_particles_by_cell(
        void *sort_storage, size_t &sort_storage_size,
        const int *associated_cells_in, int *associated_cells_out,
        const int *particle_indices_in, int *particle_indices_out,
        size_t particle_count
    );
};

#endif
