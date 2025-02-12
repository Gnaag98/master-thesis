#ifndef THESIS_CHARGE_DENSITY_SHARED_2D_CUH
#define THESIS_CHARGE_DENSITY_SHARED_2D_CUH

#include "int_array.cuh"

namespace thesis::shared_2d {
    __global__
    void initialize_particle_indices(
        size_t particle_count, int *indices
    );

    __global__
    void initialize_particle_cell_indices(
        const float *pos_x, const float *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, int *cell_indices
    );

    void sort_particles_by_cell(
        void *sort_storage, size_t &sort_storage_size,
        int *particle_cell_indices_before, int *particle_cell_indices_after,
        int *particle_indices_before, int *particle_indices_after,
        size_t particle_count
    );

    __global__
    void initialize_particle_occupancy(
        size_t particle_count, const int *cell_indices,
        int *particle_indices_rel_cell, int *particle_count_per_cell
    );

    __global__
    void charge_density(
        const float *pos_x, const float *pos_y, size_t particle_count,
        float particle_charge, int3 grid_dimensions,
        int cell_size, const int *particle_indices,
        const int *particle_cell_indices, const int *particle_indices_rel_cell,
        const int *particle_count_per_cell, float *densities
    );
};

#endif
