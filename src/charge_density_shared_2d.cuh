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

    __global__
    void associate_blocks_with_cells(
        size_t particle_count, size_t max_block_count,
        const int *associated_cell_indices, int *cell_indices,
        int *first_particle_indices, int *cell_particle_counts,
        int *block_count, int *max_particles_per_cell
    );

    __global__
    /**
     * Computes 2D charge density using global and shared memory.
     * 
     * NOTE: One block per associated cell.
     */
    void charge_density(
        const float *pos_x, const float *pos_y, size_t particle_count,
        float particle_charge, int3 grid_dimensions, int cell_size,
        const int *particle_indices, const int *cell_indices,
        const int *first_particle_indices, const int *cell_particle_counts,
        float *densities
    );
};

#endif
