#ifndef THESIS_CHARGE_DENSITY_SHARED_2D_CUH
#define THESIS_CHARGE_DENSITY_SHARED_2D_CUH

#include "common.cuh"
#include "int_array.cuh"

namespace thesis::shared_2d {
    __global__
    void initialize_particle_indices(
        size_t particle_count, int *indices
    );

    __global__
    void associate_particles_with_cells(
        const FP *pos_x, const FP *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, int *cell_indices
    );

    void sort_particles_by_cell(
        void *sort_storage, size_t &sort_storage_size,
        const int *associated_cells_in, int *associated_cells_out,
        const int *particle_indices_in, int *particle_indices_out,
        size_t particle_count
    );

    __global__
    void contextualize_cell_associations(
        size_t particle_count, const int *associated_cell_indices,
        int *particle_indices_rel_cell, int *particle_count_per_cell
    );

    __global__
    /**
     * Computes 2D charge density using global and shared memory.
     * 
     * NOTE: Must use same block size as the kernel that generated the
     * contextual cell data.
     */
    void charge_density(
        const FP *pos_x, const FP *pos_y, size_t particle_count,
        int3 grid_dimensions, int cell_size, const int *particle_indices,
        const int *associated_cells, const int *indices_rel_cell,
        const int *particle_count_per_cell, FP *densities
    );
};

#endif
