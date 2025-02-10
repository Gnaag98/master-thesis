#ifndef THESIS_CHARGE_DENSITY_SHARED_2D_CUH
#define THESIS_CHARGE_DENSITY_SHARED_2D_CUH

#include "int_array.cuh"

namespace thesis {
    class ChargeDensityShared2d {
    public:
        ChargeDensityShared2d(
            size_t particle_count, size_t block_count
        );
        ~ChargeDensityShared2d();

        /// Computes 2D charge density using global and shared memory.
        void compute(
            const float *pos_x, const float *pos_y, size_t particle_count,
            float particle_charge, int3 grid_dimensions, int cell_size,
            float *densities
        );

    private:
        void sort_particles_by_cell(size_t particle_count);

        DeviceIntArray particle_indices_before;
        DeviceIntArray particle_indices_after;
        DeviceIntArray particle_cell_indices_before;
        DeviceIntArray particle_cell_indices_after;
        DeviceIntArray particle_indices_rel_cell;
        DeviceIntArray particle_count_per_cell;

        const size_t block_count;

        void *sort_storage = nullptr;
        size_t sort_storage_size;
    };
};

#endif
