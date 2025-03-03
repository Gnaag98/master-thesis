#ifndef THESIS_PARTICLES_CUH
#define THESIS_PARTICLES_CUH

#include <filesystem>
#include <vector>

#include "common.cuh"

namespace thesis {
    struct DeviceParticles;
    
    struct HostParticles {
        std::vector<FP> pos_x;
        std::vector<FP> pos_y;
        std::vector<FP> pos_z;

        HostParticles(int count);
        HostParticles(
            std::filesystem::path positions_filepath
        );

        void copy(const DeviceParticles &particles);

        void save(std::filesystem::path filepath);
    };

    struct DeviceParticles {
        FP *pos_x;
        FP *pos_y;
        FP *pos_z;

        DeviceParticles(const HostParticles &particles);
        DeviceParticles(const DeviceParticles &) = delete;
        ~DeviceParticles();

        void copy(const HostParticles &particles);
    };
};

#endif
