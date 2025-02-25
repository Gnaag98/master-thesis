#ifndef THESIS_PARTICLES_CUH
#define THESIS_PARTICLES_CUH

#include <filesystem>
#include <vector>

namespace thesis {
    struct DeviceParticles;
    
    struct HostParticles {
        std::vector<float> pos_x;
        std::vector<float> pos_y;
        std::vector<float> pos_z;

        HostParticles(int count);
        HostParticles(
            std::filesystem::path positions_filepath
        );

        void copy(const DeviceParticles &particles);

        void save(std::filesystem::path filepath);
    };

    struct DeviceParticles {
        float *pos_x;
        float *pos_y;
        float *pos_z;

        DeviceParticles(const HostParticles &particles);
        DeviceParticles(const DeviceParticles &) = delete;
        ~DeviceParticles();

        void copy(const HostParticles &particles);
    };
};

#endif
