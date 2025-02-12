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
        float charge;

        HostParticles(int count, float charge);
        HostParticles(
            std::filesystem::path positions_filepath, float charge
        );

        void copy(const DeviceParticles &particles);

        void save_positions(std::filesystem::path filepath);
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
