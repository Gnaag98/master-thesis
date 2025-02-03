#ifndef AMITIS_PARTICLES_HPP
#define AMITIS_PARTICLES_HPP

#include <filesystem>
#include <vector>

namespace amitis {
    struct DeviceParticles;
    
    struct HostParticles {
        std::vector<float> pos_x;
        std::vector<float> pos_y;
        std::vector<float> pos_z;
        float charge;

        HostParticles(int count, float charge);

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
