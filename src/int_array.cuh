#ifndef THESIS_INDICES_HPP
#define THESIS_INDICES_HPP

#include <filesystem>
#include <vector>

namespace thesis {
    struct DeviceIntArray;
    
    struct HostIntArray {
        std::vector<int> i;

        HostIntArray(size_t count);

        void copy(const DeviceIntArray &indices);

        void save(std::filesystem::path filepath);
    };

    struct DeviceIntArray {
        int *i;

        // TODO: Allow device indices without corresponding host indices.
        DeviceIntArray(const HostIntArray &indices);
        DeviceIntArray(const DeviceIntArray &) = delete;
        ~DeviceIntArray();

        void copy(const HostIntArray &indices);
    };
};

#endif
