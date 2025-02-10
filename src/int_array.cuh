#ifndef THESIS_INDICES_CUH
#define THESIS_INDICES_CUH

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

        explicit DeviceIntArray(size_t count);
        DeviceIntArray(const HostIntArray &indices);
        DeviceIntArray(const DeviceIntArray &) = delete;
        ~DeviceIntArray();

        void copy(const HostIntArray &indices);
    };
};

#endif
