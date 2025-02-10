#ifndef THESIS_GRID_HPP
#define THESIS_GRID_HPP

#include <filesystem>
#include <vector>

namespace thesis {
    struct DeviceGrid;

    struct HostGrid {
        std::vector<float> cells;
        int3 dimensions;

        HostGrid(int3 dimensions);

        void copy(const DeviceGrid &grid);

        void save(std::filesystem::path filepath);
    };

    struct DeviceGrid {
        float *cells;
        int3 dimensions;

        DeviceGrid(int3 dimensions);
        DeviceGrid(const DeviceGrid &) = delete;
        ~DeviceGrid();

        void copy(const HostGrid &grid);
    };
};

#endif
