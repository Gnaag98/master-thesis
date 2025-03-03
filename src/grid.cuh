#ifndef THESIS_GRID_CUH
#define THESIS_GRID_CUH

#include <filesystem>
#include <vector>

#include "common.cuh"

namespace thesis {
    struct DeviceGrid;

    struct HostGrid {
        std::vector<FP> cells;
        int3 dimensions;

        HostGrid(int3 dimensions);

        void copy(const DeviceGrid &grid);

        void save(std::filesystem::path filepath);
    };

    struct DeviceGrid {
        FP *cells;
        int3 dimensions;

        DeviceGrid(int3 dimensions);
        DeviceGrid(const DeviceGrid &) = delete;
        ~DeviceGrid();

        void copy(const HostGrid &grid);
    };
};

#endif
