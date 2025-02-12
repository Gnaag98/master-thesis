#ifndef THESIS_TIMER_CUH
#define THESIS_TIMER_CUH

#include <chrono>
#include <string>

namespace thesis {
    class Timer {
    public:
        Timer(std::string name);
        ~Timer();
    private:
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
    };
};

#endif
