#include "timer.cuh"

#include <iostream>

using namespace std::chrono;

thesis::Timer::Timer(std::string name)
    : name{ name }, start{ high_resolution_clock::now() } {
}

thesis::Timer::~Timer() {
    using namespace std::chrono_literals;

    const auto end = high_resolution_clock::now();
    const auto duration = end - start;
    std::cout << name << " took ";
    if (duration > 0ms) {
        const auto duration_ms = duration_cast<milliseconds>(duration);
        std::cout << duration_ms.count() << " ms.\n";
    } else {
        const auto duration_us = duration_cast<microseconds>(duration);
        std::cout << duration_us.count() << " µs.\n";
    }
}
