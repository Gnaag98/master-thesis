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
    const auto duration_us = duration_cast<microseconds>(duration);
    std::cout << name << " took " << duration_us.count() << " µs.\n";
}
