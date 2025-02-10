cmake_minimum_required(VERSION 3.20)
project(master_thesis LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    src/main.cu
    src/particles.cu
    src/particles.cuh
    src/grid.cu
    src/grid.cuh
    src/int_array.cu
    src/int_array.cuh
    src/particle_generation.cu
    src/particle_generation.cuh
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_compile_options(${PROJECT_NAME}
    PRIVATE -arch=sm_75
    PRIVATE --expt-relaxed-constexpr
)
