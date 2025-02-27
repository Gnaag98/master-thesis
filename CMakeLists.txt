cmake_minimum_required(VERSION 3.20)
project(master_thesis LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    src/main.cu
    src/common.cuh
    src/particles.cu
    src/particles.cuh
    src/grid.cu
    src/grid.cuh
    src/int_array.cu
    src/int_array.cuh
    src/particle_generation.cu
    src/particle_generation.cuh
    src/charge_density_global_2d.cu
    src/charge_density_global_2d.cuh
    src/charge_density_shared_2d.cu
    src/charge_density_shared_2d.cuh
    src/timer.cu
    src/timer.cuh
    src/program_options.cu
    src/program_options.cuh
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_ARCHITECTURES 75
)

target_compile_options(${PROJECT_NAME}
    PRIVATE --expt-relaxed-constexpr
    PRIVATE --Werror=all-warnings
    #PRIVATE -lineinfo
)

target_link_libraries(${PROJECT_NAME}
    /usr/local/lib/libcnpy.so
    z
)

# Comment/uncomment this line to disable/enable all compile definitions.
#[[
target_compile_definitions(${PROJECT_NAME}
    PRIVATE DEBUG
)
#]]
