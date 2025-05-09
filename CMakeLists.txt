cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(matrix_benchmark LANGUAGES CXX CUDA)

## Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# fetch google benchmark
option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
#option(OPENMP "Enables OpenMP" ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
include(FetchContent)


include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.0
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()


FetchContent_MakeAvailable(googlebenchmark)


# Set CUDA architectures
# Modify these based on your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Set compiler flags
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
endif()

link_directories(
        /usr/local/cuda/lib64/
)

# Create executable
add_executable(spmm_benchmark spmm_benchmark.cpp)


target_include_directories(spmm_benchmark
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA libraries
target_link_libraries(spmm_benchmark
        PUBLIC
        cudart
        cublas
        cusparse
        benchmark::benchmark
)

# Set output directories
set_target_properties(spmm_benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


# blocked spmm
add_executable(blocked_mm_bench spmm-blocked-bench.cpp)

target_include_directories(blocked_mm_bench
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA libraries
target_link_libraries(blocked_mm_bench
        PUBLIC
        cudart
        cublas
        cusparse
        benchmark::benchmark
)




############# SpTRSV ################
# Create executable
add_executable(sptrsv_benchmark sptrsv_benchmark.cpp)


target_include_directories(sptrsv_benchmark
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA libraries
target_link_libraries(sptrsv_benchmark
        PUBLIC
        cudart
        cublas
        cusparse
        benchmark::benchmark
)

# Set output directories
set_target_properties(sptrsv_benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Optional: Enable some useful compiler warnings
if(MSVC)
    target_compile_options(spmm_benchmark PRIVATE /W4)
    target_compile_options(blocked_mm_bench PRIVATE /W4)
    target_compile_options(sptrsv_benchmark PRIVATE /W4)
else()
    target_compile_options(spmm_benchmark PRIVATE -Wall -Wextra)
    target_compile_options(blocked_mm_bench PRIVATE -Wall -Wextra)
    target_compile_options(sptrsv_benchmark PRIVATE -Wall -Wextra)
endif()

# Print CUDA info (optional, but useful for debugging)
message(STATUS "CUDA version: ${CMAKE_CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")