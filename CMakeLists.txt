cmake_minimum_required(VERSION 3.15...3.30)
project(points_lsd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POINTS_LSD_BUILD_TESTS "Build the C++ test executable (requires OpenCV)" OFF)
# OpenMP only parallelises batched_lsd and adds a runtime DLL/dylib dependency to wheels; keep it opt-in.
option(POINTS_LSD_USE_OPENMP "Link OpenMP for the batched entry point" OFF)

add_subdirectory(pybind11)

if(POINTS_LSD_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Core LSD library (pure C++, no external dependencies).
add_library(tlsd STATIC src/lsd.cpp)
target_include_directories(tlsd PUBLIC src)
if(NOT MSVC)
    target_compile_options(tlsd PRIVATE -w)
endif()

# Python bindings.
pybind11_add_module(points_lsd src/PYAPI.cpp)
target_link_libraries(points_lsd PRIVATE tlsd)
if(POINTS_LSD_USE_OPENMP)
    target_link_libraries(points_lsd PRIVATE OpenMP::OpenMP_CXX)
endif()
if(POINTS_LSD_VERSION_INFO)
    target_compile_definitions(points_lsd PRIVATE VERSION_INFO=${POINTS_LSD_VERSION_INFO})
endif()

# Optional C++ test, needs OpenCV.
if(POINTS_LSD_BUILD_TESTS)
    find_package(OpenCV REQUIRED)
    add_executable(test_tlsd tests/test_tlsd.cpp)
    target_include_directories(test_tlsd PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_tlsd PRIVATE tlsd ${OpenCV_LIBS})
endif()
