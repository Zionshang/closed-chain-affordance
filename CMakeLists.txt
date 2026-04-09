cmake_minimum_required(VERSION 3.22)

project(closed_chain_affordance LANGUAGES CXX)

option(CCA_BUILD_PYTHON "Build Python bindings" ON)

if(DEFINED SKBUILD)
    set(CCA_INSTALL_CPP_ARTIFACTS OFF CACHE BOOL "Install the C++ libraries and CMake package files" FORCE)
else()
    option(CCA_INSTALL_CPP_ARTIFACTS "Install the C++ libraries and CMake package files" ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(affordance_util)
add_subdirectory(cc_affordance_planner)

if(CCA_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
