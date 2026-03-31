cmake_minimum_required(VERSION 3.22)

project(closed_chain_affordance LANGUAGES CXX)

option(CCA_BUILD_PYTHON "Build Python bindings" ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(affordance_util)
add_subdirectory(cc_affordance_planner)

if(CCA_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
