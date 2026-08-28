cmake_minimum_required(VERSION 3.16)

project(cca_python LANGUAGES CXX)

# Build both C++ libraries from this checkout so the extension never depends
# on a stale system installation of affordance_util.
set(CCA_BUILD_PYTHON_BINDINGS ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../affordance_util affordance_util)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cc_affordance_planner cc_affordance_planner)
