cmake_minimum_required(VERSION 3.12)

project(cc_affordance_planner)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include Eigen library
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(include ${EIGEN3_INCLUDE_DIR})

# Prefer the sibling source tree in a repository checkout so new planner and
# utility APIs cannot accidentally compile against a stale system install. A
# separately distributed planner package still falls back to find_package().
if(NOT TARGET affordance_util::affordance_util)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../affordance_util/CMakeLists.txt")
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/../affordance_util"
            "${CMAKE_CURRENT_BINARY_DIR}/affordance_util"
        )
    else()
        find_package(affordance_util REQUIRED)
    endif()
endif()

# Create the library with source files
add_library(${PROJECT_NAME} STATIC
    src/${PROJECT_NAME}/${PROJECT_NAME}.cpp
    src/${PROJECT_NAME}/${PROJECT_NAME}_interface.cpp
    src/${PROJECT_NAME}/${PROJECT_NAME}_util.cpp
    src/${PROJECT_NAME}/rm_cca.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link this package's library against affordance_util and pthread
target_link_libraries(${PROJECT_NAME} PUBLIC affordance_util::affordance_util pthread)

option(CCA_BUILD_PYTHON_BINDINGS "Build the pybind11 Python extension" OFF)
if(CCA_BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(cca_cpp MODULE python/cca_cpp_bindings.cpp)
    target_link_libraries(cca_cpp PRIVATE ${PROJECT_NAME})
    set_target_properties(cca_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
endif()

# Add include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set the library installation path
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install the header files
install(FILES include/${PROJECT_NAME}/${PROJECT_NAME}.hpp
              include/${PROJECT_NAME}/${PROJECT_NAME}_interface.hpp
              include/${PROJECT_NAME}/${PROJECT_NAME}_util.hpp
              include/${PROJECT_NAME}/rm_cca.hpp
        DESTINATION include/${PROJECT_NAME}
)

# Generate and install CMake package configuration files. The following steps enable finding this library with the find_package call in downstream packages
include(CMakePackageConfigHelpers)

# Define the version of the package
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION 1.0.0)

# Generate the Targets file for find_package()
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE cc_affordance_planner::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Create the main package configuration file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

# Install the generated Config.cmake file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
)

# Add the cc_affordance_planner library tester and link it against the cc_affordance_planner library
add_executable(${PROJECT_NAME}_tester src/${PROJECT_NAME}_tester.cpp)
target_link_libraries(${PROJECT_NAME}_tester ${PROJECT_NAME})

# Lightweight deterministic RM-CCA tests, including a Kinova floating-base integration case.
add_executable(rm_cca_tester src/rm_cca_tester.cpp)
target_compile_definitions(rm_cca_tester PRIVATE
    CCA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../affordance_util/src/test"
)
target_link_libraries(rm_cca_tester ${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
    add_test(NAME fixed_base_cca_regression COMMAND ${PROJECT_NAME}_tester)
    add_test(NAME rm_cca_deterministic_and_integration COMMAND rm_cca_tester)
endif()
