cmake_minimum_required(VERSION 3.12)

project(affordance_util)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include Eigen library
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

# Find necessary cmake packages
find_package(yaml-cpp REQUIRED)
find_package(urdfdom REQUIRED)

# Define the affordance_util library
add_library(${PROJECT_NAME} STATIC src/${PROJECT_NAME}/${PROJECT_NAME}.cpp)
add_library(affordance_util::affordance_util ALIAS ${PROJECT_NAME})

# Link libraries against the affordance_util library and make them available to downstream packages
target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp urdfdom_model)

# Add include directories with different paths for build and install contexts
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Generate and install CMake package configuration files. The following steps enable finding this library with the find_package call in downstream packages
include(CMakePackageConfigHelpers)

# Define the version of the package
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION 1.0.0)

if(CCA_INSTALL_CPP_ARTIFACTS)
    install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )

    install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION include
    )

    install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE affordance_util::
        DESTINATION lib/cmake/${PROJECT_NAME}
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
    )
endif()