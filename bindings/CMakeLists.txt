find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    RESULT_VARIABLE PYBIND11_INCLUDE_DIR_STATUS
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYBIND11_INCLUDE_DIR_STATUS EQUAL 0)
    message(FATAL_ERROR
        "pybind11 was not found in ${Python3_EXECUTABLE}. Install it in the selected Python "
        "environment or configure Python3_EXECUTABLE accordingly.")
endif()

if(DEFINED SKBUILD)
    set(_cca_python_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
else()
    set(_cca_python_output_dir "${CMAKE_SOURCE_DIR}/python")
endif()

add_library(closed_chain_affordance MODULE cca_bindings.cpp)

target_link_libraries(closed_chain_affordance
    PRIVATE
        Python3::Module
        affordance_util::affordance_util
        cc_affordance_planner::cc_affordance_planner
)

target_include_directories(closed_chain_affordance
    PRIVATE
        "${PYBIND11_INCLUDE_DIR}"
        /usr/include/eigen3
)
target_compile_features(closed_chain_affordance PRIVATE cxx_std_20)

set_target_properties(closed_chain_affordance PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_cca_python_output_dir}"
    PREFIX ""
)

install(TARGETS closed_chain_affordance
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

install(FILES "${CMAKE_SOURCE_DIR}/python/closed_chain_affordance/__init__.pyi"
    DESTINATION .
    RENAME closed_chain_affordance.pyi
)
