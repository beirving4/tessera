cmake_minimum_required(VERSION 3.15)
project(AsymptoticTetra VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_WITH_HDF5 "Build with HDF5 support for GADGET-4" ON)

# Find threads
find_package(Threads REQUIRED)

# Find HDF5 (for GADGET-4 I/O)
if(BUILD_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    
    # Fetch HighFive (modern C++ HDF5 wrapper)
    include(FetchContent)
    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
        GIT_TAG v2.9.0
    )
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(HighFive)
endif()

# macOS specific: Ensure C++ standard library headers are found
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${MACOS_SDK_PATH}/usr/include/c++/v1")
        include_directories(SYSTEM "${MACOS_SDK_PATH}/usr/include/c++/v1")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE MATH_SOURCES "src/math/*.cpp")
file(GLOB_RECURSE GEOM_SOURCES "src/geom/*.cpp")
file(GLOB_RECURSE DENSITY_SOURCES "src/density/*.cpp")
file(GLOB_RECURSE IO_SOURCES "src/io/*.cpp")
file(GLOB_RECURSE RENDER_SOURCES "src/render/*.cpp")
file(GLOB_RECURSE HALO_SOURCES "src/halo/*.cpp")
file(GLOB_RECURSE STATS_SOURCES "src/stats/*.cpp")

set(ALL_SOURCES
    ${MATH_SOURCES}
    ${GEOM_SOURCES}
    ${DENSITY_SOURCES}
    ${IO_SOURCES}
    ${RENDER_SOURCES}
    ${HALO_SOURCES}
    ${STATS_SOURCES}
)

# Core library
add_library(asymptotic_tetra_core ${ALL_SOURCES})
target_include_directories(asymptotic_tetra_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(asymptotic_tetra_core PUBLIC Threads::Threads)

# Link HDF5 if enabled
if(BUILD_WITH_HDF5)
    target_link_libraries(asymptotic_tetra_core PUBLIC HighFive)
    target_compile_definitions(asymptotic_tetra_core PUBLIC ASYMPTOTIC_TETRA_HAS_HDF5)
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(asymptotic_tetra_core PRIVATE -Wall -Wextra -O3 -ffast-math)
elseif(MSVC)
    target_compile_options(asymptotic_tetra_core PRIVATE /W4 /O2 /fp:fast)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
    
    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fetch pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    # Python module
    pybind11_add_module(_asymptotic_tetra
        python/bindings/module.cpp
        python/bindings/geom_bindings.cpp
        python/bindings/math_bindings.cpp
        python/bindings/density_bindings.cpp
        python/bindings/tetra_density_bindings.cpp
        python/bindings/io_bindings.cpp
        python/bindings/render_bindings.cpp
        python/bindings/cosmo_bindings.cpp
        python/bindings/halo_bindings.cpp
        python/bindings/stats_bindings.cpp
    )
    target_link_libraries(_asymptotic_tetra PRIVATE asymptotic_tetra_core)
    target_include_directories(_asymptotic_tetra PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
    
    # Copy Python package to build directory
    add_custom_command(TARGET _asymptotic_tetra POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/asymptotic_tetra
            ${CMAKE_CURRENT_BINARY_DIR}/asymptotic_tetra
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:_asymptotic_tetra>
            ${CMAKE_CURRENT_BINARY_DIR}/asymptotic_tetra/
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS asymptotic_tetra_core
    EXPORT AsymptoticTetraTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
