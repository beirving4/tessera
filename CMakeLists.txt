cmake_minimum_required(VERSION 3.15)
project(tessera VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_WITH_HDF5 "Build with HDF5 support for GADGET-4" ON)
option(BUILD_WITH_VORO "Build with Voronoi density support (voro++)" ON)

# Find threads
find_package(Threads REQUIRED)

# Find OpenMP (for ORIGAMI and density computation)
if(APPLE)
    # On macOS, we need to help CMake find libomp from Homebrew
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xclang -fopenmp")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        include_directories("${LIBOMP_PREFIX}/include")
        link_directories("${LIBOMP_PREFIX}/lib")
    endif()
endif()
find_package(OpenMP)

# macOS specific: Ensure C++ standard library headers are found
# Must be before FetchContent so fetched libraries inherit these paths
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

# Find HDF5 (for GADGET-4 I/O)
if(BUILD_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    
    # Fetch HighFive (modern C++ HDF5 wrapper)
    include(FetchContent)

    # HighFive has cmake_minimum_required(VERSION 3.1) which CMake 4.x rejects
    # Set policy minimum to allow it to configure
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()

    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
        GIT_TAG v2.10.1
    )
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(HighFive)
endif()

# Fetch voro++ for Voronoi density computation
if(BUILD_WITH_VORO)
    include(FetchContent)
    FetchContent_Declare(
        voro
        GIT_REPOSITORY https://github.com/chr1shr/voro.git
        GIT_TAG master
    )
    set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
    set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(voro)
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
file(GLOB_RECURSE ORIGAMI_SOURCES "src/origami/*.cpp")

# Remove HDF5-dependent files when HDF5 is disabled
if(NOT BUILD_WITH_HDF5)
    list(FILTER IO_SOURCES EXCLUDE REGEX "gadget4\\.cpp$")
    list(FILTER IO_SOURCES EXCLUDE REGEX "merger_tree\\.cpp$")
    list(FILTER IO_SOURCES EXCLUDE REGEX "position_tracker\\.cpp$")
endif()

# Remove voro++-dependent files when voro++ is disabled
if(NOT BUILD_WITH_VORO)
    list(FILTER DENSITY_SOURCES EXCLUDE REGEX "voronoi_density\\.cpp$")
endif()

set(ALL_SOURCES
    ${MATH_SOURCES}
    ${GEOM_SOURCES}
    ${DENSITY_SOURCES}
    ${IO_SOURCES}
    ${RENDER_SOURCES}
    ${HALO_SOURCES}
    ${STATS_SOURCES}
    ${ORIGAMI_SOURCES}
)

# Core library (static for embedding in Python extension)
add_library(tessera_core STATIC ${ALL_SOURCES})
target_include_directories(tessera_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tessera_core PUBLIC Threads::Threads)

# Link OpenMP if available
if(APPLE AND LIBOMP_PREFIX)
    # Manual OpenMP linking for macOS - more reliable than CMake's OpenMP detection
    target_compile_options(tessera_core PUBLIC -Xclang -fopenmp)
    target_include_directories(tessera_core PUBLIC "${LIBOMP_PREFIX}/include")
    target_link_libraries(tessera_core PUBLIC "${LIBOMP_PREFIX}/lib/libomp.dylib")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(tessera_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link HDF5 if enabled
if(BUILD_WITH_HDF5)
    target_link_libraries(tessera_core PUBLIC HighFive)
    target_compile_definitions(tessera_core PUBLIC TESSERA_HAS_HDF5)
endif()

# Link voro++ if enabled
if(BUILD_WITH_VORO)
    target_link_libraries(tessera_core PUBLIC voro++)
    target_compile_definitions(tessera_core PUBLIC TESSERA_HAS_VORO)
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(tessera_core PRIVATE -Wall -Wextra -O3 -ffast-math)
elseif(MSVC)
    target_compile_options(tessera_core PRIVATE /W4 /O2 /fp:fast)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

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

    # Find NumPy include directory
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_NOT_FOUND
    )
    if(NUMPY_NOT_FOUND)
        message(FATAL_ERROR "NumPy not found")
    endif()
    message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")

    # Python module
    pybind11_add_module(_tessera
        python/bindings/module.cpp
        python/bindings/geom_bindings.cpp
        python/bindings/math_bindings.cpp
        python/bindings/density_bindings.cpp
        python/bindings/tetra_density_bindings.cpp
        python/bindings/sph_bindings.cpp
        python/bindings/io_bindings.cpp
        python/bindings/render_bindings.cpp
        python/bindings/cosmo_bindings.cpp
        python/bindings/halo_bindings.cpp
        python/bindings/stats_bindings.cpp
        python/bindings/origami_bindings.cpp
        python/bindings/voronoi_density_bindings.cpp
    )
    target_link_libraries(_tessera PRIVATE tessera_core)
    target_include_directories(_tessera PRIVATE ${NUMPY_INCLUDE_DIR})

    # Copy Python package to build directory (for local development)
    add_custom_command(TARGET _tessera POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tessera
            ${CMAKE_CURRENT_BINARY_DIR}/tessera
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:_tessera>
            ${CMAKE_CURRENT_BINARY_DIR}/tessera/
    )

    # Install Python module (for pip install)
    install(TARGETS _tessera LIBRARY DESTINATION .)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/tessera/
            DESTINATION .
            FILES_MATCHING PATTERN "*.py")
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS tessera_core
    EXPORT TesseraTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
