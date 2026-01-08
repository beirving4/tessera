# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

AsymptoticTetra is a high-performance C++ library with Python bindings for phase-space tessellation of cosmological N-body simulations. It is a modern C++17 rewrite of [gotetra](https://github.com/phil-mansfield/gotetra) with Python as the primary interface via pybind11.

## Build Commands

```bash
# Configure and build (from repository root)
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j$(sysctl -n hw.ncpu)

# Build with tests enabled
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=ON
make -j$(sysctl -n hw.ncpu)

# Rebuild after changes (from build directory)
make -j$(sysctl -n hw.ncpu)
```

## Using the Python Package

After building, add the build directory to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/AsymptoticTetra/build:$PYTHONPATH
```

Import and use:
```python
import asymptotic_tetra as at
```

## Architecture

### Layer Structure
1. **C++ Core** (`include/`, `src/`) - High-performance computational core
2. **pybind11 Bindings** (`python/bindings/`) - Bridge to Python
3. **Python Package** (`python/asymptotic_tetra/`) - Pythonic interface and backward compatibility

### C++ Module Organization

The library is organized into namespaces under `asymptotic_tetra::`:

- **`geom`** - Geometry primitives with periodic boundary support
  - `Vec3f/Vec3d`: 3D vectors with periodic boundary operations (`sub()` handles wrapping)
  - `Tetra`: Tetrahedron with Monte Carlo sampling (Rocchini-Cignoni algorithm)
  - `Grid/GridLocation`: Grid indexing for density fields
  - `CellBounds`: Axis-aligned bounding boxes

- **`math`** - Random number generation
  - `Generator`: Supports Tausworthe, Xorshift, and std::mt19937
  - Batch generation via `uniform_array()`

- **`density`** - Density field computation
  - `Quantity` enum: Density, DensityGradient, Velocity, VelocityDivergence, VelocityCurl
  - `Buffer` hierarchy: `ScalarBuffer`, `VectorBuffer`, `DensityBuffer`, `GradientBuffer`, `VelocityBuffer`, `DivergenceBuffer`, `CurlBuffer`
  - `Interpolator`: Monte Carlo interpolation engine

- **`io`** - File I/O for simulation formats
  - `SheetHeader`: Phase sheet format (binary compatible with Go version)
  - `GadgetHeader`: Gadget-2 simulation format
  - `CosmologyHeader`: Cosmological parameters (z, Ω_m, Ω_Λ, h100)

- **`render`** - Multi-threaded rendering pipeline
  - `Manager`: Coordinates file I/O, workload distribution, parallel processing
  - `Box`: Defines output regions
  - `Workspace`: Per-thread state for parallel rendering

- **`cosmo`** - Cosmological calculations
  - `rho_critical()`, `rho_average()`: Density calculations in cosmological units

- **`halo`** - Halo finding and analysis
  - `HaloGrid`: Spatial hash grid for halo positions
  - `SubhaloFinder`: Identifies subhalo relationships based on sphere overlap

### Key Design Patterns

- **Caching**: `Tetra` caches volume and barycenter calculations
- **In-place operations**: `*_self()` methods for zero-allocation (e.g., `scale_self()`, `mod_self()`)
- **SIMD-friendly layouts**: Uses `std::array<float, 3>` internally
- **Periodic boundaries**: `Vec3f::sub(other, width)` handles simulation box wrapping

### Python Bindings Structure

Each C++ module has a corresponding `*_bindings.cpp` file in `python/bindings/`. The main module (`module.cpp`) creates submodules and calls `bind_*()` functions.

Top-level convenience imports are re-exported in `python/asymptotic_tetra/__init__.py`:
```python
import asymptotic_tetra as at
at.Vec3f, at.Tetra, at.Generator, at.Quantity  # etc.
```

### Backward Compatibility

`python/asymptotic_tetra/gotetra_compat.py` provides the original gotetra.py API:
```python
from asymptotic_tetra.gotetra_compat import read_header, read_grid
```

This reads `.gtet` files with automatic endianness detection and supports both v1 and v2 file formats.

## File Format Notes

- `SheetHeader` uses `#pragma pack(push, 1)` for exact binary layout matching the Go version
- Endianness is detected from a flag in the first 8 bytes of files
- Vector fields are stored as three separate arrays (xs, ys, zs) in `.gtet` files
