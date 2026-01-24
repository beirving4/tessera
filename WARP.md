# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

tessera is a high-performance C++ library with Python bindings for phase-space tessellation of cosmological N-body simulations. It provides:

- **ORIGAMI morphology classification** - Classify particles as void, wall, filament, or halo based on shell-crossing
- **Tetrahedron-based density estimation** - Compute density fields using Lagrangian tessellation
- **Direct particle density** - Memory-efficient density at particle positions
- **GADGET-4 I/O** - Read HDF5 snapshots and merger trees

It is a modern C++17 rewrite of [gotetra](https://github.com/phil-mansfield/gotetra) with Python as the primary interface via pybind11.

## Build Commands

```bash
# Configure and build (from repository root)
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS

# Build with HDF5 support (required for GADGET-4 I/O)
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DTESSERA_USE_HDF5=ON

# Rebuild after changes (from build directory)
make -j4
```

## Using the Python Package

After building, add the build directory to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/tessera/build:$PYTHONPATH
```

Import and use:
```python
import _tessera as ts

# ORIGAMI pipeline (most common workflow)
config = ts.origami.PipelineConfig(grid_size=256, box_size=256.0)
config.density_output_cells = 256
config.pdf_n_bins = 100
result = ts.origami.run_pipeline(positions, particle_ids, config)

# Direct particle density (memory-efficient alternative)
config = ts.density.ParticleDensityConfig()
config.lagrangian_grid_size = 256
config.box_size = 256.0
result = ts.density.compute_particle_density(sorted_positions, config)
```

## Architecture

### Layer Structure
1. **C++ Core** (`include/`, `src/`) - High-performance computational core
2. **pybind11 Bindings** (`python/bindings/`) - Bridge to Python
3. **Python Package** (`python/tessera/`) - Pythonic interface and backward compatibility

### C++ Module Organization

The library is organized into namespaces under `tessera::`:

- **`origami`** - ORIGAMI morphology classification
  - `OrigamiConfig/OrigamiResult`: Configuration and results for morphology computation
  - `OrigamiPipelineConfig/OrigamiPipelineResult`: Unified pipeline configuration
  - `compute_morphology()`: Classify particles by shell-crossing count
  - `run_pipeline()`: Complete workflow (sort → ORIGAMI → density → PDF)
  - `detect_id_ordering()`: Auto-detect GADGET ID convention (x-major vs z-major)
  - `sort_to_lagrangian()`: Sort particles to Lagrangian order

- **`density`** - Density field computation
  - `TetraDensityConfig/TetraDensityResult3D`: 3D density grid computation
  - `ParticleDensityConfig/ParticleDensityResult`: Direct particle density
  - `compute_tetra_density_3d()`: Monte Carlo density on 3D grid
  - `compute_particle_density()`: Volume-based density at particles (memory-efficient)
  - `compute_tetra_density_2d_direct()`: Memory-efficient 2D slices
  - `sort_by_lagrangian_id()`: Sort positions by particle ID
  - `Quantity` enum: Density, DensityGradient, Velocity, VelocityDivergence, VelocityCurl
  - `Buffer` hierarchy: ScalarBuffer, VectorBuffer, DensityBuffer, etc.

- **`stats`** - Statistical analysis
  - `histogram()`, `histogram_log()`: Compute histograms with OpenMP
  - `histogram_to_pdf()`: Convert counts to probability density
  - `compute_jackknife_conditional_histogram()`: Per-class PDFs with error estimation

- **`io`** - File I/O for simulation formats
  - `read_gadget4_snapshot()`: Read GADGET-4 HDF5 snapshots
  - `Gadget4Snapshot`: Positions, velocities, IDs, header
  - `MergerTree`, `HaloTracker`: Merger tree analysis (if HDF5 enabled)
  - `SheetHeader`: Legacy phase sheet format (.gtet files)

- **`geom`** - Geometry primitives with periodic boundary support
  - `Vec3f/Vec3d`: 3D vectors with periodic boundary operations
  - `Tetra`: Tetrahedron with Monte Carlo sampling (Rocchini-Cignoni algorithm)
  - `Grid/GridLocation`: Grid indexing for density fields

- **`math`** - Mathematical utilities
  - `Generator`: Random number generation (Tausworthe, Xorshift, mt19937)
  - `SobolGenerator`: Quasi-random sequences
  - `parallel_argsort()`: OpenMP-accelerated sorting

- **`cosmo`** - Cosmological calculations
  - `rho_critical()`, `rho_average()`: Density calculations

- **`halo`** - Halo finding and analysis
  - `HaloGrid`: Spatial hash grid for halo positions
  - `SubhaloFinder`: Identifies subhalo relationships

- **`render`** - Multi-threaded rendering pipeline
  - `Manager`: Coordinates file I/O and parallel processing

### Key Design Patterns

- **Unified Pipeline**: `run_pipeline()` handles sorting, ORIGAMI, density, and PDFs in one call
- **Memory Efficiency**: Direct particle density avoids O(grid³) allocation
- **Streaming Indices**: Tetrahedra indices generated on-the-fly, not pre-allocated
- **Thread-Local Buffers**: Density grids use per-thread accumulation to avoid atomics
- **Caching**: `Tetra` caches volume and barycenter calculations
- **In-place Operations**: `*_self()` methods for zero-allocation
- **Periodic Boundaries**: `Vec3f::sub(other, width)` handles simulation box wrapping

### Python Bindings Structure

Each C++ module has a corresponding `*_bindings.cpp` file in `python/bindings/`. The main module (`module.cpp`) creates submodules:

```python
import _tessera as ts
ts.origami    # ORIGAMI morphology and pipeline
ts.density    # Density computation
ts.stats      # Histograms and PDFs
ts.io         # File I/O (GADGET-4, merger trees)
ts.geom       # Geometry primitives
ts.math       # Random numbers, sorting
ts.cosmo      # Cosmology calculations
ts.halo       # Halo analysis
ts.render     # Rendering pipeline
```

### Configuration Options

**PipelineConfig** (ORIGAMI + density + PDF):
```python
config = ts.origami.PipelineConfig(grid_size, box_size)
config.density_output_cells = 256      # Grid resolution
config.density_n_samples = 100         # MC samples per tetrahedron
config.use_direct_particle_density = False  # True for memory-efficient mode
config.pdf_n_bins = 100                # Histogram bins
config.pdf_jackknife = True            # Enable error estimation
config.n_threads = 0                   # 0 = auto-detect
```

**ParticleDensityConfig** (direct density only):
```python
config = ts.density.ParticleDensityConfig()
config.lagrangian_grid_size = 256
config.box_size = 256.0
config.n_threads = 1
```

## Common Workflows

### 1. Full ORIGAMI Analysis with PDFs
```python
import _tessera as ts
import numpy as np

# Load GADGET-4 snapshot
snap = ts.io.read_gadget4_snapshot("snapshot_000.hdf5")
positions = np.array(snap.positions)
particle_ids = np.array(snap.particle_ids)
box_size = snap.header.box_size
grid_size = int(round(len(positions) ** (1/3)))

# Run unified pipeline
config = ts.origami.PipelineConfig(grid_size, box_size)
config.density_output_cells = grid_size
config.sample_density_at_particles = True
config.pdf_n_bins = 100
config.pdf_jackknife = True

result = ts.origami.run_pipeline(positions, particle_ids, config)

print(f"Void: {result.f_void:.1%}, Halo: {result.f_halo:.1%}")
```

### 2. Memory-Efficient Direct Density
```python
# For large simulations (N >= 1024³) with limited RAM
config = ts.origami.PipelineConfig(grid_size, box_size)
config.use_direct_particle_density = True  # Skip 3D grid allocation
config.pdf_n_bins = 100

result = ts.origami.run_pipeline(positions, particle_ids, config)
```

### 3. 2D Density Slice
```python
config = ts.density.TetraDensityConfig()
config.lagrangian_grid_size = grid_size
config.output_cells = 512
config.box_size = box_size

result = ts.density.compute_tetra_density_2d_direct(
    sorted_positions, config,
    projection_axis=2,  # z-axis
    slice_min=127.0, slice_max=129.0
)
```

## File Format Notes

- **GADGET-4 HDF5**: Primary format, read via `ts.io.read_gadget4_snapshot()`
- **Merger Trees**: `ts.io.MergerTree` for halo evolution analysis
- **Legacy .gtet**: `python/tessera/gotetra_compat.py` for backward compatibility
- **SheetHeader**: Uses `#pragma pack(push, 1)` for exact binary layout

## Performance Notes

- **N=256³**: ~3 minutes for full pipeline (single-threaded)
- **N=1024³**: ~18-22 hours for full pipeline, ~40 GB RAM with grid-based MC
- **Direct density**: ~50x faster, ~2.5x less memory than grid-based MC
- **Scaling**: Near-linear with particle count, benefits from multi-threading

## Documentation

- `docs/ORIGAMI_PIPELINE.md` - Pipeline architecture and API
- `docs/direct_particle_density.md` - Direct density method details
- `docs/PARALLEL_SCALING_ANALYSIS.md` - Multi-threading performance
- `docs/PIPELINE_BENCHMARK_RESULTS.md` - Benchmark data
