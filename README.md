# tessera

> **tes·ser·a** /ˈtesərə/ *noun*
>
> A small block of stone, tile, glass, or other material used in the construction of a mosaic. From Latin *tessera*, from Greek *tessares* "four" — referring to the four corners of the tiles used to tessellate a surface.

---

A high-performance C++ library with Python bindings for phase-space tessellation of cosmological N-body simulations. This is a modern rewrite of [gotetra](https://github.com/phil-mansfield/gotetra) with C++ as the computational core and Python as the primary interface.

## Overview

tessera computes density fields from N-body simulations using tetrahedron-based phase-space tessellation. Unlike traditional particle-mesh methods, this approach:

- **Handles stream crossing**: Properly resolves multi-stream regions where particle trajectories cross
- **Preserves phase-space structure**: Uses the Lagrangian-to-Eulerian mapping encoded in particle IDs
- **Provides sub-grid resolution**: Monte Carlo sampling within tetrahedra captures structure below the grid scale

The library also includes **ORIGAMI morphological classification** (Falck, Neyrinck & Szalay 2012) for identifying cosmic web structures (voids, walls, filaments, halos).

## Features

- **Tetrahedron-based density fields**: 3D volumes and 2D z-projections with Monte Carlo sampling
- **Subbox extraction**: Compute density in regions centered on halos or other structures
- **Physical space computation**: Option to compute density in physical (not comoving) coordinates
- **ORIGAMI classification**: Identify void/wall/filament/halo morphology per particle
- **GADGET-4 I/O**: Read HDF5 snapshots (single or distributed) and FOF/Subfind catalogs
- **Multi-threaded**: Parallel computation with OpenMP
- **Validated**: Tested against original gotetra with >0.999 correlation

## Installation

### Prerequisites

- CMake 3.15+
- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- Python 3.8+ with NumPy
- HDF5 (optional, for GADGET-4 I/O)
- pybind11 (fetched automatically)

### Quick Install (pip)

The easiest way to install tessera is via pip:

```bash
git clone https://github.com/beirving4/tessera.git
cd tessera
pip install .
```

This will build the C++ extension and install it as a Python package. Verify the installation:

```bash
python -c "import tessera; print('Modules:', tessera.available_modules())"
```

### Building from Source (Development)

For development or if you need more control over the build:

```bash
git clone https://github.com/beirving4/tessera.git
cd tessera
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_HDF5=ON
make -j4
```

Then add the build directory to your Python path:

```bash
export PYTHONPATH=/path/to/tessera/build:$PYTHONPATH
python -c "import _tessera as ts; print('Modules:', [m for m in dir(ts) if not m.startswith('_')])"
```

## Quick Start

### Computing a 3D Density Field

```python
import numpy as np
import h5py
import _tessera as ts

# Load GADGET-4 snapshot
with h5py.File('snapshot_034.hdf5', 'r') as f:
    positions = np.ascontiguousarray(f['PartType1/Coordinates'][:], dtype=np.float64)
    particle_ids = np.ascontiguousarray(f['PartType1/ParticleIDs'][:], dtype=np.int64)
    box_size = float(f['Header'].attrs['BoxSize'])

# Determine grid size from particle count (assumes N^3 particles)
n_particles = len(positions)
grid_size = int(round(n_particles ** (1/3)))

# Sort particles by Lagrangian ID (required for tetrahedron construction)
sorted_positions = ts.density.sort_by_lagrangian_id(positions, particle_ids, grid_size)

# Configure density computation
config = ts.density.TetraDensityConfig()
config.lagrangian_grid_size = grid_size  # Particles per dimension
config.box_size = box_size
config.output_cells = 256                 # Output grid resolution
config.n_samples = 100                    # Monte Carlo samples per tetrahedron
config.n_threads = 4                      # OpenMP threads (0 = auto)
config.periodic = True                    # Periodic boundary conditions
config.particle_mass = 1.0                # Mass per particle

# Compute 3D density
result = ts.density.compute_tetra_density_3d(sorted_positions, config)
density_3d = np.array(result.density).reshape(256, 256, 256)
```

### Computing a 2D Projected Density

```python
# 2D z-projection (integrate along z-axis)
result_2d = ts.density.compute_tetra_density_2d_projection(sorted_positions, config, axis=2)
density_2d = np.array(result_2d.density).reshape(256, 256)
```

### Subbox Extraction (Halo-Centric Density)

```python
# Extract a 10 Mpc/h cube centered on a halo
config.subbox_enabled = True
config.subbox_origin = (halo_x - 5.0, halo_y - 5.0, halo_z - 5.0)
config.subbox_width = (10.0, 10.0, 10.0)
config.output_cells = 128

result = ts.density.compute_tetra_density_3d(sorted_positions, config)
```

### Physical Space Density (for a > 1 simulations)

For simulations run beyond a=1, you can compute density in physical coordinates while maintaining comoving grid extents:

```python
# Scale positions to physical coordinates
scale_factor = 100.0  # a=100
positions_physical = sorted_positions * scale_factor
box_physical = box_size * scale_factor

# Update config for physical box
config.box_size = box_physical
if config.subbox_enabled:
    config.subbox_origin = tuple(o * scale_factor for o in config.subbox_origin)
    config.subbox_width = tuple(w * scale_factor for w in config.subbox_width)

# Compute density in physical space
result = ts.density.compute_tetra_density_3d(positions_physical, config)
```

### ORIGAMI Morphology Classification

```python
# Configure ORIGAMI
origami_config = ts.origami.OrigamiConfig()
origami_config.lagrangian_grid_size = grid_size
origami_config.box_size = box_size
origami_config.n_threads = 1
origami_config.n_split = 1

# Compute morphology (0=void, 1=wall, 2=filament, 3=halo)
result = ts.origami.compute_morphology(sorted_positions, origami_config)

print(f"Void:     {result.n_void:,} particles ({result.f_void:.1%})")
print(f"Wall:     {result.n_wall:,} particles ({result.f_wall:.1%})")
print(f"Filament: {result.n_filament:,} particles ({result.f_filament:.1%})")
print(f"Halo:     {result.n_halo:,} particles ({result.f_halo:.1%})")

# Get per-particle classification
morphology = np.array(result.morphology)  # uint8 array, values 0-3
```

### Sampling Density at Particle Positions

```python
# After computing 3D density, sample at particle locations
ts.origami.sample_density_at_particles(
    density_3d,        # 3D density array [z, y, x]
    sorted_positions,  # Particle positions
    box_size,
    result             # ORIGAMI result object (modified in-place)
)

particle_density = np.array(result.particle_density)
```

## Examples

The `examples/` directory contains complete scripts demonstrating various use cases:

| Script | Description |
|--------|-------------|
| `basic_usage.py` | Core API: vectors, tetrahedra, random generators |
| `gadget4_io.py` | Reading GADGET-4 snapshots and halo catalogs |
| `tetra_density.py` | Reference implementation of density algorithm |
| `density_slice.py` | 2D density projections with visualization |
| `halo_density.py` | Halo-centric density extraction with tri-panel plots |
| `origami_morphology.py` | Full ORIGAMI workflow with conditional PDFs |
| `overdensity_pdf_origami.py` | Overdensity distributions by morphological class |

Run an example:
```bash
python examples/origami_morphology.py snapshot_034.hdf5 -o origami.h5 --plot origami.png
```

## Modules

### `ts.density` - Tetrahedron Density Fields

- `TetraDensityConfig`: Configuration for density computation
- `compute_tetra_density_3d()`: Full 3D density field
- `compute_tetra_density_2d_projection()`: 2D projection along an axis
- `sort_by_lagrangian_id()`: Sort particles to Lagrangian grid order

### `ts.origami` - Morphological Classification

- `OrigamiConfig`: Configuration for ORIGAMI algorithm
- `compute_morphology()`: Classify particles as void/wall/filament/halo
- `sample_density_at_particles()`: Interpolate density grid at particle positions
- `deposit_morphology_to_grid()`: Create grid-based morphology fields

### `ts.io` - File I/O

- `read_gadget4_header()`: Read GADGET-4 HDF5 header
- `read_gadget4_positions()`: Read particle coordinates
- `read_gadget4_velocities()`: Read particle velocities
- `read_fof_catalog()`: Read Friends-of-Friends group catalog
- `read_subfind_catalog()`: Read Subfind subhalo catalog

### `ts.geom` - Geometry Primitives

- `Vec3f`, `Vec3d`: 3D vectors with periodic operations
- `Tetra`: Tetrahedron with volume, containment, Monte Carlo sampling
- `CellBounds`: Axis-aligned bounding boxes

### `ts.math` - Random Number Generation

- `Generator`: High-quality RNG (Tausworthe, Xorshift, Mersenne Twister)
- Efficient batch generation for Monte Carlo sampling

### `ts.cosmo` - Cosmology

- `critical_density()`: Critical density at redshift z
- `average_density()`: Mean matter density

## Algorithm

The density computation follows the gotetra algorithm:

1. **Lagrangian Grid**: Particles are indexed by their initial grid positions (encoded in particle IDs)
2. **Tetrahedron Decomposition**: Each Lagrangian cell is split into 6 tetrahedra
3. **Monte Carlo Sampling**: Random points within each tetrahedron deposit mass onto the Eulerian grid
4. **Stream Crossing**: Multi-stream regions are naturally handled as tetrahedra can overlap

The ORIGAMI algorithm detects shell-crossing by checking for sign reversals in particle ordering along Cartesian and diagonal directions.

## Validation

The library has been validated against the original gotetra implementation:

| Test Case | Correlation | Mean Relative Diff |
|-----------|-------------|-------------------|
| Full box (a=1) | 0.99999 | 0.35% |
| Full box (a=100) | 0.99999 | 0.40% |
| Halo subbox (a=1) | 0.99989 | 0.17% |
| Halo subbox (a=100) | 0.99999 | 0.15% |

See `tests/gotetra_validation/` for validation scripts and results.

## References

- **gotetra**: Mansfield, P. - [github.com/phil-mansfield/gotetra](https://github.com/phil-mansfield/gotetra)
- **ORIGAMI**: Falck, B., Neyrinck, M. C., & Szalay, A. S. 2012, ApJ, 754, 126
- **Phase-space tessellation**: Abel, T., Hahn, O., & Kaehler, R. 2012, MNRAS, 427, 61 - [Tracing the dark matter sheet in phase space](https://academic.oup.com/mnras/article/427/1/61/1032914)
- **Visualization**: Kaehler, R., Hahn, O., & Abel, T. 2012, IEEE TVCG, 18, 2078 - [A Novel Approach to Visualizing Dark Matter Simulations](https://ieeexplore.ieee.org/document/6327223)
- **Sheet simulation**: Hahn, O., Abel, T., & Kaehler, R. 2013, MNRAS, 434, 1171 - [A new approach to simulating collisionless dark matter fluids](https://academic.oup.com/mnras/article/434/2/1171/1064908)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{tessera,
  author = {Irving, Bryen and Mansfield, Phil},
  title = {tessera: Phase-space tessellation for cosmological simulations},
  url = {https://github.com/beirving4/tessera},
  note = {Based on gotetra by Phil Mansfield}
}
```
