# AsymptoticTetra

A high-performance C++ library with Python bindings for phase-space tessellation of cosmological N-body simulations. This is a modern rewrite of [gotetra](https://github.com/phil-mansfield/gotetra) with the computational workhorse in C++ and Python as the primary interface.

## Features

- **High-performance C++ core**: Optimized implementations of tetrahedron-based Monte Carlo density estimation
- **Python-first interface**: Clean, Pythonic API with NumPy integration
- **Backward compatible**: Drop-in replacement for the original gotetra Python interface
- **Multi-threaded**: Parallel rendering support for large simulations
- **Flexible output**: Support for density, velocity, gradients, divergence, and curl fields

## Installation

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- Python 3.8+ with NumPy
- pybind11 (automatically fetched if not installed)

### Building from Source

```bash
cd AsymptoticTetra
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j4
```

### Installing the Python Package

After building, the Python package is available in `build/asymptotic_tetra/`:

```bash
# Add to Python path or install
export PYTHONPATH=/path/to/build:$PYTHONPATH

# Or copy to site-packages
cp -r build/asymptotic_tetra /path/to/python/site-packages/
```

## Quick Start

### Python Usage

```python
import asymptotic_tetra as at
import numpy as np

# Create vectors and tetrahedra
v1 = at.Vec3f(0, 0, 0)
v2 = at.Vec3f(1, 0, 0)
v3 = at.Vec3f(0, 1, 0)
v4 = at.Vec3f(0, 0, 1)

tet = at.Tetra(v1, v2, v3, v4)
print(f"Volume: {tet.volume()}")
print(f"Barycenter: {tet.barycenter()}")

# Check if a point is inside
test_point = at.Vec3f(0.1, 0.1, 0.1)
print(f"Contains point: {tet.contains(test_point)}")

# Random number generation
gen = at.Generator.new_time_seed()
samples = gen.uniform_array(0, 1, 1000)
```

### Reading gotetra Output Files

```python
# Backward compatible with original gotetra.py
from asymptotic_tetra.gotetra_compat import read_header, read_grid

# Read file header
header = read_header("density_field.gtet")
print(f"Redshift: {header.cosmo.redshift}")
print(f"Box width: {header.cosmo.box_width} Mpc/h")
print(f"Grid dimensions: {header.dim}")

# Read density grid
density = read_grid("density_field.gtet")
print(f"Grid shape: {density.shape}")
```

### Working with Density Fields

```python
from asymptotic_tetra.density import Quantity, create_buffer

# Create a density buffer
buf = create_buffer(Quantity.Density, len=1000000)

# Check quantity properties
print(f"Requires velocity: {at.density.requires_velocity(Quantity.Velocity)}")
print(f"Can project: {at.density.can_project(Quantity.Density)}")
```

## Architecture

```
AsymptoticTetra/
├── include/           # C++ headers
│   ├── geom/          # Geometry (Vec, Tetra, Grid, CellBounds)
│   ├── math/          # Random number generators
│   ├── density/       # Density interpolation and buffers
│   ├── io/            # File I/O (Gadget, Sheet formats)
│   └── render/        # Rendering manager and boxes
├── src/               # C++ implementation
├── python/            # Python bindings and package
│   ├── bindings/      # pybind11 bindings
│   └── asymptotic_tetra/  # Python package
└── tests/             # Unit tests
```

## Modules

### `geom` - Geometry Primitives

- `Vec3f`: 3D vector with periodic boundary operations
- `Tetra`: Tetrahedron with volume, containment, and Monte Carlo sampling
- `Grid`, `GridLocation`: Grid indexing and physical location
- `CellBounds`: Axis-aligned bounding boxes

### `math` - Random Number Generation

- `Generator`: High-quality random number generator
- Supports Tausworthe, Xorshift, and standard library generators
- Efficient batch generation

### `density` - Density Field Computation

- `Quantity`: Density, Velocity, DensityGradient, VelocityCurl, VelocityDivergence
- `Buffer`: Memory management for field data
- Monte Carlo interpolation

### `io` - File I/O

- `SheetHeader`, `GadgetHeader`: Simulation file format headers
- `CosmologyHeader`: Cosmological parameters
- Read/write for Gadget-2 and phase sheet formats

## Performance Notes

The C++ core is optimized for performance:

- SIMD-friendly data layouts
- Cache-conscious algorithms
- Multi-threaded rendering
- Memory-efficient buffer management

Typical speedups over pure Python implementations: 10-100x depending on operation.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Based on the original [gotetra](https://github.com/phil-mansfield/gotetra) by Phil Mansfield.
The tetrahedron sampling algorithm is based on C. Rocchini & P. Cignoni (2001).

## Citation

If you use this software in your research, please cite:

```bibtex
@software{asymptotic_tetra,
  author = {Based on gotetra by Phil Mansfield},
  title = {AsymptoticTetra: Phase-space tessellation for cosmological simulations},
  url = {https://github.com/your-repo/AsymptoticTetra}
}
```
