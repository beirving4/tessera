# AsymptoticTetra Examples

This directory contains example scripts demonstrating how to use the asymptotic_tetra library.

## Prerequisites

Make sure you've built the library with Python bindings:

```bash
cd /path/to/AsymptoticTetra
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_HDF5=ON
make -j4
```

Then add the build directory to your Python path:

```bash
export PYTHONPATH=/path/to/AsymptoticTetra/build:$PYTHONPATH
```

## Examples

### basic_usage.py

Demonstrates core functionality:
- Vec3f/Vec3d vectors with periodic boundary operations
- Random number generators (Tausworthe, Xorshift, Mersenne)
- Tetrahedra (volume, barycenter, point containment)
- Density quantities and buffers
- Cell bounds for domain decomposition
- Cosmology functions (critical density, average density)

```bash
python examples/basic_usage.py
```

### gadget4_io.py

Demonstrates reading GADGET-4 HDF5 files:
- Reading snapshot headers
- Reading particle positions and velocities
- Reading FOF group catalogs
- Reading Subfind subhalo catalogs

```bash
# Edit the file paths in the script first
python examples/gadget4_io.py
```

### histogram.py

Demonstrates the histogram module:
- Configuring histogram binning (linear/log)
- Defining spatial regions (HistBox)
- Reading density grids
- Using HistManager for parallel histogram computation

```bash
python examples/histogram.py
```

### density_slice.py

Creates a 2D projected density slice from a GADGET-4 snapshot:
- Reads particles from single or distributed snapshots
- Selects particles within a thin slab
- Projects to a 2D grid to create a density field
- Saves output to HDF5 with full metadata
- Optional visualization (requires matplotlib)

```bash
# Basic usage
python examples/density_slice.py snapshot_034.hdf5 -o density.h5

# Distributed snapshot with custom slice parameters
python examples/density_slice.py snapdir_009/ -n 9 --center 125 --thickness 25 -o density.h5

# With visualization
python examples/density_slice.py snapshot.hdf5 -o density.h5 --plot density.png --cmap ocean
```

Inspired by [gotetra's density visualizations](https://pages.astro.umd.edu/~diemer/erebos/web/viz/).

## Quick Start

```python
import asymptotic_tetra as at

# Check available modules
print(dir(at))

# Basic vector operations
v = at.Vec3f(1, 2, 3)
print(v.norm())

# Check HDF5 support
print(f"HDF5 support: {at.io.HAS_HDF5}")

# Read GADGET-4 snapshot (if HDF5 enabled)
if at.io.HAS_HDF5:
    header = at.io.read_gadget4_header("snapshot.hdf5")
    print(f"Box size: {header.box_size}")
```
