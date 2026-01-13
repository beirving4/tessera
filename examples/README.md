# AsymptoticTetra Examples

This directory contains example scripts demonstrating how to use the asymptotic_tetra library.

## Prerequisites

Build the library with Python bindings and HDF5 support:

```bash
cd /path/to/AsymptoticTetra
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_HDF5=ON
make -j4
```

Add the build directory to your Python path:

```bash
export PYTHONPATH=/path/to/AsymptoticTetra/build:$PYTHONPATH
```

## Examples

### Core API

#### `basic_usage.py`

Demonstrates core functionality:
- `Vec3f`/`Vec3d` vectors with periodic boundary operations
- Random number generators (Tausworthe, Xorshift, Mersenne Twister)
- Tetrahedra (volume, barycenter, point containment)
- Density quantities and buffers
- Cell bounds for domain decomposition
- Cosmology functions

```bash
python examples/basic_usage.py
```

#### `gadget4_io.py`

Reading GADGET-4 HDF5 files:
- Snapshot headers and particle data
- FOF group catalogs
- Subfind subhalo catalogs

```bash
python examples/gadget4_io.py /path/to/snapshot_034.hdf5
```

### Density Fields

#### `tetra_density.py`

Reference Python implementation of the tetrahedron density algorithm:
- Lagrangian grid construction
- Tetrahedron decomposition (6 per cell)
- Rocchini-Cignoni Monte Carlo sampling
- Comparison with C++ implementation

```bash
python examples/tetra_density.py
```

#### `density_slice.py`

2D projected density from a GADGET-4 snapshot:
- Single or distributed snapshot support
- Thin slab selection with custom center/thickness
- HDF5 output with full metadata
- Visualization with matplotlib

```bash
python examples/density_slice.py snapshot_034.hdf5 -o density.h5 --plot density.png
python examples/density_slice.py snapshot_034.hdf5 -o density.h5 --center 128 --thickness 10
```

#### `density_slice_batch.py`

Batch processing of multiple snapshots for density evolution studies.

```bash
python examples/density_slice_batch.py snapdir/ -o output/ --snapshots 0 10 20 30
```

#### `halo_density.py`

Halo-centric density field extraction:
- Automatic halo catalog detection
- Subbox extraction centered on halos
- Tri-panel visualization (XY, XZ, YZ slices)

```bash
python examples/halo_density.py snapshot_034.hdf5 -o halo.h5 --plot halo.png
python examples/halo_density.py snapshot_034.hdf5 -o halo.h5 --halo-index 0  # Most massive
python examples/halo_density.py snapshot_034.hdf5 -o halo.h5 --width 20      # 20 Mpc/h box
```

### ORIGAMI Morphology

#### `origami_morphology.py`

Full ORIGAMI workflow:
- Morphological classification (void/wall/filament/halo)
- 3D density computation
- Density sampling at particle positions
- Grid deposition of morphology fields
- Conditional PDFs P(1+δ | class)
- 6-panel diagnostic visualization

```bash
python examples/origami_morphology.py snapshot_034.hdf5 -o origami.h5 --plot origami.png
python examples/origami_morphology.py snapshot_034.hdf5 -o origami.h5 --resolution 256
```

#### `overdensity_pdf_origami.py`

Overdensity PDF analysis by morphological class:
- Histograms and PDFs for all particles and per class
- HDF5 output with bin edges and statistics
- 2x1 figure with shared x-axis (histograms top, PDFs bottom)

```bash
python examples/overdensity_pdf_origami.py
```

### Statistics

#### `histogram.py`

Histogram computation module:
- Configuring linear/log binning
- Defining spatial regions (HistBox)
- Using HistManager for parallel computation

```bash
python examples/histogram.py
```

#### `density_pdf.py` / `density_pdf_batch.py`

Overdensity PDF computation:
- Full-box and subregion PDFs
- Batch processing across snapshots
- Evolution of PDF with scale factor

## Quick Reference

```python
import _asymptotic_tetra as at

# List available modules
print([m for m in dir(at) if not m.startswith('_')])
# ['cosmo', 'density', 'geom', 'halo', 'io', 'math', 'origami', 'render', 'stats']

# Check HDF5 support
print(f"HDF5 support: {at.io.HAS_HDF5}")

# Basic vector operations
v = at.Vec3f(1, 2, 3)
print(f"Norm: {v.norm()}")

# Tetrahedron
tet = at.Tetra(at.Vec3f(0,0,0), at.Vec3f(1,0,0), at.Vec3f(0,1,0), at.Vec3f(0,0,1))
print(f"Volume: {tet.volume()}")
```

## Data Requirements

Most examples require GADGET-4 HDF5 snapshots with:
- `PartType1/Coordinates`: Particle positions (N, 3)
- `PartType1/ParticleIDs`: Lagrangian grid IDs (N,)
- `Header/BoxSize`: Simulation box size

Particle IDs should encode Lagrangian positions as:
- z-major (GADGET-4 default): `ID = 1 + z + y*N + x*N²`
- x-major: `ID = 1 + x + y*N + z*N²`

The library auto-detects ID ordering in most cases.
