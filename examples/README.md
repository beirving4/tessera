# tessera Examples

This directory contains example scripts demonstrating how to use the tessera library.

## Prerequisites

**Python 3.10+** is required.

### Quick Install (Recommended)

```bash
pip install .
```

### Building from Source

For development or more control over the build:

```bash
cd /path/to/tessera
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_HDF5=ON
make -j4
```

Add the build directory to your Python path:

```bash
export PYTHONPATH=/path/to/tessera/build:$PYTHONPATH
```

### Additional Dependencies

Some examples use the `visualize/` and `utils/` modules from the repository root:

```python
from visualize import plot_density_slice, plot_morphology_slice
from utils import load_snapshot, sort_positions_lagrangian
```

These are automatically available when running from the repository root.

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
- Optional jackknife resampling for uncertainty estimation
- HDF5 output with bin edges and statistics

```bash
python examples/overdensity_pdf_origami.py snapshot_034.hdf5 -o pdf.h5 --plot pdf.png
python examples/overdensity_pdf_origami.py snapshot_034.hdf5 -o pdf.h5 --jackknife  # With uncertainties
```

#### `origami_slice_render.py`

2D thin-slice visualization of ORIGAMI morphology:
- Slice through 3D morphology grid
- Discrete colormap for void/wall/filament/halo
- Publication-quality figures

```bash
python examples/origami_slice_render.py snapshot_034.hdf5 --plot morphology.png
```

### Time-Series Visualization

#### `time_series_pipeline.py`

Complete pipeline for Diemer-style cosmic evolution visualization:
- Generates 2D projections from all snapshots
- Builds static time-series image (x=time, y=space)
- Creates evolution animation with callout annotations

```bash
python examples/time_series_pipeline.py \
    --snapshot-dir /path/to/snapshots \
    --output-dir ./output \
    --n-threads 4
```

#### `generate_projections.py`

Generate 2D density projections from a series of snapshots:
- Processes multiple snapshots in parallel
- Saves projections to HDF5 with metadata

```bash
python examples/generate_projections.py --snapshot-dir /path/to/snapshots --output-dir ./output
```

#### `build_time_series.py`

Build static time-series image from pre-computed projections:
- X-axis represents cosmic time (scale factor)
- Y-axis represents spatial position

```bash
python examples/build_time_series.py --projections ./output/density_projections.h5 --output time_series.png
```

#### `animate_evolution.py`

Create evolution animation with callout annotations:
- Configurable annotation template for cosmic events
- MP4/GIF output formats

```bash
python examples/animate_evolution.py --projections ./output/density_projections.h5 --output evolution.mp4
```

### Halo Evolution

#### `halo_evolution/halo_evolution_pipeline.py`

Complete pipeline for visualizing the evolution of a dark matter halo across cosmic time:
- High-performance C++ merger tree reading with SIMD-optimized search
- Staged data loading (~36x faster first lookup by loading only search data initially)
- Branch tracing with coordinate unwrapping for smooth tracking
- Density field rendering centered on halo position at each epoch
- Multi-panel figures and MP4 animations

```bash
# Trace most massive halo and generate visualization
python examples/halo_evolution/halo_evolution_pipeline.py \
    --tree-file /path/to/trees.hdf5 \
    --snapshot-dir /path/to/snapshots \
    --output-dir ./halo_evolution_output

# Specific tree with scale factor limits
python examples/halo_evolution/halo_evolution_pipeline.py \
    --tree-file /path/to/trees.hdf5 \
    --snapshot-dir /path/to/snapshots \
    --output-dir ./output \
    --tree-id 0 \
    --a-min 0.1

# Multi-panel figure only (no animation)
python examples/halo_evolution/halo_evolution_pipeline.py \
    --tree-file /path/to/trees.hdf5 \
    --snapshot-dir /path/to/snapshots \
    --output-dir ./output \
    --no-animation \
    --epochs 0.1 1.0 10.0 100.0
```

**Note:** On macOS, there may be HDF5 library conflicts between h5py and tessera. Use `--extract-branch-only` to save branch info, then run density rendering separately with `--branch-file`.

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
import tessera as ts

# List available modules
print([m for m in dir(ts) if not m.startswith('_')])
# ['cosmo', 'density', 'geom', 'halo', 'io', 'math', 'origami', 'render', 'stats']

# Check HDF5 support
print(f"HDF5 support: {ts.io.HAS_HDF5}")

# Basic vector operations
v = ts.Vec3f(1, 2, 3)
print(f"Norm: {v.norm()}")

# Tetrahedron
tet = ts.Tetra(ts.Vec3f(0,0,0), ts.Vec3f(1,0,0), ts.Vec3f(0,1,0), ts.Vec3f(0,0,1))
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
