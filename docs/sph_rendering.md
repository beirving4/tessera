# SPH Density Rendering

This document describes tessera's high-performance SPH (Smoothed Particle Hydrodynamics)
density renderer, an alternative to phase-space tessellation.

## Overview

SPH rendering computes density fields by smoothing particle contributions with a
kernel function. Unlike tessellation, SPH has no artifacts from elongated tetrahedra
and works without requiring Lagrangian particle ordering.

### When to Use SPH

| Situation | Recommendation |
|-----------|----------------|
| Scale factor a < 5 | Use tessellation (more accurate) |
| Scale factor a > 10 | Use SPH (no streak artifacts) |
| Halo-centered visualization | SPH is ideal |
| Full-box density field | Tessellation is faster |
| No Lagrangian IDs available | SPH is the only option |

## Quick Start

```python
import tessera as ts
import numpy as np

# Load particle positions
positions = np.random.uniform(0, 100, (1000000, 3))

# Render 2D density projection
result = ts.density.render_sph_density_2d(
    positions,
    center=(50.0, 50.0, 50.0),
    box_width=20.0,
    output_cells=256,
    sim_box_size=100.0
)

print(f"Shape: {result.density.shape}")
print(f"Render time: {result.render_time_ms:.1f} ms")
```

## API Reference

### High-Level Function

```python
ts.density.render_sph_density_2d(
    positions,           # numpy array (N, 3)
    center,              # tuple (x, y, z) - region center
    box_width,           # float - width of render region
    output_cells=256,    # int - output grid resolution
    sim_box_size=0.0,    # float - simulation box size (for periodicity)
    projection_axis=2,   # int - 0=x, 1=y, 2=z
    kernel=SPHKernel.CUBIC_SPLINE,
    smoothing_length=0.0,  # 0 = adaptive
    n_threads=0          # 0 = auto
) -> SPHResult2D
```

### SPHConfig

Full configuration for SPH rendering:

```python
config = ts.density.SPHConfig()

# Grid settings
config.output_cells = 256        # Output resolution
config.box_size = 10.0           # Render region size
config.center = (50, 50, 50)     # Region center

# SPH parameters
config.kernel = ts.density.SPHKernel.CUBIC_SPLINE
config.smoothing_length = 0.0    # 0 = adaptive
config.neighbors_target = 32     # Target neighbors for adaptive h
config.h_min = 0.0               # Min smoothing length (0 = no limit)
config.h_max = 0.0               # Max smoothing length (0 = no limit)

# Simulation parameters
config.sim_box_size = 100.0      # For periodic boundaries
config.periodic = True
config.particle_mass = 1.0

# Performance
config.n_threads = 0             # 0 = auto (use all cores)
config.projection_axis = 2       # Project along z
```

### Convenience Constructor

For halo-centered rendering:

```python
config = ts.density.SPHConfig.for_halo(
    halo_center=(50.0, 50.0, 50.0),
    box_width=10.0,
    sim_box=100.0,
    resolution=256
)
```

### SPHRenderer Class

For repeated rendering with the same configuration:

```python
renderer = ts.density.SPHRenderer(config)

# Render multiple times
result1 = renderer.render_2d(particles1)
result2 = renderer.render_2d(particles2)

# Also available
result_3d = renderer.render_3d(particles)
```

### SPHParticles

Structure-of-arrays particle container:

```python
# From numpy arrays
particles = ts.density.SPHParticles(
    x, y, z,           # Position arrays (float32)
    mass=None,         # Optional mass array
    h=None             # Optional smoothing length array
)

# Or build incrementally
particles = ts.density.SPHParticles()
particles.add(1.0, 2.0, 3.0, mass=1.0, h=0.5)
```

### SPHResult2D

Result of 2D rendering:

```python
result.density          # numpy array (cells, cells)
result.counts           # particles per cell
result.cells            # grid dimension
result.cell_width       # physical cell size
result.mean_density     # mean surface density
result.total_mass       # total mass in projection
result.render_time_ms   # rendering time
result.kernel_evals     # total kernel evaluations
result.extent           # (x_min, x_max, y_min, y_max)
```

## Kernel Functions

### Available Kernels

| Kernel | Support Radius | Speed | Smoothness |
|--------|---------------|-------|------------|
| `CUBIC_SPLINE` | 2h | Fastest | Good |
| `WENDLAND_C2` | 2h | Moderate | Better |
| `WENDLAND_C4` | 2h | Slower | Best |
| `QUINTIC_SPLINE` | 3h | Slowest | Best |

### Kernel Selection

```python
# Fastest - good for most cases
config.kernel = ts.density.SPHKernel.CUBIC_SPLINE

# Smoother - reduces noise in sparse regions
config.kernel = ts.density.SPHKernel.WENDLAND_C2

# String conversion
name = ts.density.sph_kernel_to_string(config.kernel)  # "cubic_spline"
kernel = ts.density.sph_kernel_from_string("wendland_c2")
```

## Smoothing Length

### Adaptive Smoothing (Recommended)

When `smoothing_length=0`, tessera automatically estimates smoothing lengths
based on local particle density:

```python
# Adaptive smoothing (default)
config.smoothing_length = 0.0
config.neighbors_target = 32  # Target ~32 neighbors within kernel

# You can also compute smoothing lengths manually
renderer.compute_smoothing_lengths(particles)
```

### Fixed Smoothing

For uniform smoothing across all particles:

```python
config.smoothing_length = 0.5  # Fixed h for all particles
```

### Estimating Smoothing Length

```python
# Estimate from particle count and box size
h = ts.density.estimate_smoothing_length(
    n_particles=1000000,
    box_size=100.0,
    neighbors_target=32
)
```

## Performance

### Benchmarks

Typical performance on Apple M1 (8 cores):

| Particles | Resolution | Kernel | Time |
|-----------|------------|--------|------|
| 1M | 256² | Cubic | 0.2s |
| 10M | 256² | Cubic | 1.5s |
| 16M | 256² | Cubic | 2.0s |
| 16M | 512² | Cubic | 8.0s |

### Optimization Tips

1. **Use cubic spline** - Wendland kernels are 5-10x slower due to numerical integration
2. **Match resolution to science** - Higher resolution increases kernel evaluations
3. **Prefilter particles** - Only include particles near the render region
4. **Use adaptive smoothing** - Better quality than fixed smoothing

### Comparison with Alternatives

| Method | 16M particles, 256² | Notes |
|--------|---------------------|-------|
| tessera SPH | ~2s | C++/SIMD/OpenMP |
| pynbody | ~30-60s | Python with C kernel |
| scipy KDE | ~120s | Pure Python |

## Examples

### Halo Evolution Visualization

```python
import tessera as ts
import h5py
import numpy as np

def render_halo_snapshot(snap_path, halo_pos, box_width, sim_box):
    """Render halo-centered density map."""
    with h5py.File(snap_path, 'r') as f:
        pos = f['PartType1/Coordinates'][:]
        scale_factor = f['Header'].attrs['Time']

    # Convert to comoving if needed
    comoving_width = box_width / scale_factor

    result = ts.density.render_sph_density_2d(
        pos,
        center=tuple(halo_pos),
        box_width=comoving_width,
        output_cells=256,
        sim_box_size=sim_box
    )

    return result.density, result.mean_density

# Render evolution
for snap in [34, 50, 60, 74]:  # a = 1, 10, 30, 100
    density, mean = render_halo_snapshot(
        f"snapshot_{snap:03d}.hdf5",
        halo_positions[snap],
        box_width=10.0,
        sim_box=256.0
    )
    overdensity = density / mean
    # Save or visualize...
```

### Comparing SPH with Tessellation

```python
import tessera as ts

# SPH rendering
sph_result = ts.density.render_sph_density_2d(
    positions,
    center=(50, 50, 50),
    box_width=10.0,
    output_cells=256,
    sim_box_size=100.0
)

# Tessellation (requires Lagrangian-sorted positions)
config = ts.density.TetraDensityConfig(
    lagrangian_grid_size=128,
    output_cells=256,
    box_size=100.0
)
tess_result = ts.density.compute_tetra_density_2d_projection(
    sorted_positions, config, projection_axis=2
)

print(f"SPH time: {sph_result.render_time_ms:.1f} ms")
print(f"Tessellation time: {tess_result.total_time_ms:.1f} ms")
```

### 3D Volume Rendering

```python
config = ts.density.SPHConfig()
config.output_cells = 128
config.box_size = 20.0
config.center = (50, 50, 50)
config.sim_box_size = 100.0

renderer = ts.density.SPHRenderer(config)
result = renderer.render_3d(particles)

# result.density has shape (128, 128, 128)
```

## Implementation Details

### Algorithm

1. **Coordinate transformation**: Particles are transformed to render-centered coordinates
2. **Spatial hash grid**: O(1) neighbor lookup using hash grid with cell size = kernel support
3. **Parallel rendering**: OpenMP distributes grid cells across threads
4. **SIMD kernel evaluation**: AVX2 evaluates 8 kernel values per instruction
5. **2D projection**: Line-of-sight integration of 3D kernel

### Memory Usage

- Particle storage: 12 bytes/particle (float32 x, y, z)
- Output grid: 8 bytes/cell (float64 density)
- Spatial hash: ~4 bytes/particle overhead

For 16M particles at 256² resolution: ~250 MB total

### Thread Safety

The SPHRenderer is thread-safe for rendering. Multiple threads can call
`render_2d()` concurrently on different particle sets with the same renderer.

## Troubleshooting

### Empty or Zero Density

Check that:
- Particles are within the render region (center ± box_width/2)
- `sim_box_size` is set correctly for periodic boundaries
- Smoothing length is not too small

### Noisy Results

Try:
- Increasing `neighbors_target` (e.g., 64 instead of 32)
- Using `WENDLAND_C2` kernel instead of `CUBIC_SPLINE`
- Increasing output resolution

### Slow Performance

Ensure:
- Using `CUBIC_SPLINE` kernel (fastest)
- OpenMP is enabled (`n_threads > 1` or `0` for auto)
- Not rendering unnecessarily high resolution

## References

- Monaghan 1992, ARA&A 30, 543 (SPH review)
- Dehnen & Aly 2012, MNRAS 425, 1068 (Kernel comparison)
- Price 2012, J. Comp. Phys. 231, 759 (Smoothing length)
