# Artifact Mitigation in Tessera Density Fields

This document describes the artifact mitigation features in tessera for handling
tessellation artifacts that appear at late cosmological times.

## Background

### The Problem

At late cosmological times (scale factor a >> 1), the tessellation-based density
computation produces characteristic radial streak artifacts. These artifacts occur
because:

1. **Matter flows radially** toward halos as the universe expands
2. **Tetrahedra become highly elongated** along these flow directions
3. **Monte Carlo samples** from elongated tetrahedra deposit in streaky patterns
4. **The result**: artificial radial features that don't represent real structure

### When Artifacts Appear

- **a < 5**: Artifacts are minimal; pure tessellation works well
- **5 ≤ a < 10**: Artifacts begin appearing in void regions
- **a > 10**: Artifacts become prominent, especially around isolated halos
- **a = 100** (island universe): Severe artifacts in all low-density regions

### Why `particle_counts` Alone Isn't Enough

The `particle_counts` feature (sample counts per grid cell) identifies regions
with few samples, which are often unreliable. However, at extreme scale factors,
artifacts occur in regions that DO have many samples - just from elongated
tetrahedra depositing along radial directions. The counts are high, but the
density pattern is artificial.

## Four Mitigation Approaches

Tessera provides four complementary approaches to mitigate these artifacts:

### 1. Gaussian Smoothing (Post-Processing)

**Best for**: Quick fixes, mild artifacts

Apply Gaussian smoothing to blur out streak artifacts while preserving overall
structure.

```python
from tessera.utils import gaussian_smooth_density, adaptive_smooth_density

# Simple Gaussian smoothing
smoothed = gaussian_smooth_density(density, sigma=1.5)

# Adaptive smoothing (stronger in low-count regions)
smoothed = adaptive_smooth_density(
    density, particle_counts,
    sigma_min=0.5,  # Well-sampled regions
    sigma_max=3.0   # Poorly-sampled regions
)
```

**Pros**:
- Simple, fast, no C++ changes needed
- Preserves total mass
- Can use `particle_counts` for adaptive kernel size

**Cons**:
- Blurs real structure along with artifacts
- Doesn't address root cause

### 2. Tetrahedra Geometry Filtering (C++ Core)

**Best for**: Preventing artifacts at the source

Skip or downweight samples from highly elongated tetrahedra before they create
artifacts.

```python
import tessera as ts

config = ts.density.TetraDensityConfig()
config.max_aspect_ratio = 50.0      # Skip tetrahedra with aspect ratio > 50
config.elongation_downweight = 0.5  # Or downweight by 50%
config.aspect_ratio_soft_threshold = 10.0  # Start downweighting at ratio > 10

result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)

# Check how many tetrahedra were filtered
print(f"Skipped {result.skipped_tetrahedra} elongated tetrahedra")
print(f"Mean aspect ratio: {result.mean_aspect_ratio:.1f}")
```

**Config options**:
- `max_aspect_ratio`: Hard cutoff - skip tetrahedra above this ratio (0 = disabled)
- `elongation_downweight`: Soft weight factor for elongated tetrahedra [0, 1]
- `aspect_ratio_soft_threshold`: Start downweighting above this ratio

**Pros**:
- Addresses root cause
- Preserves sharp features in well-behaved regions
- Provides quality metrics for diagnostics

**Cons**:
- May reduce total deposited mass (mass conservation affected)
- Requires tuning thresholds for each scale factor

### 3. SPH Density Rendering (C++ Core)

**Best for**: Extreme scale factors (a > 10) where tessellation breaks down

Bypass tessellation entirely and use the built-in SIMD-optimized SPH renderer.
This is a high-performance C++ implementation with OpenMP parallelization.

```python
import tessera as ts

# High-level convenience function
result = ts.density.render_sph_density_2d(
    positions,                    # Raw particle positions (N, 3)
    center=(50.0, 50.0, 50.0),   # Region center
    box_width=10.0,              # Render region width
    output_cells=256,            # Output resolution
    sim_box_size=256.0,          # Simulation box size (for periodicity)
    projection_axis=2,           # Project along z-axis
    kernel=ts.density.SPHKernel.CUBIC_SPLINE,
    smoothing_length=0.0,        # 0 = adaptive
    n_threads=0                  # 0 = auto
)

density = result.density
print(f"Rendered in {result.render_time_ms:.1f} ms")

# Or with full configuration control
config = ts.density.SPHConfig.for_halo(
    halo_center=(50.0, 50.0, 50.0),
    box_width=10.0,
    sim_box=256.0,
    resolution=256
)
renderer = ts.density.SPHRenderer(config)
result = renderer.render_2d(particles)
```

**Available kernels**:
- `SPHKernel.CUBIC_SPLINE` - M4 cubic spline (fastest, recommended)
- `SPHKernel.WENDLAND_C2` - Smoother than cubic
- `SPHKernel.WENDLAND_C4` - Very smooth
- `SPHKernel.QUINTIC_SPLINE` - M6 quintic (highest accuracy)

**Pros**:
- No tessellation artifacts by design
- Works without Lagrangian ordering (simpler pipeline)
- 15-30x faster than Python-based alternatives (pynbody)
- SIMD-optimized kernel evaluation (AVX2)
- OpenMP parallelization

**Cons**:
- May over-smooth fine structure compared to tessellation
- Different density estimator (not exactly comparable to tessellation)

See [SPH Rendering Documentation](sph_rendering.md) for full details.

### 4. Hybrid Approach (Automatic Method Selection)

**Best for**: Processing multiple snapshots with varying scale factors

Let tessera automatically choose the best method based on scale factor.

```python
from tessera.utils import compute_density_auto, HybridDensityConfig

config = HybridDensityConfig(
    tessellation_max_scale_factor=5.0,   # Pure tessellation for a < 5
    sph_min_scale_factor=10.0,           # SPH for a > 10
    smoothing_sigma=1.5,                 # Smoothing for transitional range
    adaptive_smoothing=True
)

# Automatic method selection based on scale factor
result = compute_density_auto(
    positions, particle_ids,
    box_size=256.0,
    scale_factor=100.0,  # Will use SPH for a=100
    config=config,
    projection_axis=2
)

print(f"Method used: {result['method']}")
# Output: "Method used: sph"
```

**Method selection logic**:
- `a < tessellation_max_scale_factor`: Pure tessellation
- `tessellation_max_scale_factor ≤ a < sph_min_scale_factor`: Tessellation + smoothing
- `a ≥ sph_min_scale_factor`: SPH

## Usage Examples

### Island Universe Visualization

For visualizing halo evolution to a=100 (island universe research):

```python
from tessera.utils import compute_density_auto, HybridDensityConfig
import numpy as np

# Configure for island universe visualization
config = HybridDensityConfig(
    tessellation_max_scale_factor=5.0,
    sph_min_scale_factor=20.0,
    smoothing_sigma=2.0,
    output_cells=512
)

# Process multiple snapshots
for snap_id in [34, 50, 60, 74]:  # a = 1, 10, 30, 100
    positions, ids, box_size, scale_factor = load_snapshot(snap_id)

    result = compute_density_auto(
        positions, ids,
        box_size=box_size,
        scale_factor=scale_factor,
        config=config,
        projection_axis=2
    )

    print(f"Snap {snap_id} (a={scale_factor:.1f}): method={result['method']}")

    # Visualize
    plot_density(result['density'], ...)
```

### Comparing Methods

```python
from tessera.utils import compare_methods

# Compare all methods on same data
results = compare_methods(
    positions, particle_ids,
    box_size=256.0,
    scale_factor=100.0,
    methods=['tessellation', 'tessellation_smoothed', 'sph'],
    projection_axis=2
)

for method, result in results.items():
    if 'error' not in result:
        print(f"{method}: mean={result['density'].mean():.2e}")
```

## Quality Metrics

### From Tessellation

```python
result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)

# Sample count statistics
counts = np.array(result.particle_counts).reshape(N, N)
print(f"Zero-count pixels: {(counts == 0).sum()}")
print(f"Low-count pixels: {(counts < 2).sum()}")

# Geometry filtering statistics (if enabled)
print(f"Skipped tetrahedra: {result.skipped_tetrahedra}")
print(f"Mean aspect ratio: {result.mean_aspect_ratio:.1f}")
print(f"Max aspect ratio: {result.max_observed_aspect_ratio:.1f}")
```

### From Hybrid Approach

```python
result = compute_density_auto(...)

metrics = result['quality_metrics']
print(f"Quality metrics: {metrics}")
```

## Recommendations

| Scale Factor | Recommended Approach |
|--------------|---------------------|
| a < 5 | Pure tessellation |
| 5 ≤ a < 10 | Tessellation with geometry filtering |
| 10 ≤ a < 30 | Tessellation + adaptive smoothing |
| a ≥ 30 | SPH-based density |

For consistency across time series, consider using the hybrid approach with
fixed thresholds to ensure smooth transitions between methods.

## References

- Abel, Hahn & Kaehler 2012, MNRAS 427, 61 (Phase-space tessellation)
- Hahn, Abel & Kaehler 2013, MNRAS 434, 1171 (Dark matter sheet approach)
- gotetra: https://github.com/phil-mansfield/gotetra
