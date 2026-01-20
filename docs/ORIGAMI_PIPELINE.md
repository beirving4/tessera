# Unified ORIGAMI Pipeline

This document describes the unified ORIGAMI pipeline added to tessera, which consolidates ID handling, sorting, and ORIGAMI computation into a single efficient C++ implementation.

## Overview

The pipeline provides:
1. **Memory efficiency**: Support for N=1024³ simulations (~24GB positions) without OOM
2. **Single source of truth**: No more duplicated helper functions across scripts
3. **Performance**: Parallelized C++ replaces slow Python loops
4. **Clean API**: Single `run_pipeline()` call handles everything

## Files

- `include/origami/pipeline.h` - Header with API definitions
- `src/origami/pipeline.cpp` - C++ implementation
- `python/bindings/origami_bindings.cpp` - Python bindings

## API Reference

### IdOrdering Enum

Specifies how particle IDs encode Lagrangian grid positions.

```python
import tessera as ts

# GADGET-4 uses z-major: ID = 1 + z + y*N + x*N^2
ts.origami.IdOrdering.ZMajor

# Some codes use x-major: ID = 1 + x + y*N + z*N^2
ts.origami.IdOrdering.XMajor

# Auto-detect from particle data (default)
ts.origami.IdOrdering.Auto
```

### PipelineConfig

Configuration for the unified pipeline.

```python
config = ts.origami.PipelineConfig(lagrangian_grid_size=256, box_size=256.0)

# ID handling
config.id_ordering = ts.origami.IdOrdering.Auto  # Auto-detect (default)
config.id_offset = 1  # GADGET-4 uses 1-indexed IDs (default)

# Memory management
config.sort_in_place = True   # Critical for large N (default)
config.positions_already_sorted = False  # Skip sorting if already done

# ORIGAMI parameters
config.n_split = 2  # Domain decomposition for parallelization
config.linear_regime_threshold = 0.99  # Void fraction threshold

# Optional: Density computation (set > 0 to enable)
config.density_output_cells = 128  # Compute density at this resolution
config.density_n_samples = 100     # Monte Carlo samples per tetrahedron
config.particle_mass = 1.0         # For density normalization
config.density_periodic = True     # Periodic boundaries

# Optional: Grid deposition (set > 0 to enable)
config.grid_cells = 64  # Deposit morphology at this resolution

# Optional: Density sampling at particles
config.sample_density_at_particles = True  # Requires density computation

# Threading
config.n_threads = 0  # 0 = auto-detect
config.seed = 0       # Random seed for density (0 = time-based)

# Validation
config.validate_ids = True  # Check ID range and uniqueness
```

### PipelineResult

Complete result from the pipeline.

```python
result = ts.origami.run_pipeline(positions, particle_ids, config)

# Core ORIGAMI outputs (always computed)
result.morphology      # Per-particle class (uint8 array), 0-3
result.n_void, result.n_wall, result.n_filament, result.n_halo  # Counts
result.f_void, result.f_wall, result.f_filament, result.f_halo  # Mass fractions
result.is_linear_regime  # True if f_void >= threshold
result.detected_ordering  # IdOrdering that was used

# Optional: Density field (if density_output_cells > 0)
result.density_3d       # 3D density array, shape (cells, cells, cells)
result.density_cells    # Cells per dimension
result.mean_density     # Mean density value

# Optional: Per-particle density (if sample_density_at_particles)
result.particle_density  # Density at each particle position

# Optional: Grid outputs (if grid_cells > 0)
result.morphology_grid         # Dominant class per cell
result.void_fraction_grid      # Per-cell fractions
result.wall_fraction_grid
result.filament_fraction_grid
result.halo_fraction_grid
result.v_void, result.v_wall, result.v_filament, result.v_halo  # Volume fractions

# Timing diagnostics
result.detection_time_ms   # ID ordering detection
result.sorting_time_ms     # Lagrangian sorting
result.morphology_time_ms  # ORIGAMI computation
result.density_time_ms     # Density field
result.sampling_time_ms    # Particle density sampling
result.grid_time_ms        # Grid deposition
result.total_time_ms       # Total pipeline time
```

## Usage Examples

### Basic Usage

```python
import tessera as ts
import numpy as np

# Load particle data (e.g., from GADGET-4)
positions = ...    # shape (N, 3), float64
particle_ids = ... # shape (N,), int64

# Configure pipeline
config = ts.origami.PipelineConfig(256, 256.0)

# Run pipeline (positions may be modified in-place)
result = ts.origami.run_pipeline(positions, particle_ids, config)

print(f"Halo fraction: {result.f_halo:.2%}")
print(f"Void fraction: {result.f_void:.2%}")
```

### Safe Version (Preserves Input)

```python
# Use run_pipeline_safe if you need to preserve the original positions
result = ts.origami.run_pipeline_safe(positions, particle_ids, config)
# positions array is NOT modified
```

### With Density Computation

```python
config = ts.origami.PipelineConfig(256, 256.0)
config.density_output_cells = 128
config.sample_density_at_particles = True

result = ts.origami.run_pipeline(positions, particle_ids, config)

# Access density outputs
density_field = result.density_3d  # 3D field
particle_densities = result.particle_density  # Per-particle

# Compute conditional PDF P(1+delta | halo)
halo_mask = result.morphology == 3
halo_densities = particle_densities[halo_mask]
```

### With Grid Deposition

```python
config = ts.origami.PipelineConfig(256, 256.0)
config.grid_cells = 64

result = ts.origami.run_pipeline(positions, particle_ids, config)

# Visualize halo fraction in a slice
import matplotlib.pyplot as plt
plt.imshow(result.halo_fraction_grid[32, :, :])  # z=32 slice
plt.colorbar(label='Halo fraction')
```

### Step-by-Step (Low-Level API)

```python
# Detect ID ordering
ordering = ts.origami.detect_id_ordering(positions, particle_ids, 256, 256.0)
print(f"Ordering: {ts.origami.id_ordering_name(ordering)}")

# Validate IDs
ts.origami.validate_particle_ids(particle_ids, 256, id_offset=1)

# Sort in-place
ts.origami.sort_to_lagrangian_inplace(positions, particle_ids, 256, ordering)

# Or sort out-of-place (creates a copy)
sorted_pos = ts.origami.sort_to_lagrangian(positions, particle_ids, 256, ordering)
```

## Memory Management

For N=1024³ with float64 positions (~24GB):

| Approach | Memory Required |
|----------|-----------------|
| Python (copy-based) | ~48 GB |
| In-place sorting | ~33 GB |
| **Savings** | **~15 GB** |

### In-Place Sorting Algorithm

The in-place sorting uses cycle decomposition:
1. Build inverse permutation array (~8GB for N=1024³)
2. Build visited flags (~1GB for bools)
3. Apply permutation by following cycles (no position copy needed)

This is critical for running on systems with limited memory.

## Implementation Notes

### ID Ordering Detection

Uses displacement correlations between neighboring IDs with periodic boundary handling:
1. For particles with IDs that differ by 1, compute the minimum-image displacement
2. Compare to expected spacing for both x-major and z-major conventions
3. The convention with smaller error is selected

This robust algorithm works even when particles have wrapped around periodic boundaries.

### Thread Safety

- All read operations are thread-safe
- `run_pipeline` and `sort_to_lagrangian_inplace` modify the input positions array
- The pipeline uses OpenMP for parallelization

### Backwards Compatibility

The existing functions (`compute_morphology`, `sample_density_at_particles`, `deposit_morphology_to_grid`) remain unchanged and can still be used directly.

## Migration Guide

### Before (Duplicated Python Code)

```python
# This code was duplicated across multiple scripts
def detect_id_ordering(pos, ids, N, box_size):
    # ... 30+ lines of Python
    pass

def sort_to_lagrangian(pos, ids, N, ordering):
    # ... 20+ lines of Python
    pass

# Manual workflow
ordering = detect_id_ordering(pos, ids, 256, 256.0)
sorted_pos = sort_to_lagrangian(pos, ids, 256, ordering)
config = ts.origami.OrigamiConfig(256, 256.0)
result = ts.origami.compute_morphology(sorted_pos, config)
```

### After (Unified Pipeline)

```python
config = ts.origami.PipelineConfig(256, 256.0)
result = ts.origami.run_pipeline(positions, particle_ids, config)
# Everything handled automatically!
```
