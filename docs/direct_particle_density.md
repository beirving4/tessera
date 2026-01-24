# Direct Particle Density: Memory-Efficient Density Estimation

## Overview

This document describes a memory-efficient approach for computing density at particle positions without constructing a full 3D density grid. This is particularly valuable for computing overdensity PDFs per ORIGAMI morphology class on large simulations (N ≥ 1024³) where memory constraints make the traditional grid-based approach infeasible.

## Theoretical Foundation

### Phase-Space Tessellation (Abel, Hahn & Kaehler 2012)

The foundational insight comes from ["Tracing the Dark Matter Sheet in Phase Space"](https://arxiv.org/abs/1111.3944) (Abel, Hahn & Kaehler, MNRAS 2012):

> **Particles are not mass carriers—they are massless vertices of a tessellation.** The mass is uniformly distributed within each tetrahedron of the tessellation.

Key concepts:
- Dark matter in N-body simulations traces a 3D "sheet" in 6D phase space
- This sheet can be tessellated into tetrahedra using particle positions as vertices
- The Lagrangian grid provides natural connectivity: each cell decomposes into 6 tetrahedra
- Mass is distributed uniformly within tetrahedra, not concentrated at particles

### Density from Tetrahedra Volumes (Hahn, Abel & Kaehler 2013)

["A New Approach to Simulating Collisionless Dark Matter Fluids"](https://arxiv.org/abs/1210.6652) extends this to density computation:

> The tessellation allows one to "project the distribution function into configuration space to obtain highly accurate densities... **all without averaging over control volumes**."

This explicitly supports bypassing 3D grids entirely. Density at any point is defined by the local tetrahedra geometry, not by grid-based averaging.

### Mathematical Formulation (PS-DTFE)

The [Phase-Space Delaunay Tessellation Field Estimator](https://arxiv.org/abs/2402.16234) provides the explicit formula for density at vertex i:

```
ρᵢ = (d + 1) × mᵢ / V(Wᵢ)
```

Where:
- `d = 3` (spatial dimensions)
- `mᵢ` = mass associated with vertex i
- `V(Wᵢ)` = total volume of all tetrahedra sharing vertex i (the "star")

For our Lagrangian tessellation with uniform particle mass:

```
ρ_particle = Σ(m_tetra) / Σ(V_tetra)
           = (n_adjacent_tetra × m_tetra) / Σ(V_tetra)
```

Where the sum is over all tetrahedra adjacent to the particle.

### Multi-Stream Regions

In caustic regions where the dark matter sheet folds over itself, multiple tetrahedra can overlap in configuration space. The PS-DTFE paper notes:

> "In multi-stream regions (where points lie in multiple simplices), densities from all containing simplices sum together."

Our Lagrangian tessellation naturally handles this: a particle's density includes contributions from all adjacent tetrahedra, which may represent different streams.

## Connection to gotetra and tessera

### gotetra's Approach

The original [gotetra](https://github.com/phil-mansfield/gotetra) code computes density via Monte Carlo sampling:

1. Decompose each Lagrangian cell into 6 tetrahedra
2. For each tetrahedron, generate random sample points
3. Deposit sample mass onto a 3D Eulerian grid
4. Normalize by cell volume to get density

This approach:
- Requires O(grid³) memory for the 3D grid
- Requires O(N³ × n_samples) Monte Carlo operations
- Is excellent for producing smooth 3D density fields

### tessera's Current Implementation

tessera reimplements gotetra's algorithm in C++ with optimizations:
- Streaming tetrahedra indices (no O(N³) index pre-allocation)
- Thread-local density grids to avoid atomic operations
- OpenMP parallelization

For PDF computation, tessera's `run_pipeline()`:
1. Computes full 3D density field (thread-local grids)
2. Samples the grid at particle positions (trilinear interpolation)
3. Computes histograms per morphology class

Memory for N=1024³ with 8 threads:
- Thread-local grids: 8 × 8 GB = **64 GB**
- Plus positions, IDs, morphology: ~40 GB
- Total: **>100 GB** (causes OOM on 256 GB nodes)

### Direct Particle Density Approach

The new approach computes density directly at particles:

1. Decompose each Lagrangian cell into 6 tetrahedra (same as before)
2. For each tetrahedron, compute its Eulerian volume
3. Accumulate volume at each of the 4 vertices (particles)
4. Particle density = total_adjacent_mass / total_adjacent_volume

Memory for N=1024³:
- Particle volume accumulator: 8 GB
- No 3D grid needed
- Total: **~48 GB** (fits comfortably on 256 GB nodes)

## Algorithm Details

### Tetrahedra Decomposition

Each Lagrangian cell (8 vertices) is decomposed into 6 tetrahedra using the standard decomposition. For a cell with corners at grid positions (i,j,k), the 8 vertices are:

```
v0 = (i,   j,   k  )    v4 = (i,   j,   k+1)
v1 = (i+1, j,   k  )    v5 = (i+1, j,   k+1)
v2 = (i,   j+1, k  )    v6 = (i,   j+1, k+1)
v3 = (i+1, j+1, k  )    v7 = (i+1, j+1, k+1)
```

The 6 tetrahedra are:
```
Tetra 0: v0, v1, v2, v4
Tetra 1: v1, v2, v4, v5
Tetra 2: v2, v4, v5, v6
Tetra 3: v1, v2, v3, v5
Tetra 4: v2, v3, v5, v6
Tetra 5: v3, v5, v6, v7
```

### Volume Calculation

The signed volume of a tetrahedron with vertices (a, b, c, d) is:

```
V = (1/6) × |det([b-a, c-a, d-a])|
```

Or equivalently:
```
V = (1/6) × |(b-a) · ((c-a) × (d-a))|
```

### Vertex Volume Accumulation

For each tetrahedron:
- Compute Eulerian volume V_tetra
- Add V_tetra / 4 to each of the 4 vertices

After processing all tetrahedra:
```
ρᵢ = (n_adjacent × m_tetra) / V_accumulated[i]
   = total_mass_at_vertex / V_accumulated[i]
```

Since each Lagrangian cell contributes 6 tetrahedra and each particle is shared by 8 cells:
- Each particle is a vertex of 6 × 8 = 48 tetrahedra
- Each tetrahedron has mass = particle_mass (total cell mass / 6)
- Total mass at vertex = 48 × (particle_mass / 6) = 8 × particle_mass

Wait, let me reconsider. Each Lagrangian cell has 8 vertices sharing its mass equally. So:
- Cell mass = 8 × particle_mass (from 8 corner particles contributing 1/8 each? No...)

Actually, in the standard interpretation:
- Total mass in simulation = N³ × particle_mass
- Each Lagrangian cell contains mass = particle_mass (one particle's worth)
- This mass is distributed among 6 tetrahedra: m_tetra = particle_mass / 6

For the accumulated volume at particle i:
```
ρᵢ = Σ(m_tetra) / Σ(V_tetra/4)
   = Σ(particle_mass/6) / Σ(V_tetra/4)
```

The number of tetrahedra adjacent to a particle (in the interior) is 24:
- 8 surrounding Lagrangian cells
- 3 tetrahedra per cell touch each vertex (on average)

So for an interior particle:
```
ρᵢ = 24 × (particle_mass/6) / (V_total/4)
   = 4 × particle_mass / (V_total/4)
   = 16 × particle_mass / V_total
```

Hmm, let me reconsider the normalization. The key insight is:
- Each tetrahedron contributes 1/4 of its volume to each vertex
- Each tetrahedron contributes 1/4 of its mass to each vertex
- So ρᵢ = Σ(m_tetra/4) / Σ(V_tetra/4) = Σ(m_tetra) / Σ(V_tetra)

This simplifies to:
```
ρᵢ = total_mass_in_adjacent_tetrahedra / total_volume_of_adjacent_tetrahedra
```

### Handling Periodic Boundaries

Tetrahedra near box boundaries may wrap around. The volume calculation must use the minimum image convention for vertex separations.

### Mean Density Normalization

For overdensity (δ = ρ/ρ̄ - 1) or normalized density (ρ/ρ̄), we need the mean density:

```
ρ̄ = total_mass / box_volume = (N³ × particle_mass) / box_size³
```

## Performance Comparison

### Computational Complexity

| Approach | Operations per Tetrahedron | Total for N=1024³ |
|----------|---------------------------|-------------------|
| Monte Carlo (100 samples) | ~100 RNG + deposit ops | 640 billion |
| Direct Volume | 1 determinant + 4 adds | 6.4 billion |

**Speedup: ~100x fewer floating-point operations**

### Memory Usage

| Resource | Monte Carlo (8 threads) | Direct Volume |
|----------|------------------------|---------------|
| 3D density grid | 64 GB | 0 |
| Particle density array | 8 GB | 8 GB |
| Positions + IDs | 32 GB | 32 GB |
| Morphology | 1 GB | 1 GB |
| **Total** | **~105 GB** | **~41 GB** |

**Memory reduction: ~2.5x**

### Expected Runtime

For `save_density_pdf_results.py` on N=1024³:
- Current: ~20 minutes per snapshot (with OOM risk)
- Proposed: ~2-5 minutes per snapshot (estimated)

## Implementation Plan

### New Functions

1. `compute_particle_density()` - Compute density at all particles from tetrahedra volumes
2. Integration with `run_pipeline()` - New config option to use direct density

### API Addition

```cpp
// In include/density/tetra_density.h
struct ParticleDensityConfig {
    int lagrangian_grid_size;
    double box_size;
    double particle_mass;
    int n_threads;
    bool periodic;
};

struct ParticleDensityResult {
    std::vector<double> density;  // Per-particle density
    double mean_density;
    double total_time_ms;
};

ParticleDensityResult compute_particle_density(
    const double* sorted_positions,  // Lagrangian-sorted
    const ParticleDensityConfig& config
);
```

### Python Binding

```python
import tessera as ts

# Direct particle density (no 3D grid)
config = ts.density.ParticleDensityConfig()
config.lagrangian_grid_size = 1024
config.box_size = 256.0
config.particle_mass = 1.0

result = ts.density.compute_particle_density(sorted_positions, config)
particle_densities = np.array(result.density)
```

## References

1. Abel, T., Hahn, O., & Kaehler, R. 2012, MNRAS, 427, 61 - [Tracing the dark matter sheet in phase space](https://arxiv.org/abs/1111.3944)

2. Hahn, O., Abel, T., & Kaehler, R. 2013, MNRAS, 434, 1171 - [A new approach to simulating collisionless dark matter fluids](https://arxiv.org/abs/1210.6652)

3. Kaehler, R., Hahn, O., & Abel, T. 2012, IEEE TVCG, 18, 2078 - [A Novel Approach to Visualizing Dark Matter Simulations](https://ieeexplore.ieee.org/document/6327223)

4. Feldbrugge, J., van de Weygaert, R.,"; Hidding, J., & Feldbrugge, J. 2024 - [Phase-space Delaunay tessellation field estimator](https://arxiv.org/abs/2402.16234)

5. Mansfield, P. - [gotetra: Phase-space tessellation in Go](https://github.com/phil-mansfield/gotetra)
