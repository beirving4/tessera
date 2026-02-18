# Parallel Scaling Analysis

Analysis of OpenMP parallel performance for the ORIGAMI pipeline on different systems.

**Date:** January 21, 2025
**Grid Size:** N = 256³ = 16,777,216 particles

## Summary

The ORIGAMI morphology computation is **memory-bandwidth limited**, not CPU-limited. Adding more threads provides minimal benefit for the morphology stage because all threads compete for the same memory bus. The main parallel benefits come from sorting and grid operations.

## Test Systems

### Linux Cluster Node

| Spec | Value |
|------|-------|
| CPU | AMD EPYC 7543P 32-Core Processor |
| Cores | 32 (1 thread per core) |
| NUMA Nodes | 2 (cores 0-15, 16-31) |
| L1d/L1i Cache | 32K / 32K |
| L2 Cache | 512K per core |
| L3 Cache | 32MB shared |
| CPU MHz | 1500-2800 MHz |
| SIMD | AVX2, SSE4.2 |

The NUMA architecture (2 nodes) means memory access patterns are critical - threads accessing memory from the "wrong" NUMA node incur latency penalties.

## Cluster Benchmark Results

### Linux Cluster (4-5 OpenMP threads)

**benchmark_snapshot.py results:**

| Stage | Serial | Parallel | Speedup |
|-------|--------|----------|---------|
| Sorting | 2.25s | 0.68s | **3.32x** |
| Morphology | 3:35.1 | 3:34.5 | 1.00x |
| Density | 31.9s | 32.0s | 1.00x |
| Grid deposition | 0.27s | 0.25s | 1.04x |
| **Total** | 4:10.1 | 4:08.2 | **1.01x** |

**benchmark_pipeline.py results (vs Python approach):**

| Metric | Python + C++ | Unified Pipeline | Improvement |
|--------|--------------|------------------|-------------|
| ID Detection | 31.45s | 13.7ms | **2,295x** |
| Sorting | 188.39s | 2.20s | **85.6x** |
| Morphology | 57.72s | 57.45s | 1.00x |
| **Total** | 277.56s | 59.76s | **4.64x** |
| Peak Memory | 2,112 MB | 416 MB | **80.3% savings** |

### macOS (Apple Silicon, libomp)

| Stage | Serial | Parallel | Speedup |
|-------|--------|----------|---------|
| Sorting | 2.27s | 0.19s | **12.21x** |
| Morphology | 1:18.2 | 1:30.7 | 0.86x (slower) |
| Density | 12.5s | 12.1s | 1.03x |
| **Total** | 1:33.3 | 1:43.3 | **0.90x** (slower) |

### Real-World Impact

**Density PDF computation script:**

| Metric | Before Optimizations | After Optimizations | Improvement |
|--------|---------------------|---------------------|-------------|
| Total time | ~7:30 | 5:49 | **22% faster** |

## Why Morphology Doesn't Scale

The ORIGAMI algorithm checks neighbors along 3 Cartesian axes and 6 diagonal directions for each particle. For a 256³ grid:

1. **Memory access pattern is scattered:**
   - X-neighbors: stride 1 (cache-friendly)
   - Y-neighbors: stride 256 × 12 bytes = 3KB (may hit L2)
   - Z-neighbors: stride 256² × 12 bytes = 768KB (always misses cache)

2. **Memory bandwidth is shared:**
   - All threads read from the same position array
   - More threads = more cache pressure, not more throughput
   - The memory bus becomes the bottleneck, not CPU cycles

3. **Work is inherently sequential per particle:**
   - Each particle must check up to ng4=64 neighbors in each direction
   - Early termination on first crossing detection
   - Cannot easily vectorize the conditional logic

4. **NUMA effects on multi-socket systems:**
   - The AMD EPYC has 2 NUMA nodes (cores 0-15, 16-31)
   - Threads accessing memory allocated on the other NUMA node incur extra latency
   - The position array (384 MB) may be split across NUMA nodes
   - This further limits scaling beyond ~8-16 threads

## Platform Differences

| Platform | OpenMP Runtime | Morphology Scaling | Notes |
|----------|---------------|-------------------|-------|
| Linux (cluster) | libgomp | 1.00x (neutral) | Real threading, good memory bandwidth |
| macOS (Apple Silicon) | libomp | 0.86x (slower) | Unified memory, different scheduler |

The macOS performance degradation is likely due to:
- Unified memory architecture (CPU/GPU share bandwidth)
- Different OpenMP runtime behavior (libomp vs libgomp)
- Apple Silicon memory subsystem characteristics

## Optimizations Applied

### January 2025 Updates

1. **Parallelized density grid reduction:**
   - Was: Serial O(threads × cells) reduction
   - Now: Parallel reduction over cells
   - Impact: Prevents density slowdown with multiple threads

2. **Dynamic scheduling for morphology:**
   - Added `schedule(dynamic, 1)` for subdomain load balancing
   - Impact: Handles cases where early-termination varies across subdomains

3. **Memory optimizations (earlier):**
   - Bit-packed crossing flags (4 arrays → 1 uint16_t)
   - Position caching and prefetching
   - Native type arithmetic (avoid float→double→float)

## Potential Future Optimizations

For significant parallel speedup in morphology, these larger changes would be needed:

| Optimization | Expected Impact | Complexity |
|--------------|-----------------|------------|
| SOA memory layout | 1.5-2x | High (API change) |
| SIMD neighbor checks (AVX2) | 2-4x for inner loop | Medium |
| Morton-order processing | Better cache locality | Medium |
| Reduced search range | Fewer memory accesses | Low (accuracy tradeoff) |

## Conclusions

1. **Memory-bound reality:** The ORIGAMI morphology algorithm is fundamentally limited by memory bandwidth, not CPU speed. Parallel scaling beyond 1-2 threads provides diminishing returns.

2. **Where parallelism helps:**
   - Sorting: 3-12x speedup depending on platform
   - Grid reduction: Prevents parallel slowdown
   - ID detection: Already fast enough to not matter

3. **Where it doesn't help:**
   - Morphology computation: Memory-bound, 1.00x scaling
   - Density Monte Carlo: Also memory-bound for grid access

4. **Practical guidance:**
   - Use 4-8 threads for best overall performance
   - 32 cores showed same results as 4-5 threads (memory-bound saturation)
   - More threads may actually slow down on some systems
   - The 4.64x pipeline speedup comes from eliminating Python overhead, not from parallel scaling

## Voronoi (VTFE) Density Parallel Scaling

**Date:** February 2025
**System:** macOS (Apple Silicon M-series, conda libomp)
**Grid Size:** N = 256³ = 16,777,216 particles

Unlike ORIGAMI morphology, the Voronoi density computation is **CPU-limited** (not memory-bandwidth limited) and scales well with threads. Each thread gets its own `voro_compute` object with private mask/queue buffers (~108 KB per thread).

### Thread-Safety Implementation

Three fixes enable safe parallel Voronoi computation:
1. **Pre-build periodic images**: `create_all_images()` before the parallel region makes `create_periodic_image()` a no-op during `compute_cell()`
2. **Per-thread `voro_compute` objects**: Each thread owns private `mask[]`/`qu[]` buffers instead of sharing the container's internal `voro_compute` member
3. **Block coordinates from `c_loop`**: Captured during sequential iteration, passed to per-thread `compute_cell()`

### Scaling Results

#### snapshot_034 (a=1, present day)

| Threads | Time | Speedup | Efficiency |
|---------|------|---------|------------|
| 1 | 705.6s | 1.0x | 100% |
| 2 | 365.2s | 1.9x | 97% |
| 4 | 194.4s | 3.6x | 91% |
| 8 | 121.7s | 5.8x | 72% |

#### snapshot_074 (a=100, far future)

| Threads | Time | Speedup | Efficiency |
|---------|------|---------|------------|
| 1 | 3200.7s | 1.0x | 100% |
| 2 | 1619.9s | 2.0x | 99% |
| 4 | 834.4s | 3.8x | 96% |
| 8 | 465.6s | 6.9x | 86% |

Volume conservation is exact across all thread counts (`vol_sum/box_vol = 1.00000000`).

### Why Voronoi Scales Better Than ORIGAMI

1. **CPU-bound work**: Each `compute_cell()` call does substantial geometric computation (plane cutting, vertex enumeration) — microseconds per cell, not memory lookups
2. **Independent cells**: Each particle's Voronoi cell is computed independently with no data dependencies between cells
3. **No shared writes**: Each thread writes to a distinct `volumes[loc.id]` element (particle IDs are unique)
4. **a=100 scales better**: Longer per-cell compute time (extreme density contrasts, elongated cells) means more useful work relative to scheduling overhead

### macOS OpenMP Fix

The previous macOS ARM64 `OMP_NUM_THREADS=1` workaround was a blanket disable caused by loading two different libomp copies (Homebrew + conda). The fix:
- `CMakeLists.txt` now prefers conda's libomp (`$CONDA_PREFIX/lib/libomp.dylib`) when building inside a conda environment
- `__init__.py` detects actual libomp conflicts at runtime and only restricts threading when necessary
- Result: **full multi-threading now works on macOS ARM64**

## Benchmark Commands

```bash
# Serial vs parallel comparison
python examples/benchmark_snapshot.py /path/to/snapshot.hdf5

# Pipeline vs Python comparison
python examples/benchmark_pipeline.py /path/to/snapshots/

# Optimization validation with baseline
python examples/benchmark_optimization.py /path/to/snapshot.hdf5 --save-baseline baseline.npz
python examples/benchmark_optimization.py /path/to/snapshot.hdf5 --compare baseline.npz
```
