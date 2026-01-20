# ORIGAMI Pipeline Benchmark Results

Benchmark comparing the unified C++ pipeline against the original Python-based approach.

**Date:** January 20, 2025
**Test System:** macOS (Apple Silicon)
**Grid Size:** N = 256³ = 16,777,216 particles
**Position Array Size:** 384 MB

## Summary

| Metric | Original | New Pipeline | Improvement |
|--------|----------|--------------|-------------|
| **Average Total Time** | 234.2 s | 83.3 s | **2.83x faster** |
| **Peak Memory** | 2,112 MB | 416 MB | **80.3% savings** |
| **Ordering Detection** | 19.4 s | 4 ms | **4,850x faster** |
| **Sorting** | 133.5 s | 2.2 s | **60x faster** |
| **Accuracy** | - | - | **100% exact match** |

## Detailed Results by Snapshot

### Snapshot 031 (z = 0.414, a = 0.707)

| Stage | Original | Pipeline | Speedup |
|-------|----------|----------|---------|
| Detection | 19.49 s | 4.0 ms | 4,872x |
| Sorting | 133.27 s | 2.18 s | 61x |
| Morphology | 89.78 s | 87.87 s | 1.02x |
| **Total** | **242.54 s** | **90.10 s** | **2.69x** |

**Fractions:** f_void=43.29%, f_wall=18.82%, f_filament=15.60%, f_halo=22.30%
**Accuracy:** EXACT MATCH on all 16,777,216 particle classifications

---

### Snapshot 032 (z = 0.258, a = 0.795)

| Stage | Original | Pipeline | Speedup |
|-------|----------|----------|---------|
| Detection | 19.44 s | 4.1 ms | 4,741x |
| Sorting | 132.83 s | 2.19 s | 61x |
| Morphology | 86.25 s | 88.84 s | 0.97x |
| **Total** | **238.52 s** | **91.08 s** | **2.62x** |

**Fractions:** f_void=39.27%, f_wall=18.83%, f_filament=16.41%, f_halo=25.49%
**Accuracy:** EXACT MATCH on all 16,777,216 particle classifications

---

### Snapshot 033 (z = 0.124, a = 0.890)

| Stage | Original | Pipeline | Speedup |
|-------|----------|----------|---------|
| Detection | 19.49 s | 3.2 ms | 6,091x |
| Sorting | 133.22 s | 2.21 s | 60x |
| Morphology | 82.39 s | 81.18 s | 1.01x |
| **Total** | **235.10 s** | **83.45 s** | **2.82x** |

**Fractions:** f_void=36.03%, f_wall=18.71%, f_filament=16.98%, f_halo=28.29%
**Accuracy:** EXACT MATCH on all 16,777,216 particle classifications

---

### Snapshot 034 (z = 0.000, a = 1.000)

| Stage | Original | Pipeline | Speedup |
|-------|----------|----------|---------|
| Detection | 19.21 s | 4.0 ms | 4,803x |
| Sorting | 132.83 s | 2.24 s | 59x |
| Morphology | 81.08 s | 79.06 s | 1.03x |
| **Total** | **233.13 s** | **81.36 s** | **2.87x** |

**Fractions:** f_void=33.21%, f_wall=18.49%, f_filament=17.42%, f_halo=30.88%
**Accuracy:** EXACT MATCH on all 16,777,216 particle classifications

---

### Snapshot 074 (z = -0.990, a = 100.0)

| Stage | Original | Pipeline | Speedup |
|-------|----------|----------|---------|
| Detection | 19.49 s | 4.0 ms | 4,872x |
| Sorting | 135.31 s | 2.20 s | 62x |
| Morphology | 66.99 s | 68.30 s | 0.98x |
| **Total** | **221.79 s** | **70.57 s** | **3.14x** |

**Fractions:** f_void=21.80%, f_wall=17.34%, f_filament=18.50%, f_halo=42.36%
**Accuracy:** EXACT MATCH on all 16,777,216 particle classifications

---

## Summary Table

| Snapshot | N | Original | Pipeline | Speedup | Mem Save | f_halo | Accurate |
|----------|---|----------|----------|---------|----------|--------|----------|
| snapshot_031 | 16,777,216 | 242.54 s | 90.10 s | 2.69x | 80.3% | 22.30% | YES |
| snapshot_032 | 16,777,216 | 238.52 s | 91.08 s | 2.62x | 80.3% | 25.49% | YES |
| snapshot_033 | 16,777,216 | 235.10 s | 83.45 s | 2.82x | 80.3% | 28.29% | YES |
| snapshot_034 | 16,777,216 | 233.13 s | 81.36 s | 2.87x | 80.3% | 30.88% | YES |
| snapshot_074 | 16,777,216 | 221.79 s | 70.57 s | 3.14x | 80.3% | 42.36% | YES |
| **Average** | - | **234.2 s** | **83.3 s** | **2.83x** | **80.3%** | - | **100%** |

## Memory Usage

| Approach | Peak Memory | Breakdown |
|----------|-------------|-----------|
| **Original (Python)** | 2,112 MB | ~384 MB positions + ~384 MB sorted copy + ~1.3 GB hash table overhead |
| **New Pipeline (C++)** | 416 MB | ~384 MB positions + ~32 MB permutation/flags |
| **Savings** | **1,696 MB** | **80.3%** |

## Key Performance Insights

### Why the Pipeline is Faster

1. **ID Ordering Detection (4,850x faster)**
   - Original: Python dict lookup with 16M entries, O(n) iteration
   - Pipeline: C++ unordered_map with only 100k samples, early exit

2. **Lagrangian Sorting (60x faster)**
   - Original: Python loop over 16M particles, dict lookups
   - Pipeline: OpenMP-parallelized C++ with direct array indexing

3. **Memory Efficiency (80% savings)**
   - Original: Creates full position copy + Python dict overhead
   - Pipeline: In-place cycle decomposition with minimal temporary storage

### Scaling Expectations for N=1024³

For N=1024³ (64x more particles):

| Metric | Expected Original | Expected Pipeline | Notes |
|--------|-------------------|-------------------|-------|
| Positions | ~24 GB | ~24 GB | Same |
| Sorting temp | ~24 GB copy | ~9 GB indices | No position copy |
| Detection | >1 hour | <1 second | Scales with sample size |
| Sorting | ~2+ hours | ~3 minutes | O(n) with good constants |
| **Total memory** | ~50+ GB | ~35 GB | Fits in 64GB RAM |

## Conclusions

1. **Accuracy**: The new pipeline produces **identical results** to the original approach for all tested snapshots.

2. **Speed**: Average **2.83x overall speedup**, with detection and sorting stages seeing **60-4850x improvements**.

3. **Memory**: **80% reduction** in peak memory usage, critical for larger simulations.

4. **Robustness**: Correct ID ordering detection for all snapshots, including the previously problematic snapshots 31-33.

The unified pipeline is ready for production use and will enable analysis of larger simulations (N=1024³) that would previously run out of memory.
