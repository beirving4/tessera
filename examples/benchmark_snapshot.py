#!/usr/bin/env python3
"""
Benchmark script to compare serial vs parallel ORIGAMI pipeline performance
on a real GADGET-4 snapshot.

Usage:
    python benchmark_snapshot.py /path/to/snapshot.hdf5
"""

import argparse
import os
import sys
import time
import tracemalloc

# Set environment variables before imports to avoid OpenMP conflicts on macOS
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# IMPORTANT: Import tessera BEFORE numpy on macOS to avoid OpenMP runtime conflicts
try:
    import tessera as ts
except ImportError:
    import _tessera as ts

import numpy as np


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024), peak / (1024 * 1024)


def run_benchmark(snapshot_path: str, n_threads: int, output_cells: int = 64,
                  n_samples: int = 10) -> dict:
    """Run the ORIGAMI pipeline and return timing/memory results."""

    # Start memory tracking
    tracemalloc.start()

    # Read snapshot
    print(f"  Reading snapshot...")
    start_read = time.perf_counter()
    snap = ts.io.read_gadget4_snapshot(snapshot_path)
    read_time = time.perf_counter() - start_read

    # Get dark matter particles (type 1 in GADGET)
    dm = snap.particles[1]
    positions = np.asarray(dm.coordinates)
    particle_ids = np.asarray(dm.particle_ids)
    box_size = snap.header.box_size
    n_particles = positions.shape[0]
    grid_size = round(n_particles ** (1/3))

    mem_after_read = get_memory_usage_mb()

    print(f"  Loaded {n_particles:,} particles (grid={grid_size}^3)")
    print(f"  Read time: {read_time:.2f}s")

    # Configure pipeline
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = output_cells
    config.density_n_samples = n_samples
    config.sample_density_at_particles = True
    config.grid_cells = output_cells  # Enable volume fractions
    config.n_threads = n_threads
    config.sort_in_place = (n_threads == 1)  # Use parallel sort for multi-threaded

    # Run pipeline
    print(f"  Running ORIGAMI pipeline (n_threads={n_threads})...")
    start_pipeline = time.perf_counter()
    result = ts.origami.run_pipeline(positions, particle_ids, config)
    pipeline_time = time.perf_counter() - start_pipeline

    mem_after_pipeline = get_memory_usage_mb()

    # Stop memory tracking
    tracemalloc.stop()

    return {
        'n_threads': n_threads,
        'n_particles': n_particles,
        'grid_size': grid_size,
        'read_time_s': read_time,
        'total_time_s': pipeline_time,
        'sorting_time_ms': result.sorting_time_ms,
        'morphology_time_ms': result.morphology_time_ms,
        'density_time_ms': result.density_time_ms,
        'grid_time_ms': result.grid_time_ms,
        'pdf_time_ms': getattr(result, 'pdf_time_ms', 0.0),
        'n_void': result.n_void,
        'n_wall': result.n_wall,
        'n_filament': result.n_filament,
        'n_halo': result.n_halo,
        'mem_after_read_mb': mem_after_read[0],
        'mem_peak_read_mb': mem_after_read[1],
        'mem_after_pipeline_mb': mem_after_pipeline[0],
        'mem_peak_pipeline_mb': mem_after_pipeline[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark ORIGAMI pipeline')
    parser.add_argument('snapshot', help='Path to GADGET-4 snapshot HDF5 file')
    parser.add_argument('--output-cells', type=int, default=64,
                        help='Output grid cells (default: 64)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of density samples (default: 10)')
    args = parser.parse_args()

    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot not found: {args.snapshot}")
        sys.exit(1)

    print("=" * 70)
    print("TESSERA ORIGAMI PIPELINE BENCHMARK")
    print("=" * 70)
    print()
    print(f"Snapshot: {args.snapshot}")
    print(f"Output cells: {args.output_cells}")
    print(f"Density samples: {args.n_samples}")
    print()

    # Run serial benchmark
    print("-" * 70)
    print("SERIAL BENCHMARK (n_threads=1, in-place sort)")
    print("-" * 70)
    serial_result = run_benchmark(
        args.snapshot, n_threads=1,
        output_cells=args.output_cells, n_samples=args.n_samples
    )
    print()

    # Run parallel benchmark
    print("-" * 70)
    print("PARALLEL BENCHMARK (n_threads=0 [auto], out-of-place sort)")
    print("-" * 70)
    parallel_result = run_benchmark(
        args.snapshot, n_threads=0,
        output_cells=args.output_cells, n_samples=args.n_samples
    )
    print()

    # Print comparison
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Stage':<25} {'Serial (ms)':<15} {'Parallel (ms)':<15} {'Speedup':<10}")
    print("-" * 65)

    stages = [
        ('Sorting', 'sorting_time_ms'),
        ('Morphology', 'morphology_time_ms'),
        ('Density', 'density_time_ms'),
        ('Grid deposition', 'grid_time_ms'),
        ('PDF computation', 'pdf_time_ms'),
    ]

    for name, key in stages:
        s = serial_result[key]
        p = parallel_result[key]
        if p > 0:
            speedup = s / p
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        print(f"{name:<25} {s:<15.2f} {p:<15.2f} {speedup_str:<10}")

    print("-" * 65)
    s_total = serial_result['total_time_s'] * 1000
    p_total = parallel_result['total_time_s'] * 1000
    speedup = s_total / p_total if p_total > 0 else float('inf')
    print(f"{'TOTAL PIPELINE':<25} {s_total:<15.2f} {p_total:<15.2f} {speedup:.2f}x")

    print()
    print("MEMORY USAGE")
    print("-" * 65)
    print(f"{'Metric':<35} {'Serial (MB)':<15} {'Parallel (MB)':<15}")
    print("-" * 65)
    print(f"{'After read':<35} {serial_result['mem_after_read_mb']:<15.1f} {parallel_result['mem_after_read_mb']:<15.1f}")
    print(f"{'Peak during read':<35} {serial_result['mem_peak_read_mb']:<15.1f} {parallel_result['mem_peak_read_mb']:<15.1f}")
    print(f"{'After pipeline':<35} {serial_result['mem_after_pipeline_mb']:<15.1f} {parallel_result['mem_after_pipeline_mb']:<15.1f}")
    print(f"{'Peak during pipeline':<35} {serial_result['mem_peak_pipeline_mb']:<15.1f} {parallel_result['mem_peak_pipeline_mb']:<15.1f}")

    print()
    print("MORPHOLOGY COUNTS (should be identical)")
    print("-" * 65)
    print(f"  Serial:   void={serial_result['n_void']:,}, wall={serial_result['n_wall']:,}, "
          f"filament={serial_result['n_filament']:,}, halo={serial_result['n_halo']:,}")
    print(f"  Parallel: void={parallel_result['n_void']:,}, wall={parallel_result['n_wall']:,}, "
          f"filament={parallel_result['n_filament']:,}, halo={parallel_result['n_halo']:,}")

    # Verify results match
    if (serial_result['n_void'] == parallel_result['n_void'] and
        serial_result['n_wall'] == parallel_result['n_wall'] and
        serial_result['n_filament'] == parallel_result['n_filament'] and
        serial_result['n_halo'] == parallel_result['n_halo']):
        print("\n  [PASS] Results match!")
    else:
        print("\n  [FAIL] Results differ!")

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
