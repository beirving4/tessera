#!/usr/bin/env python3
"""
Test script to verify parallel performance on macOS and other platforms.

This script benchmarks the ORIGAMI pipeline with different thread counts
to verify that parallelization is working correctly.

Usage:
    # Test with automatic thread detection
    python test_parallel_performance.py

    # Test with specific thread count
    python test_parallel_performance.py --threads 4

    # Test on macOS with OpenMP workaround
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 python test_parallel_performance.py
"""

import argparse
import os
import time

# Set environment variables before imports to avoid OpenMP conflicts
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# IMPORTANT: Import tessera BEFORE numpy on macOS to avoid OpenMP runtime conflicts
# When numpy is imported first, it initializes its OpenMP runtime, which can conflict
# with tessera's OpenMP when using multiple threads.
try:
    import tessera as ts
except ImportError:
    import _tessera as ts

import numpy as np


def generate_test_particles(grid_size: int, box_size: float = 100.0) -> tuple:
    """Generate test particle data with small random displacements."""
    n_particles = grid_size ** 3

    # Create Lagrangian grid positions
    x = np.linspace(0, box_size, grid_size, endpoint=False) + box_size / (2 * grid_size)
    positions = np.array(np.meshgrid(x, x, x, indexing='ij')).reshape(3, -1).T

    # Add small random displacements to simulate structure formation
    np.random.seed(42)
    displacements = np.random.normal(0, box_size / grid_size * 0.3, positions.shape)
    positions += displacements

    # Wrap to box
    positions = positions % box_size

    # Create particle IDs (x-major ordering)
    particle_ids = np.arange(n_particles, dtype=np.int64)

    return positions.astype(np.float64), particle_ids, box_size


def benchmark_pipeline(positions: np.ndarray, particle_ids: np.ndarray,
                       box_size: float, grid_size: int, n_threads: int,
                       use_parallel_sort: bool = True) -> dict:
    """Run the ORIGAMI pipeline and return timing results."""
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.n_threads = n_threads
    config.sort_in_place = not use_parallel_sort  # out-of-place = parallel
    config.id_offset = 0  # Our test IDs start at 0
    config.density_output_cells = 64
    config.density_n_samples = 10
    config.sample_density_at_particles = True
    config.grid_cells = 64
    # Note: PDF computation has a known issue on macOS, skip for now
    # config.pdf_n_bins = 50

    # Make a copy for the pipeline (it may modify positions)
    pos_copy = positions.copy()

    start = time.perf_counter()
    result = ts.origami.run_pipeline(pos_copy, particle_ids, config)
    total_time = time.perf_counter() - start

    return {
        'total_time_s': total_time,
        'sorting_time_ms': result.sorting_time_ms,
        'morphology_time_ms': result.morphology_time_ms,
        'density_time_ms': result.density_time_ms,
        'grid_time_ms': result.grid_time_ms,
        'pdf_time_ms': getattr(result, 'pdf_time_ms', 0.0),
        'n_void': result.n_void,
        'n_wall': result.n_wall,
        'n_filament': result.n_filament,
        'n_halo': result.n_halo,
    }


def main():
    parser = argparse.ArgumentParser(description='Test parallel performance')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Particles per dimension (default: 64)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads (default: auto)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare serial vs parallel performance')
    args = parser.parse_args()

    print("=" * 60)
    print("TESSERA PARALLEL PERFORMANCE TEST")
    print("=" * 60)
    print()

    # Check OpenMP status
    try:
        import os
        omp_threads = os.environ.get('OMP_NUM_THREADS', 'not set')
        print(f"OMP_NUM_THREADS: {omp_threads}")
        print(f"KMP_DUPLICATE_LIB_OK: {os.environ.get('KMP_DUPLICATE_LIB_OK', 'not set')}")
    except Exception:
        pass

    print(f"Grid size: {args.grid_size}^3 = {args.grid_size**3:,} particles")
    print()

    # Generate test data
    print("Generating test particles...")
    positions, particle_ids, box_size = generate_test_particles(args.grid_size)
    print(f"  Positions shape: {positions.shape}")
    print()

    if args.compare:
        # Compare serial vs parallel
        print("Benchmarking serial (n_threads=1)...")
        serial_result = benchmark_pipeline(
            positions, particle_ids, box_size, args.grid_size,
            n_threads=1, use_parallel_sort=False
        )

        print("Benchmarking parallel (n_threads=0, out-of-place sort)...")
        parallel_result = benchmark_pipeline(
            positions, particle_ids, box_size, args.grid_size,
            n_threads=0, use_parallel_sort=True
        )

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
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
            speedup = s / p if p > 0 else float('inf')
            print(f"{name:<25} {s:<15.2f} {p:<15.2f} {speedup:<10.2f}x")

        print("-" * 65)
        s_total = serial_result['total_time_s'] * 1000
        p_total = parallel_result['total_time_s'] * 1000
        speedup = s_total / p_total if p_total > 0 else float('inf')
        print(f"{'TOTAL':<25} {s_total:<15.2f} {p_total:<15.2f} {speedup:<10.2f}x")

        print()
        print("Morphology counts (should be identical):")
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

    else:
        # Single benchmark run
        n_threads = args.threads if args.threads is not None else 0
        print(f"Benchmarking with n_threads={n_threads} (0=auto)...")

        result = benchmark_pipeline(
            positions, particle_ids, box_size, args.grid_size,
            n_threads=n_threads, use_parallel_sort=True
        )

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print()
        print(f"Total time: {result['total_time_s']*1000:.2f} ms")
        print()
        print("Stage breakdown:")
        print(f"  Sorting:        {result['sorting_time_ms']:.2f} ms")
        print(f"  Morphology:     {result['morphology_time_ms']:.2f} ms")
        print(f"  Density:        {result['density_time_ms']:.2f} ms")
        print(f"  Grid deposit:   {result['grid_time_ms']:.2f} ms")
        print(f"  PDF:            {result['pdf_time_ms']:.2f} ms")
        print()
        print("Morphology counts:")
        print(f"  Void:     {result['n_void']:,}")
        print(f"  Wall:     {result['n_wall']:,}")
        print(f"  Filament: {result['n_filament']:,}")
        print(f"  Halo:     {result['n_halo']:,}")

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
