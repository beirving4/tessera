#!/usr/bin/env python3
"""
Benchmark script for validating ORIGAMI optimizations.
Saves morphology results for comparison and tracks runtime/memory.

Usage:
    python benchmark_optimization.py /path/to/snapshot.hdf5 --save-baseline
    python benchmark_optimization.py /path/to/snapshot.hdf5 --compare baseline.npz
"""

import argparse
import os
import resource
import sys
import time

# Set environment variables before imports
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Import tessera BEFORE numpy on macOS
try:
    import tessera as ts
except ImportError:
    import _tessera as ts

import numpy as np
from datetime import timedelta


def format_time(elapsed_seconds: float) -> str:
    """Format time in human-readable form using timedelta for longer durations."""
    if elapsed_seconds < 1.0:
        return f"{elapsed_seconds:.6f} seconds"
    td = timedelta(seconds=elapsed_seconds)
    return str(td)


def format_time_ms(ms: float) -> str:
    """Format time given in milliseconds."""
    return format_time(ms / 1000.0)


def run_benchmark(positions, particle_ids, grid_size, box_size, n_threads=1):
    """Run ORIGAMI pipeline and return results with timing."""

    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = 64
    config.density_n_samples = 10
    config.sample_density_at_particles = True
    config.grid_cells = 64
    config.n_threads = n_threads
    config.sort_in_place = True  # Use in-place for consistent comparison

    # Make a copy to avoid modifying original
    pos_copy = positions.copy()
    ids_copy = particle_ids.copy()

    # Run pipeline
    start = time.perf_counter()
    result = ts.origami.run_pipeline(pos_copy, ids_copy, config)
    elapsed = time.perf_counter() - start

    # Get memory
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    return {
        'morphology': np.array(result.morphology),
        'n_void': result.n_void,
        'n_wall': result.n_wall,
        'n_filament': result.n_filament,
        'n_halo': result.n_halo,
        'total_time_ms': elapsed * 1000,
        'sorting_time_ms': result.sorting_time_ms,
        'morphology_time_ms': result.morphology_time_ms,
        'density_time_ms': result.density_time_ms,
        'grid_time_ms': result.grid_time_ms,
        'peak_memory_mb': mem_mb,
    }


def validate_results(current, baseline):
    """Compare current results against baseline."""

    # Check morphology array
    morph_match = np.array_equal(current['morphology'], baseline['morphology'])

    # Check counts
    counts_match = (
        current['n_void'] == baseline['n_void'] and
        current['n_wall'] == baseline['n_wall'] and
        current['n_filament'] == baseline['n_filament'] and
        current['n_halo'] == baseline['n_halo']
    )

    if not morph_match:
        diff_count = np.sum(current['morphology'] != baseline['morphology'])
        diff_pct = diff_count / len(current['morphology']) * 100
        print(f"  [FAIL] Morphology mismatch: {diff_count:,} particles ({diff_pct:.4f}%)")

    if not counts_match:
        print(f"  [FAIL] Count mismatch:")
        print(f"    Baseline: void={baseline['n_void']:,}, wall={baseline['n_wall']:,}, "
              f"filament={baseline['n_filament']:,}, halo={baseline['n_halo']:,}")
        print(f"    Current:  void={current['n_void']:,}, wall={current['n_wall']:,}, "
              f"filament={current['n_filament']:,}, halo={current['n_halo']:,}")

    return morph_match and counts_match


def print_results(results, label=""):
    """Print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS{': ' + label if label else ''}")
    print(f"{'=' * 60}")
    print(f"Total time:      {format_time_ms(results['total_time_ms'])}")
    print(f"  Sorting:       {format_time_ms(results['sorting_time_ms'])}")
    print(f"  Morphology:    {format_time_ms(results['morphology_time_ms'])}")
    print(f"  Density:       {format_time_ms(results['density_time_ms'])}")
    print(f"  Grid:          {format_time_ms(results['grid_time_ms'])}")
    print(f"Peak memory:     {results['peak_memory_mb']:,.1f} MB")
    print(f"\nMorphology counts:")
    print(f"  Void:     {results['n_void']:,}")
    print(f"  Wall:     {results['n_wall']:,}")
    print(f"  Filament: {results['n_filament']:,}")
    print(f"  Halo:     {results['n_halo']:,}")


def print_comparison(current, baseline):
    """Print comparison between current and baseline."""
    print(f"\n{'=' * 75}")
    print("COMPARISON (Current vs Baseline)")
    print(f"{'=' * 75}")

    def speedup(curr, base):
        if curr > 0:
            return base / curr
        return float('inf')

    print(f"{'Metric':<15} {'Baseline':<22} {'Current':<22} {'Change':<15}")
    print("-" * 75)

    # Time comparisons
    metrics = [
        ('Total time', 'total_time_ms'),
        ('Morphology', 'morphology_time_ms'),
        ('Sorting', 'sorting_time_ms'),
        ('Density', 'density_time_ms'),
    ]

    for name, key in metrics:
        base_val = baseline[key]
        curr_val = current[key]
        sp = speedup(curr_val, base_val)
        change = f"{sp:.2f}x {'faster' if sp > 1 else 'slower'}"
        print(f"{name:<15} {format_time_ms(base_val):<22} {format_time_ms(curr_val):<22} {change:<15}")

    # Memory comparison
    base_mem = baseline['peak_memory_mb']
    curr_mem = current['peak_memory_mb']
    mem_diff = curr_mem - base_mem
    mem_pct = (mem_diff / base_mem) * 100 if base_mem > 0 else 0
    base_mem_str = f"{base_mem:,.1f} MB"
    curr_mem_str = f"{curr_mem:,.1f} MB"
    print(f"{'Peak memory':<15} {base_mem_str:<22} {curr_mem_str:<22} {mem_diff:+.1f} MB ({mem_pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ORIGAMI optimizations')
    parser.add_argument('snapshot', help='Path to GADGET-4 snapshot')
    parser.add_argument('--save-baseline', type=str, metavar='FILE',
                        help='Save results as baseline to FILE')
    parser.add_argument('--compare', type=str, metavar='FILE',
                        help='Compare against baseline FILE')
    parser.add_argument('--n-threads', type=int, default=1,
                        help='Number of threads (default: 1)')
    args = parser.parse_args()

    print("=" * 60)
    print("ORIGAMI OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print(f"Snapshot: {args.snapshot}")
    print(f"Threads: {args.n_threads}")

    # Load snapshot
    print("\nLoading snapshot...")
    snap = ts.io.read_gadget4_snapshot(args.snapshot)
    dm = snap.particles[1]
    positions = np.asarray(dm.coordinates).copy()
    particle_ids = np.asarray(dm.particle_ids).copy()
    box_size = snap.header.box_size
    n_particles = positions.shape[0]
    grid_size = round(n_particles ** (1/3))

    print(f"Particles: {n_particles:,} ({grid_size}^3)")
    print(f"Box size: {box_size}")

    del snap, dm  # Free memory

    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(positions, particle_ids, grid_size, box_size, args.n_threads)

    print_results(results)

    # Save baseline if requested
    if args.save_baseline:
        print(f"\nSaving baseline to {args.save_baseline}...")
        np.savez_compressed(
            args.save_baseline,
            morphology=results['morphology'],
            n_void=results['n_void'],
            n_wall=results['n_wall'],
            n_filament=results['n_filament'],
            n_halo=results['n_halo'],
            total_time_ms=results['total_time_ms'],
            sorting_time_ms=results['sorting_time_ms'],
            morphology_time_ms=results['morphology_time_ms'],
            density_time_ms=results['density_time_ms'],
            grid_time_ms=results['grid_time_ms'],
            peak_memory_mb=results['peak_memory_mb'],
        )
        print("Baseline saved.")

    # Compare if requested
    if args.compare:
        print(f"\nLoading baseline from {args.compare}...")
        baseline_data = np.load(args.compare)
        baseline = {
            'morphology': baseline_data['morphology'],
            'n_void': int(baseline_data['n_void']),
            'n_wall': int(baseline_data['n_wall']),
            'n_filament': int(baseline_data['n_filament']),
            'n_halo': int(baseline_data['n_halo']),
            'total_time_ms': float(baseline_data['total_time_ms']),
            'sorting_time_ms': float(baseline_data['sorting_time_ms']),
            'morphology_time_ms': float(baseline_data['morphology_time_ms']),
            'density_time_ms': float(baseline_data['density_time_ms']),
            'grid_time_ms': float(baseline_data['grid_time_ms']),
            'peak_memory_mb': float(baseline_data['peak_memory_mb']),
        }

        print("\nValidating results...")
        if validate_results(results, baseline):
            print("  [PASS] Results match baseline!")

        print_comparison(results, baseline)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
