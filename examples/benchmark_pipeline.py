#!/usr/bin/env python3
"""
Benchmark: Unified ORIGAMI Pipeline vs Original Approach

Compares accuracy, speed, and memory usage between:
1. Original approach: Python helpers + compute_morphology
2. New unified pipeline: run_pipeline / run_pipeline_safe

Usage:
    python benchmark_pipeline.py
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import h5py

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add parent directory to path to import utils
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Import shared utilities
from utils import load_snapshot_h5py, infer_grid_size

try:
    from config import SNAPSHOT_BASE
except ImportError:
    raise ImportError("Please create examples/config.py with your local paths.")

# Import tessera
try:
    import tessera as ts
except ImportError:
    REPO_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(REPO_ROOT / 'python'))
    import tessera as ts

# Snapshots to benchmark
SNAPSHOTS = ['snapshot_031', 'snapshot_032', 'snapshot_033', 'snapshot_034', 'snapshot_074']


def load_snapshot(snapshot_name):
    """Load particle data from GADGET-4 snapshot."""
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"

    if not snapshot_path.exists():
        print(f"  Snapshot not found: {snapshot_path}")
        return None

    # Use utils function for loading
    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )

    # Read redshift separately (not returned by load_snapshot_h5py)
    with h5py.File(snapshot_path, 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])

    # Infer grid size using utils function
    grid_size = infer_grid_size(len(positions))

    return {
        'positions': positions,
        'particle_ids': particle_ids,
        'box_size': box_size,
        'redshift': redshift,
        'scale_factor': scale_factor,
        'grid_size': grid_size,
        'n_particles': len(positions)
    }


def detect_id_ordering_python(positions, particle_ids, box_size, grid_size):
    """Original Python implementation for ID ordering detection."""
    grid_size = int(grid_size)
    particle_ids_int = particle_ids.astype(np.int64)
    id_to_idx = {int(pid): i for i, pid in enumerate(particle_ids_int)}

    def get_pos(pid):
        if pid in id_to_idx:
            return positions[id_to_idx[pid]]
        return None

    def periodic_diff(a, b, box):
        diff = a - b
        diff = np.where(diff > box/2, diff - box, diff)
        diff = np.where(diff < -box/2, diff + box, diff)
        return diff

    spacing = box_size / grid_size
    test_ids = [1, grid_size//4, grid_size//2, grid_size*10, grid_size*100]

    x_major_score = 0.0
    z_major_score = 0.0
    n_valid = 0

    for base_id in test_ids:
        if base_id < 1 or base_id >= len(positions):
            continue

        p1 = get_pos(base_id)
        p2 = get_pos(base_id + 1)
        if p1 is None or p2 is None:
            continue

        diff = periodic_diff(p2, p1, box_size)
        z_expected = np.array([0.0, 0.0, spacing])
        x_expected = np.array([spacing, 0.0, 0.0])

        z_err = np.linalg.norm(diff - z_expected)
        x_err = np.linalg.norm(diff - x_expected)

        z_major_score += z_err
        x_major_score += x_err
        n_valid += 1

    for base_id in test_ids:
        next_y_id = base_id + grid_size
        if next_y_id >= len(positions):
            continue

        p1 = get_pos(base_id)
        p2 = get_pos(next_y_id)
        if p1 is None or p2 is None:
            continue

        diff = periodic_diff(p2, p1, box_size)
        z_major_score += abs(diff[0])
        x_major_score += abs(diff[2])

    return 'x-major' if x_major_score < z_major_score else 'z-major'


def sort_to_lagrangian_python(positions, particle_ids, grid_size, ordering):
    """Original Python implementation for Lagrangian sorting."""
    grid_size = int(grid_size)
    n_particles = len(positions)
    sorted_positions = np.empty_like(positions)

    for i in range(n_particles):
        pid = particle_ids[i] - 1  # Convert to 0-indexed

        if ordering == 'z-major':
            z = pid % grid_size
            y = (pid // grid_size) % grid_size
            x = pid // (grid_size * grid_size)
            target_idx = x + y * grid_size + z * grid_size * grid_size
        else:
            target_idx = pid

        sorted_positions[target_idx] = positions[i]

    return sorted_positions


def run_original_approach(data):
    """Run the original Python-based approach."""
    positions = data['positions'].copy()  # Don't modify original
    particle_ids = data['particle_ids']
    box_size = data['box_size']
    grid_size = data['grid_size']

    # Start timing
    start_time = time.perf_counter()

    # Step 1: Detect ordering (Python)
    t0 = time.perf_counter()
    ordering = detect_id_ordering_python(positions, particle_ids, box_size, grid_size)
    detection_time = time.perf_counter() - t0

    # Step 2: Sort to Lagrangian order (Python)
    t0 = time.perf_counter()
    sorted_positions = sort_to_lagrangian_python(positions, particle_ids, grid_size, ordering)
    sorting_time = time.perf_counter() - t0

    # Step 3: Compute morphology (C++)
    t0 = time.perf_counter()
    config = ts.origami.OrigamiConfig(grid_size, box_size)
    result = ts.origami.compute_morphology(sorted_positions, config)
    morphology_time = time.perf_counter() - t0

    total_time = time.perf_counter() - start_time

    return {
        'ordering': ordering,
        'morphology': np.array(result.morphology),
        'f_void': result.f_void,
        'f_wall': result.f_wall,
        'f_filament': result.f_filament,
        'f_halo': result.f_halo,
        'is_linear_regime': result.is_linear_regime,
        'detection_time_ms': detection_time * 1000,
        'sorting_time_ms': sorting_time * 1000,
        'morphology_time_ms': morphology_time * 1000,
        'total_time_ms': total_time * 1000,
    }


def run_new_pipeline(data, in_place=True):
    """Run the new unified pipeline."""
    positions = data['positions'].copy() if in_place else data['positions']
    particle_ids = data['particle_ids']
    box_size = data['box_size']
    grid_size = data['grid_size']

    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.sort_in_place = in_place

    if in_place:
        result = ts.origami.run_pipeline(positions, particle_ids, config)
    else:
        result = ts.origami.run_pipeline_safe(positions, particle_ids, config)

    ordering_name = ts.origami.id_ordering_name(result.detected_ordering)

    return {
        'ordering': ordering_name,
        'morphology': np.array(result.morphology),
        'f_void': result.f_void,
        'f_wall': result.f_wall,
        'f_filament': result.f_filament,
        'f_halo': result.f_halo,
        'is_linear_regime': result.is_linear_regime,
        'detection_time_ms': result.detection_time_ms,
        'sorting_time_ms': result.sorting_time_ms,
        'morphology_time_ms': result.morphology_time_ms,
        'total_time_ms': result.total_time_ms,
    }


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)  # Convert to MB


def compare_results(original, pipeline, name):
    """Compare results between original and pipeline approaches."""
    print(f"\n  Comparison for {name}:")

    # Check ordering detection
    orig_ordering = original['ordering']
    pipe_ordering = pipeline['ordering']
    ordering_match = orig_ordering == pipe_ordering
    print(f"    Ordering: original={orig_ordering}, pipeline={pipe_ordering} {'OK' if ordering_match else 'MISMATCH!'}")

    # Check fractions
    for cls in ['void', 'wall', 'filament', 'halo']:
        orig_f = original[f'f_{cls}']
        pipe_f = pipeline[f'f_{cls}']
        diff = abs(orig_f - pipe_f)
        status = 'OK' if diff < 1e-6 else f'DIFF={diff:.2e}'
        print(f"    f_{cls}: original={orig_f:.6f}, pipeline={pipe_f:.6f} {status}")

    # Check morphology array
    morph_match = np.array_equal(original['morphology'], pipeline['morphology'])
    if morph_match:
        print(f"    Morphology array: EXACT MATCH")
    else:
        n_diff = np.sum(original['morphology'] != pipeline['morphology'])
        pct_diff = 100 * n_diff / len(original['morphology'])
        print(f"    Morphology array: {n_diff} differences ({pct_diff:.4f}%)")

    return ordering_match and morph_match


def format_time(ms):
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms*1000:.1f} us"
    elif ms < 1000:
        return f"{ms:.1f} ms"
    else:
        return f"{ms/1000:.2f} s"


def main():
    print("=" * 70)
    print("ORIGAMI Pipeline Benchmark")
    print("=" * 70)
    print(f"\nComparing:")
    print("  - Original: Python helpers + C++ compute_morphology")
    print("  - Pipeline: Unified C++ run_pipeline")
    print()

    results_table = []

    for snapshot_name in SNAPSHOTS:
        print(f"\n{'='*70}")
        print(f"Snapshot: {snapshot_name}")
        print('='*70)

        # Load data
        print(f"Loading snapshot...")
        data = load_snapshot(snapshot_name)
        if data is None:
            continue

        print(f"  Grid: {data['grid_size']}^3 = {data['n_particles']:,} particles")
        print(f"  z = {data['redshift']:.4f}, a = {data['scale_factor']:.4f}")
        mem_positions = data['positions'].nbytes / (1024**2)
        print(f"  Position array: {mem_positions:.1f} MB")

        # Run original approach with memory tracking
        print(f"\nRunning original approach (Python helpers)...")
        original_result, original_mem = measure_memory(run_original_approach, data)
        print(f"  Ordering detected: {original_result['ordering']}")
        print(f"  Detection time: {format_time(original_result['detection_time_ms'])}")
        print(f"  Sorting time: {format_time(original_result['sorting_time_ms'])}")
        print(f"  Morphology time: {format_time(original_result['morphology_time_ms'])}")
        print(f"  Total time: {format_time(original_result['total_time_ms'])}")
        print(f"  Peak memory: {original_mem:.1f} MB")

        # Run new pipeline with memory tracking
        print(f"\nRunning new pipeline (unified C++)...")
        pipeline_result, pipeline_mem = measure_memory(run_new_pipeline, data, in_place=True)
        print(f"  Ordering detected: {pipeline_result['ordering']}")
        print(f"  Detection time: {format_time(pipeline_result['detection_time_ms'])}")
        print(f"  Sorting time: {format_time(pipeline_result['sorting_time_ms'])}")
        print(f"  Morphology time: {format_time(pipeline_result['morphology_time_ms'])}")
        print(f"  Total time: {format_time(pipeline_result['total_time_ms'])}")
        print(f"  Peak memory: {pipeline_mem:.1f} MB")

        # Compare results
        is_accurate = compare_results(original_result, pipeline_result, snapshot_name)

        # Calculate speedup
        speedup = original_result['total_time_ms'] / pipeline_result['total_time_ms']
        mem_savings = original_mem - pipeline_mem

        print(f"\n  Summary:")
        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Memory savings: {mem_savings:.1f} MB ({100*mem_savings/original_mem:.1f}%)")
        print(f"    Accurate: {'YES' if is_accurate else 'NO'}")

        results_table.append({
            'snapshot': snapshot_name,
            'n_particles': data['n_particles'],
            'original_time': original_result['total_time_ms'],
            'pipeline_time': pipeline_result['total_time_ms'],
            'speedup': speedup,
            'original_mem': original_mem,
            'pipeline_mem': pipeline_mem,
            'mem_savings_pct': 100 * mem_savings / original_mem if original_mem > 0 else 0,
            'accurate': is_accurate,
            'f_halo': pipeline_result['f_halo'],
        })

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Snapshot':<15} {'N':>12} {'Original':>12} {'Pipeline':>12} {'Speedup':>8} {'Mem Save':>10} {'f_halo':>8} {'OK':>5}")
    print("-" * 70)
    for r in results_table:
        print(f"{r['snapshot']:<15} {r['n_particles']:>12,} {format_time(r['original_time']):>12} "
              f"{format_time(r['pipeline_time']):>12} {r['speedup']:>7.2f}x {r['mem_savings_pct']:>9.1f}% "
              f"{r['f_halo']:>7.2%} {'YES' if r['accurate'] else 'NO':>5}")
    print("-" * 70)

    # Overall statistics
    if results_table:
        avg_speedup = np.mean([r['speedup'] for r in results_table])
        avg_mem_save = np.mean([r['mem_savings_pct'] for r in results_table])
        all_accurate = all(r['accurate'] for r in results_table)
        print(f"\nOverall: {avg_speedup:.2f}x average speedup, {avg_mem_save:.1f}% average memory savings")
        print(f"All results accurate: {'YES' if all_accurate else 'NO'}")


if __name__ == '__main__':
    main()
