#!/usr/bin/env python3
"""
ORIGAMI Validation: Compare tessera against original ORIGAMI code.

This script validates the tessera ORIGAMI implementation by comparing
its output against the original ORIGAMI code from Falck, Neyrinck & Szalay (2012).

Results are saved to this directory for reproducibility and documentation.
Memory and runtime benchmarks are included for comparison.
"""

import numpy as np
import struct
import subprocess
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil
import time
import tracemalloc

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Avoid OpenMP threading issues on macOS

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add build directory and tests directory to path
sys.path.insert(0, str(REPO_ROOT / 'build'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))
sys.path.insert(0, str(REPO_ROOT))

# Import local configuration
try:
    from config import SNAPSHOT_BASE, ORIGAMI_BIN
except ImportError:
    raise ImportError(
        "Please create tests/config.py with your local paths.\n"
        "Copy config.example.py to config.py and update the paths."
    )

# Import shared utilities
from utils import (
    load_snapshot,
    infer_grid_size,
    check_tessera_available,
)

# Import visualization utilities
from visualize import (
    create_origami_validation_figure,
    plot_morphology_slices_3panel,
)

# Import tessera
try:
    import tessera as ts
except ImportError:
    import _tessera as ts

# Snapshots to validate
SNAPSHOTS = [
    ("snapshot_034.hdf5", "z=0 (a=1)", "Present day"),
    ("snapshot_074.hdf5", "z=-0.99 (a=100)", "Far future"),
]


def convert_to_xmajor(positions, particle_ids, grid_size):
    """Convert GADGET z-major ordering to ORIGAMI x-major ordering.

    Note: This is kept as a Python loop because it's specific to the
    original ORIGAMI code's expected input format (x-major ordering).
    tessera uses z-major ordering internally.
    """
    n_particles = len(positions)
    sorted_positions = np.zeros((n_particles, 3), dtype=np.float64)

    for i in range(n_particles):
        idx = particle_ids[i] - 1  # 0-indexed z-major
        z = idx % grid_size
        y = (idx // grid_size) % grid_size
        x = idx // (grid_size * grid_size)
        xmajor_idx = x + y * grid_size + z * grid_size * grid_size
        sorted_positions[xmajor_idx] = positions[i]

    return np.ascontiguousarray(sorted_positions)


def write_pos_file(positions, filepath):
    """Write positions to binary format for original ORIGAMI."""
    n = len(positions)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('i', n))
        f.write(positions[:, 0].astype(np.float32).tobytes())
        f.write(positions[:, 1].astype(np.float32).tobytes())
        f.write(positions[:, 2].astype(np.float32).tobytes())


def read_tag_file(filepath):
    """Read morphology tags from original ORIGAMI output."""
    with open(filepath, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        tags = np.frombuffer(f.read(n), dtype=np.uint8)
    return tags


def run_original_origami(sorted_positions, box_size, grid_size):
    """Run original ORIGAMI code with timing."""
    # Use /tmp for intermediate files (original code has issues with long paths)
    work_dir = Path(tempfile.mkdtemp(prefix='origami_'))

    pos_file = work_dir / "positions.pos"
    params_file = work_dir / "params.txt"
    tag_label = "result"

    # Write position file
    write_pos_file(sorted_positions, pos_file)

    # Write parameter file
    with open(params_file, 'w') as f:
        f.write(f"# ORIGAMI validation\n")
        f.write(f"posfile\t\t{pos_file}\n")
        f.write(f"outdir\t\t{work_dir}/\n")
        f.write(f"taglabel\t{tag_label}\n")
        f.write(f"boxsize\t\t{box_size}\n")
        f.write(f"np1d\t\t{grid_size}\n")
        f.write(f"nsplit\t\t1\n")
        f.write(f"numfiles\t0\n")
        f.write(f"\n")
        f.write(f"buffer\t\t0.1\n")
        f.write(f"ndiv\t\t2\n")
        f.write(f"\n")
        f.write(f"volcut\t\t.01\n")
        f.write(f"npmin\t\t20\n")
        f.write(f"halolabel\ttest\n")

    # Run original ORIGAMI with timing
    t_start = time.perf_counter()

    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    result = subprocess.run(
        [str(ORIGAMI_BIN), str(params_file)],
        capture_output=True,
        text=True,
        env=env
    )

    t_elapsed = time.perf_counter() - t_start

    if result.returncode not in [0, 1]:
        shutil.rmtree(work_dir)
        raise RuntimeError(f"Original ORIGAMI failed (code {result.returncode}): {result.stderr}\n{result.stdout}")

    # Read results
    tag_file = work_dir / f"{tag_label}tag.dat"
    if not tag_file.exists():
        shutil.rmtree(work_dir)
        raise RuntimeError(f"Original ORIGAMI did not create output file. stdout: {result.stdout}")
    tags = read_tag_file(tag_file)

    # Clean up temp directory
    shutil.rmtree(work_dir)

    return tags, t_elapsed


def run_tessera_origami(sorted_positions, box_size, grid_size):
    """Run tessera ORIGAMI implementation with timing and memory tracking."""
    tracemalloc.start()
    t_start = time.perf_counter()

    config = ts.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    result = ts.origami.compute_morphology(sorted_positions, config)

    # Deposit to grid to compute volume fractions
    ts.origami.deposit_morphology_to_grid(sorted_positions, box_size, 128, result)

    t_elapsed = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.array(result.morphology), result, t_elapsed, peak_mem


def compare_results(orig_tags, tessera_tags, tessera_result):
    """Compare morphology classifications."""
    n = len(orig_tags)
    matches = np.sum(orig_tags == tessera_tags)
    match_rate = matches / n

    class_names = ['Void', 'Wall', 'Filament', 'Halo']
    stats = {
        'total_particles': int(n),
        'matches': int(matches),
        'match_rate': float(match_rate),
        'perfect_match': bool(match_rate == 1.0),
        'mass_fractions': {
            'void': float(tessera_result.f_void),
            'wall': float(tessera_result.f_wall),
            'filament': float(tessera_result.f_filament),
            'halo': float(tessera_result.f_halo),
        },
        'volume_fractions': {
            'void': float(tessera_result.v_void),
            'wall': float(tessera_result.v_wall),
            'filament': float(tessera_result.v_filament),
            'halo': float(tessera_result.v_halo),
        },
        'per_class': {}
    }

    for cls_idx, cls_name in enumerate(class_names):
        orig_count = int(np.sum(orig_tags == cls_idx))
        tessera_count = int(np.sum(tessera_tags == cls_idx))
        agreement = int(np.sum((orig_tags == cls_idx) & (tessera_tags == cls_idx)))
        stats['per_class'][cls_name.lower()] = {
            'original_count': orig_count,
            'tessera_count': tessera_count,
            'agreement': agreement,
            'original_fraction': orig_count / n,
            'tessera_fraction': tessera_count / n,
        }

    return stats


def save_results(output_dir, label, orig_tags, tessera_tags, stats, benchmarks,
                 box_size, redshift, scale_factor, description):
    """Save comparison results to JSON (no HDF5 to keep repo size small)."""
    # Save JSON summary only
    json_path = output_dir / f"{label}_validation.json"
    with open(json_path, 'w') as f:
        json.dump({
            'snapshot': label,
            'description': description,
            'box_size': box_size,
            'redshift': redshift,
            'scale_factor': scale_factor,
            'created': datetime.now().isoformat(),
            'benchmarks': benchmarks,
            **stats
        }, f, indent=2)

    return None, json_path


def main():
    print("=" * 70)
    print("ORIGAMI Validation: tessera vs Original Code")
    print("=" * 70)
    print(f"Output directory: {SCRIPT_DIR}")
    print()

    if not ORIGAMI_BIN.exists():
        print(f"ERROR: Original ORIGAMI binary not found at {ORIGAMI_BIN}")
        print("Please build it first.")
        return 1

    all_results = {}

    for snap_name, time_label, description in SNAPSHOTS:
        snap_path = SNAPSHOT_BASE / snap_name
        label = snap_name.replace('.hdf5', '')

        print(f"\n{'='*70}")
        print(f"Validating: {snap_name} ({time_label})")
        print("=" * 70)

        if not snap_path.exists():
            print(f"  WARNING: Snapshot not found: {snap_path}")
            continue

        # Load snapshot using shared utilities
        print("Loading snapshot...")
        t_load_start = time.perf_counter()
        tracemalloc.start()

        positions, particle_ids, snap_header = load_snapshot(
            snap_path,
            particle_type=1,
            read_ids=True
        )

        current_mem, peak_mem_load = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t_load = time.perf_counter() - t_load_start

        n_particles = len(positions)
        grid_size = infer_grid_size(n_particles)
        box_size = snap_header.box_size
        redshift = snap_header.redshift
        scale_factor = snap_header.time

        print(f"  Particles: {n_particles:,}")
        print(f"  Grid: {grid_size}^3")
        print(f"  Box size: {box_size}")
        print(f"  Redshift: {redshift:.4f}")
        print(f"  Scale factor: {scale_factor:.4f}")
        print(f"  Load time: {t_load:.2f}s, Peak memory: {peak_mem_load / 1024**2:.1f} MB")

        # Convert to x-major ordering (required for original ORIGAMI)
        print("\nConverting to x-major ordering...")
        t_convert_start = time.perf_counter()
        tracemalloc.start()

        sorted_positions = convert_to_xmajor(positions, particle_ids, grid_size)

        current_mem, peak_mem_convert = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t_convert = time.perf_counter() - t_convert_start
        print(f"  Convert time: {t_convert:.2f}s, Peak memory: {peak_mem_convert / 1024**2:.1f} MB")

        # Run original ORIGAMI
        print("\nRunning original ORIGAMI code...")
        orig_tags, t_original = run_original_origami(sorted_positions, box_size, grid_size)
        print(f"  Original ORIGAMI time: {t_original:.2f}s")

        # Run tessera ORIGAMI
        print("Running tessera ORIGAMI...")
        tessera_tags, tessera_result, t_tessera, peak_mem_tessera = run_tessera_origami(
            sorted_positions, box_size, grid_size
        )
        print(f"  tessera time: {t_tessera:.2f}s, Peak memory: {peak_mem_tessera / 1024**2:.1f} MB")

        # Compute speedup
        speedup = t_original / t_tessera if t_tessera > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")

        # Compare results
        print("\nComparing results...")
        stats = compare_results(orig_tags, tessera_tags, tessera_result)

        # Print comparison
        print(f"\n  Match rate: {stats['match_rate']:.6%} ({stats['matches']:,}/{stats['total_particles']:,})")
        print()
        print(f"  {'Class':<10} {'Original':>14} {'tessera':>16} {'Agreement':>12}")
        print(f"  {'-'*10} {'-'*14} {'-'*16} {'-'*12}")

        class_names = ['void', 'wall', 'filament', 'halo']
        for cls_name in class_names:
            s = stats['per_class'][cls_name]
            orig_pct = f"{s['original_count']:,} ({s['original_fraction']:.2%})"
            tessera_pct = f"{s['tessera_count']:,} ({s['tessera_fraction']:.2%})"
            print(f"  {cls_name.capitalize():<10} {orig_pct:>14} {tessera_pct:>16} {s['agreement']:>12,}")

        print(f"\n  Mass Fractions:   Void={stats['mass_fractions']['void']:.2%}, "
              f"Wall={stats['mass_fractions']['wall']:.2%}, "
              f"Filament={stats['mass_fractions']['filament']:.2%}, "
              f"Halo={stats['mass_fractions']['halo']:.2%}")
        print(f"  Volume Fractions: Void={stats['volume_fractions']['void']:.2%}, "
              f"Wall={stats['volume_fractions']['wall']:.2%}, "
              f"Filament={stats['volume_fractions']['filament']:.2%}, "
              f"Halo={stats['volume_fractions']['halo']:.2%}")

        if stats['perfect_match']:
            print("\n  RESULT: PERFECT MATCH")
        else:
            print(f"\n  RESULT: {stats['total_particles'] - stats['matches']:,} disagreements")

        # Prepare benchmarks
        benchmarks = {
            'load_time_s': t_load,
            'convert_time_s': t_convert,
            'original_origami_time_s': t_original,
            'tessera_time_s': t_tessera,
            'speedup': speedup,
            'peak_memory_load_mb': peak_mem_load / 1024**2,
            'peak_memory_convert_mb': peak_mem_convert / 1024**2,
            'peak_memory_tessera_mb': peak_mem_tessera / 1024**2,
        }

        # Save results
        print("\nSaving results...")
        _, json_path = save_results(
            SCRIPT_DIR, label, orig_tags, tessera_tags, stats, benchmarks,
            box_size, redshift, scale_factor, description
        )
        print(f"  Saved: {json_path.name}")

        # Generate comparison images
        print("Generating images...")
        img_path = SCRIPT_DIR / f"origami_{label}.png"
        create_origami_validation_figure(
            stats, img_path,
            title=f"ORIGAMI Validation: {time_label} - {description}",
        )
        print(f"  Saved: {img_path.name}")

        # Generate morphology slices if grid data available
        if hasattr(tessera_result, 'morphology_grid') and len(tessera_result.morphology_grid) > 0:
            morph_grid = np.array(tessera_result.morphology_grid)
            slice_path = SCRIPT_DIR / f"morphology_slices_{label}.png"
            plot_morphology_slices_3panel(
                morph_grid, slice_path,
                title=f"Morphology Grid: {time_label}",
            )
            print(f"  Saved: {slice_path.name}")

        # Store for summary
        all_results[label] = {
            'stats': stats,
            'benchmarks': benchmarks,
        }

    # Save combined summary
    summary = {
        'validation_date': datetime.now().isoformat(),
        'reference': 'Falck, Neyrinck & Szalay 2012, ApJ 754, 126',
        'snapshots': {}
    }

    all_passed = True
    for label, data in all_results.items():
        stats = data['stats']
        benchmarks = data['benchmarks']
        summary['snapshots'][label] = {
            'match_rate': stats['match_rate'],
            'perfect_match': stats['perfect_match'],
            'mass_fractions': stats['mass_fractions'],
            'volume_fractions': stats['volume_fractions'],
            'benchmarks': benchmarks,
        }
        if not stats['perfect_match']:
            all_passed = False

    summary['all_tests_passed'] = all_passed

    summary_path = SCRIPT_DIR / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print("\n  Benchmark Summary:")
    print(f"  {'Snapshot':<15} {'Original (s)':<14} {'tessera (s)':<14} {'Speedup':<10} {'Match Rate':<12} {'Status':<8}")
    print(f"  {'-'*15} {'-'*14} {'-'*14} {'-'*10} {'-'*12} {'-'*8}")
    for label, data in all_results.items():
        stats = data['stats']
        benchmarks = data['benchmarks']
        status = "PASS" if stats['perfect_match'] else "FAIL"
        print(f"  {label:<15} {benchmarks['original_origami_time_s']:<14.2f} {benchmarks['tessera_time_s']:<14.2f} {benchmarks['speedup']:<10.2f}x {stats['match_rate']:<12.4%} {status:<8}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
