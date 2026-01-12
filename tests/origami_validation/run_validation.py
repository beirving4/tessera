#!/usr/bin/env python3
"""
ORIGAMI Validation: Compare AsymptoticTetra against original ORIGAMI code.

This script validates the AsymptoticTetra ORIGAMI implementation by comparing
its output against the original ORIGAMI code from Falck, Neyrinck & Szalay (2012).

Results are saved to this directory for reproducibility and documentation.
"""

import numpy as np
import h5py
import struct
import subprocess
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil

# Configuration - adjust these paths as needed
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add build directory to path
sys.path.insert(0, str(REPO_ROOT / 'build'))

# Path to original ORIGAMI binary
ORIGAMI_BIN = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/origami/code/origamitag")

# Snapshots to validate
SNAPSHOT_BASE = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output")
SNAPSHOTS = [
    ("snapshot_034.hdf5", "z=0 (a=1)", "Present day"),
    ("snapshot_074.hdf5", "z=-0.99 (a=100)", "Far future"),
]


def load_snapshot(snapshot_path):
    """Load particle data from GADGET-4 snapshot."""
    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = float(f['Header'].attrs['BoxSize'])
        redshift = f['Header'].attrs['Redshift']
        scale_factor = f['Header'].attrs['Time']
    return positions, particle_ids, float(box_size), float(redshift), float(scale_factor)


def convert_to_xmajor(positions, particle_ids, grid_size):
    """Convert GADGET z-major ordering to ORIGAMI x-major ordering."""
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
    """Run original ORIGAMI code."""
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

    # Run original ORIGAMI
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    result = subprocess.run(
        [str(ORIGAMI_BIN), str(params_file)],
        capture_output=True,
        text=True,
        env=env
    )

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

    return tags


def run_asymptotic_origami(sorted_positions, box_size, grid_size):
    """Run AsymptoticTetra ORIGAMI implementation."""
    import _asymptotic_tetra as at

    config = at.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    result = at.origami.compute_morphology(sorted_positions, config)

    # Deposit to grid to compute volume fractions
    at.origami.deposit_morphology_to_grid(sorted_positions, box_size, 128, result)

    return np.array(result.morphology), result


def compare_results(orig_tags, asym_tags, asym_result):
    """Compare morphology classifications."""
    n = len(orig_tags)
    matches = np.sum(orig_tags == asym_tags)
    match_rate = matches / n

    class_names = ['Void', 'Wall', 'Filament', 'Halo']
    stats = {
        'total_particles': int(n),
        'matches': int(matches),
        'match_rate': float(match_rate),
        'perfect_match': bool(match_rate == 1.0),
        'mass_fractions': {
            'void': float(asym_result.f_void),
            'wall': float(asym_result.f_wall),
            'filament': float(asym_result.f_filament),
            'halo': float(asym_result.f_halo),
        },
        'volume_fractions': {
            'void': float(asym_result.v_void),
            'wall': float(asym_result.v_wall),
            'filament': float(asym_result.v_filament),
            'halo': float(asym_result.v_halo),
        },
        'per_class': {}
    }

    for cls_idx, cls_name in enumerate(class_names):
        orig_count = int(np.sum(orig_tags == cls_idx))
        asym_count = int(np.sum(asym_tags == cls_idx))
        agreement = int(np.sum((orig_tags == cls_idx) & (asym_tags == cls_idx)))
        stats['per_class'][cls_name.lower()] = {
            'original_count': orig_count,
            'asymptotic_count': asym_count,
            'agreement': agreement,
            'original_fraction': orig_count / n,
            'asymptotic_fraction': asym_count / n,
        }

    return stats


def save_results(output_dir, label, orig_tags, asym_tags, stats, box_size, redshift, scale_factor, description):
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
            **stats
        }, f, indent=2)

    return None, json_path


def main():
    print("=" * 70)
    print("ORIGAMI Validation: AsymptoticTetra vs Original Code")
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

        # Load snapshot
        print("Loading snapshot...")
        positions, particle_ids, box_size, redshift, scale_factor = load_snapshot(snap_path)
        n_particles = len(positions)
        grid_size = int(round(n_particles ** (1/3)))

        print(f"  Particles: {n_particles:,}")
        print(f"  Grid: {grid_size}^3")
        print(f"  Box size: {box_size}")
        print(f"  Redshift: {redshift:.4f}")
        print(f"  Scale factor: {scale_factor:.4f}")

        # Convert to x-major ordering
        print("\nConverting to x-major ordering...")
        sorted_positions = convert_to_xmajor(positions, particle_ids, grid_size)

        # Run original ORIGAMI
        print("Running original ORIGAMI code...")
        orig_tags = run_original_origami(sorted_positions, box_size, grid_size)

        # Run AsymptoticTetra ORIGAMI
        print("Running AsymptoticTetra ORIGAMI...")
        asym_tags, asym_result = run_asymptotic_origami(sorted_positions, box_size, grid_size)

        # Compare results
        print("\nComparing results...")
        stats = compare_results(orig_tags, asym_tags, asym_result)

        # Print comparison
        print(f"\n  Match rate: {stats['match_rate']:.6%} ({stats['matches']:,}/{stats['total_particles']:,})")
        print()
        print(f"  {'Class':<10} {'Original':>14} {'AsymptoticTetra':>16} {'Agreement':>12}")
        print(f"  {'-'*10} {'-'*14} {'-'*16} {'-'*12}")

        class_names = ['void', 'wall', 'filament', 'halo']
        for cls_name in class_names:
            s = stats['per_class'][cls_name]
            orig_pct = f"{s['original_count']:,} ({s['original_fraction']:.2%})"
            asym_pct = f"{s['asymptotic_count']:,} ({s['asymptotic_fraction']:.2%})"
            print(f"  {cls_name.capitalize():<10} {orig_pct:>14} {asym_pct:>16} {s['agreement']:>12,}")

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

        # Save results
        print("\nSaving results...")
        _, json_path = save_results(
            SCRIPT_DIR, label, orig_tags, asym_tags, stats,
            box_size, redshift, scale_factor, description
        )
        print(f"  Saved: {json_path.name}")

        all_results[label] = stats

    # Save combined summary
    summary = {
        'validation_date': datetime.now().isoformat(),
        'origami_binary': str(ORIGAMI_BIN),
        'asymptotic_tetra_repo': str(REPO_ROOT),
        'reference': 'Falck, Neyrinck & Szalay 2012, ApJ 754, 126',
        'snapshots': {}
    }

    all_passed = True
    for label, stats in all_results.items():
        summary['snapshots'][label] = {
            'match_rate': stats['match_rate'],
            'perfect_match': stats['perfect_match'],
            'mass_fractions': stats['mass_fractions'],
            'volume_fractions': stats['volume_fractions'],
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
    print(f"\nResults saved to: {SCRIPT_DIR}")
    print("\nSummary:")
    for label, stats in all_results.items():
        status = "PASS" if stats['perfect_match'] else "FAIL"
        print(f"  {label}: {status} ({stats['match_rate']:.4%} match)")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    sys.exit(main())
