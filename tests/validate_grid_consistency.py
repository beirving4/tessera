#!/usr/bin/env python3
"""
Validate that the grid-based Monte Carlo pipeline produces results
consistent with previously saved density PDF results.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import h5py
from pathlib import Path
import sys

# Add tessera build directory
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


ORIGAMI_CLASSES = ['Void', 'Wall', 'Filament', 'Halo']


def load_snapshot(snapshot_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load particle positions, IDs, and box size from snapshot."""
    print(f"  Loading snapshot: {snapshot_path}")

    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = f['Header'].attrs['BoxSize']

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))
    print(f"    Particles: {n_particles:,} ({grid_size}^3)")

    return (
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(particle_ids, dtype=np.int64),
        float(box_size),
    )


def load_saved_results(results_path: Path) -> dict:
    """Load previously saved PDF results from HDF5."""
    print(f"  Loading saved results: {results_path}")

    results = {}
    with h5py.File(results_path, 'r') as f:
        # Bins
        results['bin_edges'] = f['Bins/edges'][:]
        results['bin_centers'] = f['Bins/centers'][:]

        # Header info
        results['mean_density'] = f['Header'].attrs['mean_density']

        # Load data for each class
        for class_name in ['All'] + ORIGAMI_CLASSES:
            grp = f[class_name]
            results[class_name.lower()] = {
                'histogram': grp['histogram'][:],
                'pdf': grp['pdf'][:],
                'n_particles': grp.attrs['n_particles'],
                'mean_overdensity': grp.attrs['mean_overdensity'],
            }

    return results


def run_grid_pipeline(
    positions: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
    grid_size: int,
    n_bins: int = 100,
) -> dict:
    """Run the grid-based MC pipeline and extract results."""
    print("  Running grid-based MC pipeline...")

    # Configure pipeline (grid-based MC is the default)
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.density_output_cells = grid_size
    config.density_n_samples = 100
    config.sample_density_at_particles = True
    config.grid_cells = grid_size
    config.n_threads = 1
    config.sort_in_place = False
    config.pdf_n_bins = n_bins
    config.pdf_log_bins = True
    config.pdf_jackknife = False

    # Ensure we're using grid-based (default)
    config.use_direct_particle_density = False

    # Run pipeline
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"    Pipeline completed in {result.total_time_ms:.1f} ms")
    print(f"    Density time: {result.density_time_ms:.1f} ms")
    print(f"    Sampling time: {result.sampling_time_ms:.1f} ms")

    # Extract results
    morphology = np.array(result.morphology)
    particle_density = np.array(result.particle_density)
    mean_density = result.mean_density

    # Compute overdensity
    overdensity = particle_density / np.mean(particle_density)

    results = {
        'bin_edges': np.array(result.pdf_bin_edges),
        'bin_centers': np.array(result.pdf_bin_centers),
        'mean_density': mean_density,
    }

    # All particles
    results['all'] = {
        'histogram': np.array(result.hist_all),
        'pdf': np.array(result.pdf_all),
        'n_particles': len(particle_density),
        'mean_overdensity': float(np.mean(overdensity)),
    }

    # Per-class
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        mask = morphology == i
        results[class_name.lower()] = {
            'histogram': np.array(getattr(result, f'hist_{class_name.lower()}')),
            'pdf': np.array(getattr(result, f'pdf_{class_name.lower()}')),
            'n_particles': int(mask.sum()),
            'mean_overdensity': float(np.mean(overdensity[mask])) if mask.sum() > 0 else 0.0,
        }

    # ORIGAMI fractions
    results['f_void'] = result.f_void
    results['f_wall'] = result.f_wall
    results['f_filament'] = result.f_filament
    results['f_halo'] = result.f_halo

    return results


def compare_results(saved: dict, computed: dict, name: str) -> bool:
    """Compare saved and computed results."""
    print(f"\n  Comparing results for {name}:")

    all_passed = True

    # Compare ORIGAMI particle counts (allow 0.1% tolerance due to MC noise)
    print(f"    ORIGAMI particle counts:")
    for class_name in ['all'] + [c.lower() for c in ORIGAMI_CLASSES]:
        saved_n = saved[class_name]['n_particles']
        computed_n = computed[class_name]['n_particles']
        rel_diff = abs(saved_n - computed_n) / max(saved_n, 1)
        match = rel_diff < 0.001  # 0.1% tolerance
        if saved_n == computed_n:
            status = "MATCH"
        elif match:
            status = f"~MATCH (diff={rel_diff:.4%})"
        else:
            status = f"MISMATCH (diff={rel_diff:.2%})"
        print(f"      {class_name:>10}: saved={saved_n:>10,}, computed={computed_n:>10,} [{status}]")
        if not match:
            all_passed = False

    # Compare mean overdensity
    print(f"    Mean overdensity:")
    for class_name in ['all'] + [c.lower() for c in ORIGAMI_CLASSES]:
        saved_mean = saved[class_name]['mean_overdensity']
        computed_mean = computed[class_name]['mean_overdensity']
        rel_diff = abs(saved_mean - computed_mean) / max(abs(saved_mean), 1e-10)
        match = rel_diff < 0.01  # 1% tolerance
        status = "MATCH" if match else f"DIFF={rel_diff:.2%}"
        print(f"      {class_name:>10}: saved={saved_mean:.6f}, computed={computed_mean:.6f} [{status}]")
        if not match:
            all_passed = False

    # Compare PDFs (correlation)
    print(f"    PDF correlation:")
    for class_name in ['all'] + [c.lower() for c in ORIGAMI_CLASSES]:
        saved_pdf = saved[class_name]['pdf']
        computed_pdf = computed[class_name]['pdf']

        # Only compare where both have data
        mask = (saved_pdf > 0) & (computed_pdf > 0)
        if mask.sum() > 5:
            # Log-space correlation
            corr = np.corrcoef(np.log10(saved_pdf[mask]), np.log10(computed_pdf[mask]))[0, 1]
            match = corr > 0.99  # High correlation expected
            status = f"r={corr:.4f}" if match else f"LOW r={corr:.4f}"
            print(f"      {class_name:>10}: {status}")
            if not match:
                all_passed = False
        else:
            print(f"      {class_name:>10}: insufficient data for comparison")

    return all_passed


def validate_snapshot(
    snapshot_idx: int,
    snapshot_dir: Path,
    results_dir: Path,
) -> bool:
    """Validate a single snapshot."""

    print(f"\n{'='*70}")
    print(f"Validating Snapshot {snapshot_idx}")
    print('='*70)

    snapshot_path = snapshot_dir / f"snapshot_{snapshot_idx:03d}.hdf5"
    results_path = results_dir / f"L256_N256_primary_density_pdf_{snapshot_idx:03d}_jackknife.hdf5"

    if not snapshot_path.exists():
        print(f"  ERROR: Snapshot not found: {snapshot_path}")
        return False

    if not results_path.exists():
        print(f"  ERROR: Saved results not found: {results_path}")
        return False

    # Load data
    positions, particle_ids, box_size = load_snapshot(snapshot_path)
    grid_size = int(round(len(positions) ** (1/3)))

    # Load saved results
    saved_results = load_saved_results(results_path)
    n_bins = len(saved_results['bin_centers'])

    # Run pipeline with grid-based MC
    computed_results = run_grid_pipeline(
        positions, particle_ids, box_size, grid_size, n_bins
    )

    # Compare
    passed = compare_results(saved_results, computed_results, f"Snapshot {snapshot_idx}")

    if passed:
        print(f"\n  VALIDATION PASSED for Snapshot {snapshot_idx}")
    else:
        print(f"\n  VALIDATION FAILED for Snapshot {snapshot_idx}")

    return passed


def main():
    print("=" * 70)
    print("Grid-Based MC Pipeline Consistency Validation")
    print("=" * 70)
    print("Checking that current pipeline produces results consistent with saved files")

    # Paths
    base_dir = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly")
    sim_dir = base_dir / "Uniform_L256_N256_primary_sandbox/gadget4/output"
    snapshot_dir = sim_dir
    results_dir = sim_dir / "density_pdf_results/results"

    all_passed = True

    for snap_idx in [34, 74]:
        passed = validate_snapshot(snap_idx, snapshot_dir, results_dir)
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL VALIDATIONS PASSED - Grid-based MC is consistent with saved results")
    else:
        print("SOME VALIDATIONS FAILED - Results may differ from saved files")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
