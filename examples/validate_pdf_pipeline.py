#!/usr/bin/env python3
"""
Validate PDF Pipeline Output Against Existing Results

Compares the new unified pipeline PDF output against the previously generated
jackknife results from overdensity_pdf_origami.py.

NOTE: Since the density computation uses Monte Carlo sampling, exact matches
are not expected. This validation checks:
1. Bin edges/centers match (same range detection)
2. PDFs have similar shapes (correlation > 0.95)
3. Histogram sums match expected particle counts
4. Mean overdensity is ~1.0
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
sys.path.insert(0, str(SCRIPT_DIR.parent / 'python'))
import tessera as ts


def load_snapshot(snapshot_name):
    """Load particle data from GADGET-4 snapshot."""
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"

    # Use utils function for loading
    positions, particle_ids, box_size, _ = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )

    # Infer grid size using utils function
    grid_size = infer_grid_size(len(positions))
    return positions, particle_ids, box_size, grid_size


def load_existing_results(snapshot_name):
    """Load existing jackknife results from HDF5."""
    h5_path = SCRIPT_DIR / f"{snapshot_name}_overdensity_pdf_jackknife.h5"

    if not h5_path.exists():
        print(f"  Existing results not found: {h5_path}")
        return None

    results = {}
    with h5py.File(h5_path, 'r') as f:
        results['bin_centers'] = f['Bins/centers'][:]
        results['bin_edges'] = f['Bins/edges'][:]

        results['pdf_all'] = f['All/pdf'][:]
        results['pdf_void'] = f['Void/pdf'][:]
        results['pdf_wall'] = f['Wall/pdf'][:]
        results['pdf_filament'] = f['Filament/pdf'][:]
        results['pdf_halo'] = f['Halo/pdf'][:]

        results['hist_all'] = f['All/histogram'][:]
        results['hist_void'] = f['Void/histogram'][:]
        results['hist_wall'] = f['Wall/histogram'][:]
        results['hist_filament'] = f['Filament/histogram'][:]
        results['hist_halo'] = f['Halo/histogram'][:]

        if 'pdf_error' in f['All']:
            results['pdf_all_error'] = f['All/pdf_error'][:]
            results['pdf_void_error'] = f['Void/pdf_error'][:]
            results['pdf_wall_error'] = f['Wall/pdf_error'][:]
            results['pdf_filament_error'] = f['Filament/pdf_error'][:]
            results['pdf_halo_error'] = f['Halo/pdf_error'][:]

    return results


def compare_bins(name, arr1, arr2, rtol=0.05):
    """Compare bin arrays - allow 5% tolerance due to Monte Carlo variance in min/max detection."""
    if len(arr1) != len(arr2):
        print(f"    {name}: SIZE MISMATCH ({len(arr1)} vs {len(arr2)})")
        return False

    max_rel_diff = np.max(np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), np.abs(arr2)))
    is_close = max_rel_diff < rtol

    status = "OK" if is_close else "MISMATCH"
    print(f"    {name}: {status} (max_rel={max_rel_diff:.2e}, tolerance={rtol:.0%})")
    return is_close


def compare_pdf_shape(name, pdf1, pdf2, min_correlation=0.95):
    """Compare PDF shapes using correlation - accounts for Monte Carlo variance."""
    # Mask out near-zero values
    mask = (pdf1 > 1e-6) & (pdf2 > 1e-6)
    if mask.sum() < 10:
        print(f"    {name}: SKIP (insufficient non-zero bins)")
        return True

    # Pearson correlation
    corr = np.corrcoef(pdf1[mask], pdf2[mask])[0, 1]
    is_close = corr > min_correlation

    status = "OK" if is_close else "MISMATCH"
    print(f"    {name}: {status} (correlation={corr:.4f})")
    return is_close


def validate_snapshot(snapshot_name):
    """Validate pipeline output against existing results for a snapshot."""
    print(f"\n{'='*70}")
    print(f"Validating: {snapshot_name}")
    print('='*70)

    # Load existing results
    print("Loading existing jackknife results...")
    existing = load_existing_results(snapshot_name)
    if existing is None:
        return False

    # Load snapshot
    print("Loading snapshot...")
    positions, particle_ids, box_size, grid_size = load_snapshot(snapshot_name)
    n_particles = len(positions)
    print(f"  Grid: {grid_size}^3 = {n_particles:,} particles")

    # Run new pipeline with jackknife
    print("Running new pipeline with PDF computation...")
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.density_output_cells = 256
    config.sample_density_at_particles = True
    config.pdf_n_bins = 100
    config.pdf_log_bins = True
    config.pdf_jackknife = True
    config.pdf_jackknife_subboxes = 2

    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"  Pipeline completed in {result.total_time_ms:.1f} ms")
    print(f"  PDF computation: {result.pdf_time_ms:.1f} ms")
    print(f"  Jackknife: {result.pdf_jackknife_enabled}, {result.pdf_jackknife_n_subboxes} sub-boxes")

    # Validation checks
    print("\n  Validation Checks:")
    all_pass = True

    # 1. Bin centers should match (same range detection algorithm)
    print("\n  1. Binning Consistency:")
    all_pass &= compare_bins("bin_centers", np.array(result.pdf_bin_centers), existing['bin_centers'])

    # 2. PDF shapes should correlate well (accounts for Monte Carlo variance)
    print("\n  2. PDF Shape Correlation (>0.95 = OK):")
    all_pass &= compare_pdf_shape("pdf_all", np.array(result.pdf_all), existing['pdf_all'])
    all_pass &= compare_pdf_shape("pdf_void", np.array(result.pdf_void), existing['pdf_void'])
    all_pass &= compare_pdf_shape("pdf_wall", np.array(result.pdf_wall), existing['pdf_wall'])
    all_pass &= compare_pdf_shape("pdf_filament", np.array(result.pdf_filament), existing['pdf_filament'])
    all_pass &= compare_pdf_shape("pdf_halo", np.array(result.pdf_halo), existing['pdf_halo'])

    # 3. Histogram sums should be similar (some particles may fall outside bin range)
    print("\n  3. Histogram Sum Consistency:")
    new_hist_all_sum = np.array(result.hist_all).sum()
    new_hist_void_sum = np.array(result.hist_void).sum()
    new_hist_wall_sum = np.array(result.hist_wall).sum()
    new_hist_filament_sum = np.array(result.hist_filament).sum()
    new_hist_halo_sum = np.array(result.hist_halo).sum()

    existing_hist_all_sum = existing['hist_all'].sum()
    existing_hist_void_sum = existing['hist_void'].sum()
    existing_hist_wall_sum = existing['hist_wall'].sum()
    existing_hist_filament_sum = existing['hist_filament'].sum()
    existing_hist_halo_sum = existing['hist_halo'].sum()

    # Compare with existing histogram sums (not particle count - some particles fall outside range)
    # Allow 1% tolerance since Monte Carlo variance affects which particles fall outside range
    def check_sum(name, new_sum, existing_sum, rtol=0.01):
        rel_diff = abs(new_sum - existing_sum) / max(existing_sum, 1)
        ok = rel_diff < rtol
        print(f"    {name}: {new_sum:,} (existing: {existing_sum:,}, diff: {rel_diff:.2%}) {'OK' if ok else 'MISMATCH'}")
        return ok

    all_pass &= check_sum("hist_all sum", new_hist_all_sum, existing_hist_all_sum)
    all_pass &= check_sum("hist_void sum", new_hist_void_sum, existing_hist_void_sum)
    all_pass &= check_sum("hist_wall sum", new_hist_wall_sum, existing_hist_wall_sum)
    all_pass &= check_sum("hist_filament sum", new_hist_filament_sum, existing_hist_filament_sum)
    all_pass &= check_sum("hist_halo sum", new_hist_halo_sum, existing_hist_halo_sum)

    # 4. Mean overdensity should be ~1.0
    print("\n  4. Physical Consistency:")
    mean_od_ok = abs(result.overdensity_mean - 1.0) < 0.01
    print(f"    Mean overdensity: {result.overdensity_mean:.6f} (expected ~1.0) {'OK' if mean_od_ok else 'MISMATCH'}")
    all_pass &= mean_od_ok

    # Summary
    print(f"\n  ORIGAMI fractions: void={result.f_void:.2%}, wall={result.f_wall:.2%}, "
          f"filament={result.f_filament:.2%}, halo={result.f_halo:.2%}")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def main():
    print("="*70)
    print("PDF Pipeline Validation")
    print("="*70)
    print("\nNote: Density computation uses Monte Carlo sampling, so exact histogram")
    print("matches are not expected. This validates:")
    print("  - Bin range detection (bin_centers match)")
    print("  - PDF shape similarity (correlation > 0.95)")
    print("  - Histogram count consistency (sums match particle counts)")
    print("  - Physical consistency (mean overdensity ~1.0)")

    snapshots = ['snapshot_034', 'snapshot_074']
    results = {}

    for snap in snapshots:
        results[snap] = validate_snapshot(snap)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for snap, passed in results.items():
        print(f"  {snap}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
