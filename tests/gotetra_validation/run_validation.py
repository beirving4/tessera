#!/usr/bin/env python3
"""
Gotetra Validation: Compare tessera against original gotetra code.

This script validates the tessera tetrahedron-based density computation
by comparing its output against pre-rendered gotetra density fields.

The validation compares:
1. Full box z-projection density fields for snap034 (a=1) and snap074 (a=100)
2. Halo subbox (10 cMpc/h) z-projection density fields tracking Group 0 across time

Results are saved as JSON summaries and 2x3 comparison images showing:
- Top row: tessera density, gotetra density, relative difference
- Bottom row: Overdensity PDF comparison, scatter plot, relative difference histogram

Memory and runtime benchmarks are included for comparison.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import time
import tracemalloc

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Avoid OpenMP threading issues on macOS

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add paths
sys.path.insert(0, str(REPO_ROOT / 'tests'))
sys.path.insert(0, str(REPO_ROOT))

# Try to import local config
try:
    from config import SNAPSHOT_BASE, GOTETRA_BASE
except ImportError:
    raise ImportError(
        "Please create tests/config.py with your local paths.\n"
        "Copy config.example.py to config.py and update the paths."
    )

# Import shared utilities
from utils import (
    load_snapshot,
    sort_positions_lagrangian,
    infer_grid_size,
    compute_mean_density,
    check_tessera_available,
)

# Import tessera
try:
    import tessera as ts
except ImportError:
    import _tessera as ts


# Validation configurations
VALIDATIONS = [
    {
        'name': 'snap034_fullbox',
        'snapshot': 'snapshot_034',
        'gotetra_ref': GOTETRA_BASE / 'gotetra_test' / 'output' / 'full_z_projection.gtet',
        'description': 'Full Box (a=1, z=0)',
        'subbox': None,
    },
    {
        'name': 'snap074_fullbox',
        'snapshot': 'snapshot_074',
        'gotetra_ref': GOTETRA_BASE / 'gotetra_validation_refs' / 'snap074_fullbox' / 'full_box.gtet',
        'description': 'Full Box (a=100, z=-0.99)',
        'subbox': None,
    },
    {
        'name': 'snap034_halo',
        'snapshot': 'snapshot_034',
        'gotetra_ref': GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap034' / 'halo_density.gtet',
        'description': 'Group 0 Halo (a=1, z=0)',
        'subbox': 'from_header',
    },
    {
        'name': 'snap074_halo',
        'snapshot': 'snapshot_074',
        'gotetra_ref': GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap074' / 'halo_density.gtet',
        'description': 'Group 0 Halo (a=100, z=-0.99)',
        'subbox': {
            'origin': (147.52755737304688, 146.6825408935547, 75.38079071044922),
            'width': 10.080586080586081,
        },
    },
]


def read_gotetra_reference(gtet_path):
    """Read reference density field from gotetra .gtet file."""
    from tessera.gotetra_compat import read_header, read_grid
    header = read_header(str(gtet_path))
    grid = read_grid(str(gtet_path))
    return grid, header


def compute_density_2d_projection(sorted_positions, box_size, grid_size, output_cells,
                                   subbox=None, scale_factor=1.0):
    """Compute 2D z-projected density field using tessera."""
    # Scale to physical coordinates if requested
    if scale_factor != 1.0:
        positions_scaled = sorted_positions * scale_factor
        box_size_scaled = box_size * scale_factor
    else:
        positions_scaled = sorted_positions
        box_size_scaled = box_size

    config = ts.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = box_size_scaled
    config.output_cells = output_cells
    config.n_threads = 1
    config.n_samples = 50
    config.periodic = True
    config.particle_mass = 1.0

    if subbox is not None:
        config.subbox_enabled = True
        width = subbox['width']
        if scale_factor != 1.0:
            width_scaled = width * scale_factor
        else:
            width_scaled = width

        if 'origin' in subbox:
            origin = subbox['origin']
            if scale_factor != 1.0:
                config.subbox_origin = (origin[0] * scale_factor, origin[1] * scale_factor, origin[2] * scale_factor)
            else:
                config.subbox_origin = (origin[0], origin[1], origin[2])
        else:
            center = subbox['center']
            if scale_factor != 1.0:
                center_scaled = (center[0] * scale_factor, center[1] * scale_factor, center[2] * scale_factor)
                config.subbox_origin = (center_scaled[0] - width_scaled/2, center_scaled[1] - width_scaled/2, center_scaled[2] - width_scaled/2)
            else:
                config.subbox_origin = (center[0] - width/2, center[1] - width/2, center[2] - width/2)
        config.subbox_width = (width_scaled, width_scaled, width_scaled)

    result = ts.density.compute_tetra_density_2d_projection(positions_scaled, config, 2)
    return result.density


def compute_density_2d_direct(sorted_positions, box_size, grid_size, output_cells,
                               slice_min, slice_max, scale_factor=1.0):
    """Compute 2D slice density field directly (memory-efficient)."""
    if scale_factor != 1.0:
        positions_scaled = sorted_positions * scale_factor
        box_size_scaled = box_size * scale_factor
        slice_min_scaled = slice_min * scale_factor
        slice_max_scaled = slice_max * scale_factor
    else:
        positions_scaled = sorted_positions
        box_size_scaled = box_size
        slice_min_scaled = slice_min
        slice_max_scaled = slice_max

    config = ts.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = box_size_scaled
    config.output_cells = output_cells
    config.n_threads = 1
    config.n_samples = 50
    config.periodic = True
    config.particle_mass = 1.0

    result = ts.density.compute_tetra_density_2d_direct(
        positions_scaled, config, 2, slice_min_scaled, slice_max_scaled
    )
    return result.density


def compare_density_fields(at_density, gotetra_density):
    """Compare two density fields and compute statistics."""
    at_norm = at_density / np.mean(at_density)
    gt_norm = gotetra_density / np.mean(gotetra_density)

    at_flat = at_norm.flatten()
    gt_flat = gt_norm.flatten()

    correlation = np.corrcoef(at_flat, gt_flat)[0, 1]

    mask = gt_norm > 0
    rel_diff = np.zeros_like(at_norm)
    rel_diff[mask] = (at_norm[mask] - gt_norm[mask]) / gt_norm[mask]

    mean_rel_diff = np.mean(np.abs(rel_diff[mask])) if np.any(mask) else 0.0
    max_rel_diff = np.max(np.abs(rel_diff[mask])) if np.any(mask) else 0.0

    return {
        'correlation': float(correlation),
        'mean_relative_difference': float(mean_rel_diff),
        'max_relative_difference': float(max_rel_diff),
        'at_density_range': [float(at_density.min()), float(at_density.max())],
        'gotetra_density_range': [float(gotetra_density.min()), float(gotetra_density.max())],
        'passed': bool(correlation > 0.999)
    }, at_norm, gt_norm, rel_diff


def create_comparison_figure(at_density, gotetra_density, at_norm, gt_norm, rel_diff,
                             title, output_path, stats, extent=None, coord_label='cMpc/h'):
    """Create a 2x3 comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import stats as scipy_stats

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Density comparison
    vmin = np.log10(np.maximum(at_norm, 0.01).min())
    vmax = np.log10(at_norm.max())

    im1 = axes[0, 0].imshow(np.log10(np.maximum(at_norm, 0.01)), origin='lower',
                            cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0, 0].set_title(r'${\rm tessera}$')
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label=r'$\log_{10}(1+\delta)$')

    im2 = axes[0, 1].imshow(np.log10(np.maximum(gt_norm, 0.01)), origin='lower',
                            cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0, 1].set_title(r'${\rm Gotetra\ (Reference)}$')
    divider2 = make_axes_locatable(axes[0, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label=r'$\log_{10}(1+\delta)$')

    diff_abs_max = np.percentile(np.abs(rel_diff[np.isfinite(rel_diff)]), 99)
    im3 = axes[0, 2].imshow(rel_diff, origin='lower', cmap='RdBu_r',
                            vmin=-diff_abs_max, vmax=diff_abs_max, extent=extent)
    axes[0, 2].set_title(r'${\rm Relative\ Difference}$')
    divider3 = make_axes_locatable(axes[0, 2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3, label=r'${\rm (tessera - Gotetra) / Gotetra}$')

    # Bottom row: Statistical comparisons
    bins = np.logspace(-2, 4, 100)
    at_flat = at_norm.flatten()
    gt_flat = gt_norm.flatten()

    axes[1, 0].hist(at_flat[at_flat > 0], bins=bins, alpha=0.7, label=r'${\rm tessera}$',
                    color='#e74c3c', density=True)
    axes[1, 0].hist(gt_flat[gt_flat > 0], bins=bins, alpha=0.7, label=r'${\rm Gotetra}$',
                    color='#3498db', density=True)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel(r'$1+\delta$')
    axes[1, 0].set_ylabel(r'${\rm PDF}$')
    axes[1, 0].set_title(r'${\rm Overdensity\ PDF}$')
    axes[1, 0].legend()

    n_sample = min(10000, len(at_flat))
    idx = np.random.choice(len(at_flat), n_sample, replace=False)
    axes[1, 1].scatter(gt_flat[idx], at_flat[idx], s=1, alpha=0.5, c='#2c3e50')
    lims = [min(gt_flat[idx].min(), at_flat[idx].min()),
            max(gt_flat[idx].max(), at_flat[idx].max())]
    axes[1, 1].plot(lims, lims, 'r-', linewidth=2, label=r'$1:1$')
    spearman_r, _ = scipy_stats.spearmanr(at_flat, gt_flat)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel(r'${\rm Gotetra}\ (1+\delta)$')
    axes[1, 1].set_ylabel(r'${\rm tessera}\ (1+\delta)$')
    axes[1, 1].set_title(r'$r=' + f'{stats["correlation"]:.6f}$')
    axes[1, 1].legend()

    rel_diff_flat = rel_diff.flatten()
    valid_diff = rel_diff_flat[np.isfinite(rel_diff_flat) & (gt_flat > 0)]
    hist_min = np.percentile(valid_diff, 1)
    hist_max = np.percentile(valid_diff, 99)
    hist_range = (min(hist_min, -0.5), max(hist_max, 0.5))

    axes[1, 2].hist(valid_diff, bins=100, range=hist_range, alpha=0.7, color='#27ae60', density=True)
    axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 2].axvline(x=np.mean(valid_diff), color='blue', linestyle='-', linewidth=2,
                       label=f'Mean: {np.mean(valid_diff):.4f}')
    axes[1, 2].set_xlabel(r'${\rm Relative\ Difference}$')
    axes[1, 2].set_ylabel(r'${\rm PDF}$')
    axes[1, 2].set_title(f'Diff Distribution (σ={np.std(valid_diff):.4f})')
    axes[1, 2].legend()

    status = "PASS" if stats['passed'] else "FAIL"
    plt.suptitle(f'{title}\nCorrelation: {stats["correlation"]:.6f} | {status}', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_validation(config):
    """Run a single validation test with benchmarking."""
    name = config['name']
    snapshot_name = config['snapshot']
    gotetra_ref = config['gotetra_ref']
    description = config['description']
    subbox_config = config['subbox']

    print(f"\n{'=' * 70}")
    print(f"Validation: {name} - {description}")
    print("=" * 70)

    if not gotetra_ref.exists():
        print(f"  WARNING: Reference file not found: {gotetra_ref}")
        return None

    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"
    if not snapshot_path.exists():
        print(f"  WARNING: Snapshot not found: {snapshot_path}")
        return None

    # Read gotetra reference
    print("Reading gotetra reference...")
    gotetra_density, header = read_gotetra_reference(gotetra_ref)
    output_cells = gotetra_density.shape[0]
    print(f"  Reference shape: {gotetra_density.shape}")

    # Extract subbox parameters
    subbox = None
    if subbox_config == 'from_header':
        origin = header.loc.origin
        span = header.loc.span
        width = float(span[0])
        center = (origin[0] + width/2, origin[1] + width/2, origin[2] + width/2)
        subbox = {'center': center, 'width': width, 'origin': tuple(origin), 'span': tuple(span)}
        print(f"  Subbox from header: origin={origin[:3]}, width={width:.4f}")
    elif subbox_config is not None:
        subbox = subbox_config.copy()
        if 'origin' in subbox and 'center' not in subbox:
            width = subbox['width']
            origin = subbox['origin']
            subbox['center'] = (origin[0] + width/2, origin[1] + width/2, origin[2] + width/2)
        print(f"  Subbox (explicit): width={subbox['width']:.4f}")

    # Load snapshot using shared utilities
    print("Loading snapshot...")
    t_load_start = time.perf_counter()
    tracemalloc.start()

    positions, particle_ids, snap_header = load_snapshot(
        snapshot_path,
        particle_type=1,
        read_ids=True
    )

    current_mem, peak_mem_load = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_load = time.perf_counter() - t_load_start

    n_particles = len(positions)
    grid_size = infer_grid_size(n_particles)
    box_size = snap_header.box_size

    print(f"  Particles: {n_particles:,}")
    print(f"  Grid: {grid_size}^3")
    print(f"  Box size: {box_size}")
    print(f"  Load time: {t_load:.2f}s, Peak memory: {peak_mem_load / 1024**2:.1f} MB")

    # Sort positions
    print("Sorting positions to Lagrangian order...")
    t_sort_start = time.perf_counter()
    tracemalloc.start()

    sorted_positions = sort_positions_lagrangian(positions, particle_ids, grid_size)

    current_mem, peak_mem_sort = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_sort = time.perf_counter() - t_sort_start
    print(f"  Sort time: {t_sort:.2f}s, Peak memory: {peak_mem_sort / 1024**2:.1f} MB")

    # Trim gotetra for full box
    if subbox is None:
        gotetra_density = gotetra_density[:-1, :-1]
        output_cells = gotetra_density.shape[0]
        print(f"  Trimmed gotetra to {output_cells}x{output_cells}")

    # Compute tessera density
    print(f"Computing tessera density ({output_cells}x{output_cells})...")
    t_compute_start = time.perf_counter()
    tracemalloc.start()

    at_density = compute_density_2d_projection(sorted_positions, box_size, grid_size,
                                                output_cells, subbox=subbox)

    current_mem, peak_mem_compute = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_compute = time.perf_counter() - t_compute_start

    print(f"  Compute time: {t_compute:.2f}s, Peak memory: {peak_mem_compute / 1024**2:.1f} MB")
    print(f"  AT range: [{at_density.min():.4e}, {at_density.max():.4e}]")

    # Compare
    print("Comparing density fields...")
    stats, at_norm, gt_norm, rel_diff = compare_density_fields(at_density, gotetra_density)

    print(f"\n  Correlation: {stats['correlation']:.6f}")
    print(f"  Mean relative difference: {stats['mean_relative_difference']:.4e}")
    print(f"  RESULT: {'PASS' if stats['passed'] else 'FAIL'}")

    # Compute extent
    if subbox is not None:
        origin = subbox.get('origin', (0, 0, 0))
        width = subbox['width']
        extent = [origin[0], origin[0] + width, origin[1], origin[1] + width]
    else:
        extent = [0, box_size, 0, box_size]

    # Create comparison figure
    image_path = SCRIPT_DIR / f"{name}_comparison.png"
    print(f"\nGenerating comparison figure...")
    create_comparison_figure(at_density, gotetra_density, at_norm, gt_norm, rel_diff,
                            description, image_path, stats, extent=extent)
    print(f"  Saved: {image_path.name}")

    # Prepare result with benchmarks
    result = {
        'name': name,
        'description': description,
        'snapshot': snapshot_name,
        'redshift': snap_header.redshift,
        'scale_factor': snap_header.time,
        'box_size': box_size,
        'output_cells': output_cells,
        'benchmarks': {
            'load_time_s': t_load,
            'sort_time_s': t_sort,
            'compute_time_s': t_compute,
            'total_time_s': t_load + t_sort + t_compute,
            'peak_memory_load_mb': peak_mem_load / 1024**2,
            'peak_memory_sort_mb': peak_mem_sort / 1024**2,
            'peak_memory_compute_mb': peak_mem_compute / 1024**2,
        },
        **stats
    }

    if subbox:
        result['subbox_center'] = list(subbox['center'])
        result['subbox_width'] = subbox['width']

    # Save JSON
    json_path = SCRIPT_DIR / f"{name}_validation.json"
    with open(json_path, 'w') as f:
        json.dump({
            'created': datetime.now().isoformat(),
            **result
        }, f, indent=2)
    print(f"  Saved: {json_path.name}")

    return result


def main():
    print("=" * 70)
    print("Gotetra Validation: tessera vs Original Code")
    print("=" * 70)
    print(f"Output directory: {SCRIPT_DIR}")
    print()

    all_results = {}

    for config in VALIDATIONS:
        result = run_validation(config)
        if result:
            all_results[config['name']] = result

    # Save summary
    all_passed = all(r.get('passed', False) for r in all_results.values())
    summary = {
        'validation_date': datetime.now().isoformat(),
        'reference': 'gotetra: github.com/phil-mansfield/gotetra',
        'method': 'Tetrahedron-based phase-space tessellation for density field computation',
        'results': all_results,
        'all_tests_passed': all_passed
    }

    summary_path = SCRIPT_DIR / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print("\n  Benchmark Summary:")
    print(f"  {'Test':<20} {'Time (s)':<12} {'Peak Mem (MB)':<15} {'Correlation':<12} {'Status':<8}")
    print(f"  {'-'*20} {'-'*12} {'-'*15} {'-'*12} {'-'*8}")
    for name, result in all_results.items():
        bench = result['benchmarks']
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {name:<20} {bench['total_time_s']:<12.2f} {bench['peak_memory_compute_mb']:<15.1f} {result['correlation']:<12.6f} {status:<8}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
