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
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Avoid OpenMP threading issues on macOS

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Try to import local config
sys.path.insert(0, str(REPO_ROOT / 'tests'))
try:
    from config import SNAPSHOT_BASE, GOTETRA_BASE
except ImportError:
    raise ImportError(
        "Please create tests/config.py with your local paths.\n"
        "Copy config.example.py to config.py and update the paths."
    )

# Import tessera (try installed package first, then development build)
try:
    import _tessera as ts
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / 'build'))
    import _tessera as ts

# Validation configurations
# - Use original gotetra reference files that were validated to work
# - Subbox parameters are read from gotetra headers to ensure exact match
VALIDATIONS = [
    {
        'name': 'snap034_fullbox',
        'snapshot': 'snapshot_034',
        'gotetra_ref': GOTETRA_BASE / 'gotetra_test' / 'output' / 'full_z_projection.gtet',
        'description': 'Full Box (a=1, z=0)',
        'subbox': None,  # Full box, no subbox
    },
    {
        'name': 'snap074_fullbox',
        'snapshot': 'snapshot_074',
        'gotetra_ref': GOTETRA_BASE / 'gotetra_validation_refs' / 'snap074_fullbox' / 'full_box.gtet',
        'description': 'Full Box (a=100, z=-0.99)',
        'subbox': None,  # Full box, no subbox
    },
    {
        'name': 'snap034_halo',
        'snapshot': 'snapshot_034',
        'gotetra_ref': GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap034' / 'halo_density.gtet',
        'description': 'Group 0 Halo (a=1, z=0)',
        'subbox': 'from_header',  # Read subbox params from gotetra header
    },
    {
        'name': 'snap074_halo',
        'snapshot': 'snapshot_074',
        'gotetra_ref': GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap074' / 'halo_density.gtet',
        'description': 'Group 0 Halo (a=100, z=-0.99)',
        # Note: gotetra header has incorrect origin (pixel alignment issue)
        # Use bounds.cfg origin and computed width = 129 * pixel_width = 10.0806
        'subbox': {
            'origin': (147.52755737304688, 146.6825408935547, 75.38079071044922),
            'width': 10.080586080586081,  # 129 * (256/3276) = 129 * 0.07814408
        },
    },
]


def load_snapshot(snapshot_path):
    """Load particle data from GADGET-4 snapshot."""
    import h5py
    with h5py.File(snapshot_path, 'r') as f:
        # Convert to required dtypes for tessera
        positions = np.ascontiguousarray(f['PartType1/Coordinates'][:], dtype=np.float64)
        particle_ids = np.ascontiguousarray(f['PartType1/ParticleIDs'][:], dtype=np.int64)
        box_size = float(f['Header'].attrs['BoxSize'])
        redshift = f['Header'].attrs['Redshift']
        scale_factor = f['Header'].attrs['Time']
    return positions, particle_ids, box_size, float(redshift), float(scale_factor)


def sort_positions_lagrangian(positions, particle_ids, grid_size):
    """Sort positions by Lagrangian ID (x-major ordering for gotetra)."""
    import _tessera as ts
    return at.density.sort_by_lagrangian_id(positions, particle_ids, grid_size)


def compute_asymptotic_density_2d(sorted_positions, box_size, grid_size, output_cells,
                                   subbox=None, scale_factor=1.0):
    """Compute 2D z-projected density field using tessera.

    Args:
        sorted_positions: Particle positions sorted by Lagrangian ID (comoving coordinates)
        box_size: Simulation box size (comoving)
        grid_size: Lagrangian grid size (e.g., 256 for 256^3 particles)
        output_cells: Number of output cells per dimension
        subbox: Subbox parameters (comoving coordinates) or None for full box
        scale_factor: Scale factor 'a' for physical space computation. If != 1.0,
                      positions and box parameters are scaled to physical coordinates
                      before density computation. This effectively computes density
                      in physical space while keeping the output grid in comoving coords.

    Returns:
        2D density array
    """
    import _tessera as ts

    # Scale to physical coordinates if requested
    if scale_factor != 1.0:
        # Scale positions: r_phys = r_com * a
        positions_scaled = sorted_positions * scale_factor
        box_size_scaled = box_size * scale_factor
    else:
        positions_scaled = sorted_positions
        box_size_scaled = box_size

    config = at.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = box_size_scaled
    config.output_cells = output_cells
    config.n_threads = 1  # Single thread to avoid OpenMP issues on macOS
    config.n_samples = 50  # Same as gotetra "Particles = 50"
    config.periodic = True
    config.particle_mass = 1.0

    if subbox is not None:
        config.subbox_enabled = True
        width = subbox['width']
        # Scale subbox parameters to physical if needed
        if scale_factor != 1.0:
            width_scaled = width * scale_factor
        else:
            width_scaled = width

        # Use exact origin from gotetra header if available, otherwise compute from center
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

    # Use Z-axis projection (axis=2)
    result = at.density.compute_tetra_density_2d_projection(positions_scaled, config, 2)
    return result.density


def read_gotetra_reference(gtet_path):
    """Read reference density field from gotetra .gtet file."""
    from tessera.gotetra_compat import read_header, read_grid
    header = read_header(str(gtet_path))
    grid = read_grid(str(gtet_path))
    return grid, header


def compare_density_fields(at_density, gotetra_density):
    """Compare two density fields and compute statistics."""
    # Normalize both fields (convert to overdensity: 1 + delta)
    at_norm = at_density / np.mean(at_density)
    gt_norm = gotetra_density / np.mean(gotetra_density)

    # Flatten for correlation
    at_flat = at_norm.flatten()
    gt_flat = gt_norm.flatten()

    # Compute Pearson correlation
    correlation = np.corrcoef(at_flat, gt_flat)[0, 1]

    # Compute relative differences (only where gotetra > 0)
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
    """Create a 2x3 comparison figure.

    Args:
        extent: [xmin, xmax, ymin, ymax] for coordinate display, or None for pixel indices
        coord_label: Label for coordinate units (e.g., 'cMpc/h' or 'Mpc/h')
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ===== Top Row: Density comparison =====

    # Top Left: tessera density
    vmin = np.log10(np.maximum(at_norm, 0.01).min())
    vmax = np.log10(at_norm.max())
    im1 = axes[0, 0].imshow(np.log10(np.maximum(at_norm, 0.01)), origin='lower',
                            cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0, 0].set_title('tessera')
    plt.colorbar(im1, ax=axes[0, 0], label=r'$\log_{10}(1+\delta)$')
    if extent is not None:
        axes[0, 0].set_xlabel(coord_label)
        axes[0, 0].set_ylabel(coord_label)

    # Top Middle: Gotetra density
    im2 = axes[0, 1].imshow(np.log10(np.maximum(gt_norm, 0.01)), origin='lower',
                            cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0, 1].set_title('Gotetra (Reference)')
    plt.colorbar(im2, ax=axes[0, 1], label=r'$\log_{10}(1+\delta)$')
    if extent is not None:
        axes[0, 1].set_xlabel(coord_label)
        axes[0, 1].set_ylabel(coord_label)

    # Top Right: Relative difference map (no clipping - use percentile-based limits)
    # Use symmetric percentile limits to show the full distribution
    diff_abs_max = np.percentile(np.abs(rel_diff[np.isfinite(rel_diff)]), 99)
    im3 = axes[0, 2].imshow(rel_diff, origin='lower', cmap='RdBu_r',
                            vmin=-diff_abs_max, vmax=diff_abs_max, extent=extent)
    axes[0, 2].set_title('Relative Difference')
    plt.colorbar(im3, ax=axes[0, 2], label='(AT - GT) / GT')
    if extent is not None:
        axes[0, 2].set_xlabel(coord_label)
        axes[0, 2].set_ylabel(coord_label)

    # ===== Bottom Row: Statistical comparisons =====

    # Bottom Left: Overdensity PDF comparison
    bins = np.logspace(-2, 4, 100)  # Extended range for high overdensities
    at_flat = at_norm.flatten()
    gt_flat = gt_norm.flatten()

    axes[1, 0].hist(at_flat[at_flat > 0], bins=bins, alpha=0.7, label='tessera',
                    color='#e74c3c', density=True)
    axes[1, 0].hist(gt_flat[gt_flat > 0], bins=bins, alpha=0.7, label='Gotetra',
                    color='#3498db', density=True)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel(r'$1+\delta$')
    axes[1, 0].set_ylabel('PDF')
    axes[1, 0].set_title('Overdensity PDF')
    axes[1, 0].legend()

    # Bottom Middle: Scatter plot (one-to-one comparison)
    # Subsample for performance
    n_sample = min(10000, len(at_flat))
    idx = np.random.choice(len(at_flat), n_sample, replace=False)

    axes[1, 1].scatter(gt_flat[idx], at_flat[idx], s=1, alpha=0.5, c='#2c3e50')

    # Add 1:1 line
    lims = [min(gt_flat[idx].min(), at_flat[idx].min()),
            max(gt_flat[idx].max(), at_flat[idx].max())]
    axes[1, 1].plot(lims, lims, 'r-', linewidth=2, label='1:1')

    # Compute Spearman correlation for comparison
    spearman_r, _ = scipy_stats.spearmanr(at_flat, gt_flat)

    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel(r'Gotetra $1+\delta$')
    axes[1, 1].set_ylabel(r'tessera $1+\delta$')
    axes[1, 1].set_title(f'Pearson r={stats["correlation"]:.6f}, Spearman r={spearman_r:.4f}')
    axes[1, 1].legend()

    # Bottom Right: Relative difference histogram (use adaptive range)
    rel_diff_flat = rel_diff.flatten()
    valid_diff = rel_diff_flat[np.isfinite(rel_diff_flat) & (gt_flat > 0)]

    # Use percentile-based range to show full distribution
    hist_min = np.percentile(valid_diff, 1)
    hist_max = np.percentile(valid_diff, 99)
    hist_range = (min(hist_min, -0.5), max(hist_max, 0.5))

    axes[1, 2].hist(valid_diff, bins=100, range=hist_range, alpha=0.7,
                    color='#27ae60', density=True)
    axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 2].axvline(x=np.mean(valid_diff), color='blue', linestyle='-', linewidth=2,
                       label=f'Mean: {np.mean(valid_diff):.4f}')
    axes[1, 2].set_xlabel('Relative Difference (AT - GT) / GT')
    axes[1, 2].set_ylabel('PDF')
    axes[1, 2].set_title(f'Diff Distribution (σ={np.std(valid_diff):.4f})')
    axes[1, 2].legend()

    # Overall title
    status = "PASS" if stats['passed'] else "FAIL"
    plt.suptitle(f'{title}\nCorrelation: {stats["correlation"]:.6f} | {status}', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_validation(config):
    """Run a single validation test."""
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

    # Read gotetra reference first to get parameters
    print("Reading gotetra reference...")
    gotetra_density, header = read_gotetra_reference(gotetra_ref)
    output_cells = gotetra_density.shape[0]
    print(f"  Reference shape: {gotetra_density.shape}")
    print(f"  Reference range: [{gotetra_density.min():.4e}, {gotetra_density.max():.4e}]")

    # Extract subbox parameters
    subbox = None
    if subbox_config == 'from_header':
        # Read subbox parameters from gotetra header
        origin = header.loc.origin
        span = header.loc.span
        width = float(span[0])
        center = (origin[0] + width/2, origin[1] + width/2, origin[2] + width/2)
        subbox = {'center': center, 'width': width, 'origin': tuple(origin), 'span': tuple(span)}
        print(f"  Subbox from header:")
        print(f"    Origin: ({origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f})")
        print(f"    Span: ({span[0]:.6f}, {span[1]:.6f}, {span[2]:.6f})")
        print(f"    Pixel width: {header.pw:.8f}")
        print(f"    Using width: {width:.6f}")
    elif subbox_config is not None:
        # Use explicit subbox parameters
        subbox = subbox_config.copy()
        if 'origin' in subbox and 'center' not in subbox:
            # Convert origin to center for reporting
            width = subbox['width']
            origin = subbox['origin']
            subbox['center'] = (origin[0] + width/2, origin[1] + width/2, origin[2] + width/2)
        print(f"  Subbox (explicit):")
        print(f"    Origin: ({subbox.get('origin', 'N/A')})")
        print(f"    Width: {subbox['width']:.6f}")

    # Load snapshot
    print("Loading snapshot...")
    positions, particle_ids, box_size, redshift, scale_factor = load_snapshot(snapshot_path)
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    print(f"  Particles: {n_particles:,}")
    print(f"  Grid: {grid_size}^3")
    print(f"  Box size: {box_size}")
    print(f"  Redshift: {redshift:.4f}")
    print(f"  Scale factor: {scale_factor:.4f}")

    # Sort positions
    print("Sorting positions to Lagrangian order...")
    sorted_positions = sort_positions_lagrangian(positions, particle_ids, grid_size)

    # For full box comparisons, gotetra outputs N+1 pixels with boundary zeros
    # We need to trim the gotetra output and use N output cells for AT
    if subbox is None:
        # Full box: trim gotetra's boundary row/col of zeros
        gotetra_density = gotetra_density[:-1, :-1]
        output_cells = gotetra_density.shape[0]
        print(f"  Trimmed gotetra to {output_cells}x{output_cells} (removed boundary zeros)")

    # Compute tessera density
    print(f"Computing tessera density ({output_cells}x{output_cells})...")
    at_density = compute_asymptotic_density_2d(sorted_positions, box_size, grid_size,
                                                output_cells, subbox=subbox)
    print(f"  AT shape: {at_density.shape}")
    print(f"  AT range: [{at_density.min():.4e}, {at_density.max():.4e}]")

    # Compare
    print("Comparing density fields...")
    stats, at_norm, gt_norm, rel_diff = compare_density_fields(at_density, gotetra_density)

    print(f"\n  Correlation: {stats['correlation']:.6f}")
    print(f"  Mean relative difference: {stats['mean_relative_difference']:.4e}")
    print(f"  Max relative difference: {stats['max_relative_difference']:.4e}")
    print(f"  RESULT: {'PASS' if stats['passed'] else 'FAIL'}")

    # Compute extent for coordinate display
    if subbox is not None:
        # Subbox: use origin and width
        origin = subbox.get('origin', (0, 0, 0))
        width = subbox['width']
        extent = [origin[0], origin[0] + width, origin[1], origin[1] + width]
        coord_label = 'cMpc/h'
    else:
        # Full box
        extent = [0, box_size, 0, box_size]
        coord_label = 'cMpc/h'

    # Create comparison figure
    image_path = SCRIPT_DIR / f"{name}_comparison.png"
    print(f"\nGenerating comparison figure...")
    create_comparison_figure(at_density, gotetra_density, at_norm, gt_norm, rel_diff,
                            description, image_path, stats, extent=extent, coord_label=coord_label)
    print(f"  Saved: {image_path.name}")

    # Prepare result
    result = {
        'name': name,
        'description': description,
        'snapshot': snapshot_name,
        'redshift': redshift,
        'scale_factor': scale_factor,
        'box_size': box_size,
        'output_cells': output_cells,
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

    # Save summary (note: paths are excluded to keep repo clean)
    all_passed = all(r.get('passed', False) for r in all_results.values())
    summary = {
        'validation_date': datetime.now().isoformat(),
        'reference': 'gotetra: github.com/phil-mansfield/gotetra',
        'method': 'Tetrahedron-based phase-space tessellation for density field computation',
        'notes': [
            'Validation compares z-projected density slices',
            'Full box comparisons use 256^3 output grid',
            'Halo comparisons use 10 cMpc/h subbox centered on Group 0 halo',
            'Correlation > 0.999 indicates excellent numerical agreement'
        ],
        'results': all_results,
        'all_tests_passed': all_passed
    }

    summary_path = SCRIPT_DIR / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {SCRIPT_DIR}")
    print("\nSummary:")
    for name, result in all_results.items():
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {name}: {status} (r={result['correlation']:.6f})")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
