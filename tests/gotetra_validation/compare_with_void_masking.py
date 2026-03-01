#!/usr/bin/env python3
"""
Compare tessera vs gotetra density fields for Group 0 halo at snapshots 34 and 74.

This script demonstrates the new particle_counts feature and compares
tessera output against original gotetra reference files.
"""

import sys
import os
from pathlib import Path

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add tessera build to path
TESSERA_BUILD = Path('/Users/bryen/Documents/GitHub/tessera/build')
sys.path.insert(0, str(TESSERA_BUILD))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

# Import tessera
try:
    import _tessera as ts
    # Also import tessera Python utilities
    sys.path.insert(0, str(TESSERA_BUILD / 'tessera'))
    from gotetra_compat import read_header, read_grid
except ImportError as e:
    print(f"Failed to import tessera: {e}")
    sys.exit(1)

# Configuration
SIM_DIR = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output")
GOTETRA_BASE = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly")
OUTPUT_DIR = Path(__file__).parent

# Gotetra reference files for halo comparison
GOTETRA_REFS = {
    34: GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap034' / 'halo_density.gtet',
    74: GOTETRA_BASE / 'halo_comparison_results' / 'gotetra_snap074' / 'halo_density.gtet',
}

# Subbox configurations (from original gotetra runs)
SUBBOX_CONFIGS = {
    34: {
        'origin': (43.17582417582418, 179.77167277167277, 96.31379731379731),
        'width': 10.0,
    },
    74: {
        'origin': (147.52755737304688, 146.6825408935547, 75.38079071044922),
        'width': 10.080586080586081,
    },
}


def read_gotetra_gtet(filepath):
    """Read a gotetra .gtet density file using tessera's gotetra_compat."""
    header = read_header(str(filepath))
    grid = read_grid(str(filepath))

    # Get dimensions from header
    output_cells = grid.shape[0]  # 2D projection grid
    origin = header.loc.origin
    span = header.loc.span

    return grid, origin, span, output_cells, header


def load_snapshot_particles(snap_dir: Path, snap_id: int):
    """Load particle positions and IDs from a snapshot."""
    snap_file = snap_dir / f"snapshot_{snap_id:03d}.hdf5"

    with h5py.File(snap_file, 'r') as hf:
        box_size = hf['Header'].attrs['BoxSize']
        scale_factor = hf['Header'].attrs['Time']
        positions = hf['PartType1/Coordinates'][:]
        ids = hf['PartType1/ParticleIDs'][:]

    return positions, ids, box_size, scale_factor


def compute_tessera_density(positions, ids, box_size, grid_size, subbox_config, output_cells):
    """Compute density using tessera with particle counts."""

    # Sort to Lagrangian order
    sorted_positions = ts.density.sort_by_lagrangian_id(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(ids, dtype=np.int64),
        grid_size,
        1  # GADGET uses 1-indexed IDs
    )

    # Configure tessera
    config = ts.density.TetraDensityConfig()
    config.box_size = box_size
    config.lagrangian_grid_size = grid_size
    config.output_cells = output_cells
    config.n_samples = 50  # Match gotetra default
    config.n_threads = 1
    config.periodic = True
    config.particle_mass = 1.0

    # Enable subbox mode
    config.subbox_enabled = True
    config.subbox_origin = tuple(subbox_config['origin'])
    width = subbox_config['width']
    config.subbox_width = (width, width, width)

    result = ts.density.compute_tetra_density_2d_projection(sorted_positions, config, 2)

    density = np.array(result.density).reshape(output_cells, output_cells)
    counts = np.array(result.particle_counts).reshape(output_cells, output_cells)

    return density, counts


def apply_void_masking(density, counts, threshold=2, floor_percentile=5):
    """Apply void masking based on particle counts."""
    masked_density = density.copy()
    sparse_mask = counts < threshold

    if np.any(~sparse_mask):
        floor_value = np.percentile(density[~sparse_mask], floor_percentile)
    else:
        floor_value = np.percentile(density, floor_percentile)

    masked_density[sparse_mask] = floor_value
    return masked_density, sparse_mask


def compare_fields(tessera_density, gotetra_density):
    """Compute comparison statistics."""
    # Normalize by mean
    ts_norm = tessera_density / np.mean(tessera_density)
    gt_norm = gotetra_density / np.mean(gotetra_density)

    # Correlation
    correlation = np.corrcoef(ts_norm.flatten(), gt_norm.flatten())[0, 1]

    # Relative difference
    mask = gt_norm > 0
    rel_diff = np.zeros_like(ts_norm)
    rel_diff[mask] = (ts_norm[mask] - gt_norm[mask]) / gt_norm[mask]

    mean_rel_diff = np.mean(np.abs(rel_diff[mask])) if np.any(mask) else 0.0

    return {
        'correlation': correlation,
        'mean_relative_difference': mean_rel_diff,
    }


def plot_comparison(tessera_orig, tessera_masked, gotetra, counts, sparse_mask,
                   extent, snap_id, scale_factor, stats_orig, stats_masked, output_file):
    """Create a 3x2 comparison figure."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Normalize for plotting
    ts_mean = tessera_orig.mean()
    gt_mean = gotetra.mean()

    vmin, vmax = 1e-1, 1e4

    # Row 1: Density fields
    # Tessera original
    ax = axes[0, 0]
    plot_data = tessera_orig / ts_mean
    plot_data = np.maximum(plot_data, 1e-3)
    im = ax.imshow(plot_data.T, extent=extent, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    ax.set_title(f'Tessera (Original)\nCorr={stats_orig["correlation"]:.6f}', fontsize=12)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$')

    # Tessera void-masked
    ax = axes[0, 1]
    plot_data = tessera_masked / ts_mean
    plot_data = np.maximum(plot_data, 1e-3)
    im = ax.imshow(plot_data.T, extent=extent, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    ax.set_title(f'Tessera (Void-Masked)\nCorr={stats_masked["correlation"]:.6f}', fontsize=12)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$')

    # Gotetra reference
    ax = axes[0, 2]
    plot_data = gotetra / gt_mean
    plot_data = np.maximum(plot_data, 1e-3)
    im = ax.imshow(plot_data.T, extent=extent, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    ax.set_title('Gotetra (Reference)', fontsize=12)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$')

    # Row 2: Diagnostics
    # Particle counts
    ax = axes[1, 0]
    im = ax.imshow(counts.T, extent=extent, origin='lower', cmap='plasma')
    ax.set_title(f'Sample Counts\n(from tessera particle_counts)', fontsize=12)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label='Samples per pixel')

    # Void mask
    ax = axes[1, 1]
    n_masked = sparse_mask.sum()
    pct_masked = 100 * n_masked / sparse_mask.size
    im = ax.imshow(sparse_mask.astype(float).T, extent=extent, origin='lower',
                   cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Void Mask (counts < 2)\n{n_masked}/{sparse_mask.size} pixels ({pct_masked:.1f}%)', fontsize=12)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label='Masked')

    # Relative difference histogram
    ax = axes[1, 2]
    ts_norm = tessera_masked / ts_mean
    gt_norm = gotetra / gt_mean
    mask = gt_norm > 0.01
    rel_diff = (ts_norm[mask] - gt_norm[mask]) / gt_norm[mask]

    ax.hist(rel_diff.flatten(), bins=100, range=(-0.5, 0.5), alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Relative Difference')
    ax.set_ylabel('Count')
    ax.set_title(f'(Tessera - Gotetra) / Gotetra\nMean |diff| = {stats_masked["mean_relative_difference"]:.4f}', fontsize=12)
    ax.set_xlim(-0.5, 0.5)

    plt.suptitle(f'Snapshot {snap_id} (a={scale_factor:.2f}): Tessera vs Gotetra with Void Masking',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    print("=" * 70)
    print("Tessera vs Gotetra Comparison with Void Masking")
    print("=" * 70)
    print()

    for snap_id in [34, 74]:
        print(f"\n{'='*70}")
        print(f"Processing Snapshot {snap_id}")
        print(f"{'='*70}")

        # Check reference file exists
        gotetra_ref = GOTETRA_REFS[snap_id]
        if not gotetra_ref.exists():
            print(f"  WARNING: Gotetra reference not found: {gotetra_ref}")
            continue

        # Read gotetra reference
        print("Reading gotetra reference...")
        gotetra_density, origin, span, output_cells, gt_header = read_gotetra_gtet(gotetra_ref)
        print(f"  Shape: {gotetra_density.shape}")
        print(f"  Origin: {origin}")
        print(f"  Span: {span}")

        # Load particles
        print("Loading particles...")
        positions, ids, box_size, scale_factor = load_snapshot_particles(SIM_DIR, snap_id)
        grid_size = int(round(len(positions) ** (1/3)))
        print(f"  Particles: {len(positions):,} (grid_size={grid_size})")
        print(f"  Box size: {box_size}")
        print(f"  Scale factor: {scale_factor}")

        # Compute tessera density
        print("Computing tessera density...")
        subbox_config = SUBBOX_CONFIGS[snap_id]
        tessera_density, counts = compute_tessera_density(
            positions, ids, box_size, grid_size, subbox_config, output_cells
        )
        print(f"  Tessera range: [{tessera_density.min():.2e}, {tessera_density.max():.2e}]")
        print(f"  Gotetra range: [{gotetra_density.min():.2e}, {gotetra_density.max():.2e}]")
        print(f"  Zero-count pixels: {(counts == 0).sum()}")
        print(f"  Low-count pixels (< 2): {(counts < 2).sum()}")

        # Apply void masking
        print("Applying void masking...")
        tessera_masked, sparse_mask = apply_void_masking(tessera_density, counts)
        print(f"  Masked {sparse_mask.sum()} pixels ({100*sparse_mask.sum()/sparse_mask.size:.1f}%)")

        # Compare
        print("Computing statistics...")
        stats_orig = compare_fields(tessera_density, gotetra_density)
        stats_masked = compare_fields(tessera_masked, gotetra_density)

        print(f"\n  Original tessera:")
        print(f"    Correlation: {stats_orig['correlation']:.6f}")
        print(f"    Mean relative diff: {stats_orig['mean_relative_difference']:.4f}")

        print(f"\n  Void-masked tessera:")
        print(f"    Correlation: {stats_masked['correlation']:.6f}")
        print(f"    Mean relative diff: {stats_masked['mean_relative_difference']:.4f}")

        # Plot comparison
        extent = [
            subbox_config['origin'][0],
            subbox_config['origin'][0] + subbox_config['width'],
            subbox_config['origin'][1],
            subbox_config['origin'][1] + subbox_config['width'],
        ]

        output_file = OUTPUT_DIR / f"snap{snap_id:03d}_gotetra_comparison_with_masking.png"
        plot_comparison(
            tessera_density, tessera_masked, gotetra_density, counts, sparse_mask,
            extent, snap_id, scale_factor, stats_orig, stats_masked, output_file
        )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
