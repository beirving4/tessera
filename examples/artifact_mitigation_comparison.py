#!/usr/bin/env python3
"""
Artifact Mitigation Comparison: Tessera vs Gotetra

This script compares all four artifact mitigation approaches on the Group 0 halo
at snapshot 74 (a=100, the "island universe" epoch) against the original gotetra
reference implementation.

The four approaches are:
1. Original tessellation (no mitigation)
2. Gaussian smoothing (post-processing)
3. Adaptive smoothing (count-weighted)
4. SPH-based particle density (bypasses tessellation)

Plus the gotetra reference for comparison.

Usage:
    python artifact_mitigation_comparison.py

Output:
    artifact_mitigation_comparison_snap074.png
"""

import sys
import os
from pathlib import Path

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add tessera paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))
sys.path.insert(0, str(REPO_ROOT / 'python'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

# Import tessera
try:
    import _tessera as ts
    from tessera.utils import (
        gaussian_smooth_density,
        adaptive_smooth_density,
        compute_particle_density_sph,
        compute_particle_density_cic,
    )
    from tessera.gotetra_compat import read_header, read_grid
except ImportError as e:
    print(f"Failed to import tessera: {e}")
    print("Make sure tessera is built: cd build && cmake .. && make -j8")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Simulation paths (update these for your system)
SIM_DIR = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output")
GOTETRA_BASE = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly")

# Target snapshot (can be overridden via command line)
SNAP_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 74  # Default: a=100, island universe epoch

# Subbox configurations for different snapshots (10 cMpc/h centered on Group 0)
# These are centered on the actual halo position from FOF catalogs
SUBBOX_CONFIGS = {
    34: {
        'origin': (43.11, 179.74, 96.28),  # Centered on Group 0 at [48.16, 184.79, 101.33]
        'width': 10.1,
    },
    74: {
        'origin': (147.48, 146.63, 75.33),  # Centered on Group 0 at [152.53, 151.68, 80.38]
        'width': 10.1,
    },
}

# Select config for current snapshot
SUBBOX_CONFIG = SUBBOX_CONFIGS[SNAP_ID]

# Gotetra reference paths for different snapshots (halo-centered versions)
GOTETRA_REFS = {
    34: GOTETRA_BASE / "gotetra_validation_refs/snap034_halo_centered/halo.gtet",
    74: GOTETRA_BASE / "gotetra_validation_refs/snap074_halo_centered/halo.gtet",
}
GOTETRA_REF = GOTETRA_REFS[SNAP_ID]

# Output
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = OUTPUT_DIR / f"artifact_mitigation_comparison_snap{SNAP_ID:03d}.png"


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_snapshot_particles(snap_dir: Path, snap_id: int):
    """Load particle positions and IDs from a snapshot."""
    snap_file = snap_dir / f"snapshot_{snap_id:03d}.hdf5"

    if not snap_file.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_file}")

    with h5py.File(snap_file, 'r') as hf:
        box_size = hf['Header'].attrs['BoxSize']
        scale_factor = hf['Header'].attrs['Time']
        positions = hf['PartType1/Coordinates'][:]
        ids = hf['PartType1/ParticleIDs'][:]

    return positions, ids, box_size, scale_factor


def load_gotetra_reference(gtet_path: Path):
    """Load gotetra reference density field."""
    if not gtet_path.exists():
        raise FileNotFoundError(f"Gotetra reference not found: {gtet_path}")

    header = read_header(str(gtet_path))
    grid = read_grid(str(gtet_path))

    return grid, header


# ============================================================================
# Density Computation Functions
# ============================================================================

def compute_tessellation_density(positions, ids, box_size, grid_size, subbox_config, output_cells):
    """Compute density using standard tessellation."""

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
    config.n_samples = 50
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

    return density, counts, sorted_positions


def compute_sph_density(positions, box_size, subbox_config, output_cells):
    """Compute density using SPH (no tessellation)."""

    # Filter particles to subbox region (with some padding)
    origin = np.array(subbox_config['origin'])
    width = subbox_config['width']
    padding = width * 0.1  # 10% padding for kernel support

    mask = (
        (positions[:, 0] >= origin[0] - padding) &
        (positions[:, 0] < origin[0] + width + padding) &
        (positions[:, 1] >= origin[1] - padding) &
        (positions[:, 1] < origin[1] + width + padding) &
        (positions[:, 2] >= origin[2] - padding) &
        (positions[:, 2] < origin[2] + width + padding)
    )

    local_positions = positions[mask].copy()

    # Shift to local coordinates
    local_positions[:, 0] -= origin[0]
    local_positions[:, 1] -= origin[1]
    local_positions[:, 2] -= origin[2]

    result = compute_particle_density_sph(
        local_positions,
        box_size=width,
        output_cells=output_cells,
        kernel='cubic_spline',
        n_neighbors=32,
        projection_axis=2,
        periodic=False  # Not periodic in subbox
    )

    return result['density']


def compute_cic_density(positions, box_size, subbox_config, output_cells):
    """Compute density using CIC (simple particle assignment)."""

    origin = np.array(subbox_config['origin'])
    width = subbox_config['width']

    # Filter particles to subbox
    mask = (
        (positions[:, 0] >= origin[0]) &
        (positions[:, 0] < origin[0] + width) &
        (positions[:, 1] >= origin[1]) &
        (positions[:, 1] < origin[1] + width) &
        (positions[:, 2] >= origin[2]) &
        (positions[:, 2] < origin[2] + width)
    )

    local_positions = positions[mask].copy()
    local_positions[:, 0] -= origin[0]
    local_positions[:, 1] -= origin[1]
    local_positions[:, 2] -= origin[2]

    result = compute_particle_density_cic(
        local_positions,
        box_size=width,
        output_cells=output_cells,
        projection_axis=2,
        periodic=False
    )

    return result['density']


# ============================================================================
# Comparison Statistics
# ============================================================================

def compute_statistics(density, reference, name=""):
    """Compute comparison statistics against reference."""
    # Normalize by mean
    d_norm = density / np.mean(density)
    r_norm = reference / np.mean(reference)

    # Correlation
    correlation = np.corrcoef(d_norm.flatten(), r_norm.flatten())[0, 1]

    # Relative difference
    mask = r_norm > 0.01
    rel_diff = np.abs(d_norm[mask] - r_norm[mask]) / r_norm[mask]
    mean_rel_diff = np.mean(rel_diff)

    return {
        'name': name,
        'correlation': correlation,
        'mean_relative_difference': mean_rel_diff,
    }


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_figure(results, gotetra_density, extent, scale_factor, output_file,
                             positions=None, subbox_config=None):
    """Create a comprehensive comparison figure."""

    fig = plt.figure(figsize=(24, 16))

    # Use gridspec for better control - now 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)

    # Normalize all densities by their mean for visualization
    vmin, vmax = 1e-1, 1e4

    # Use original tessellation as reference if no gotetra
    reference = gotetra_density if gotetra_density is not None else results['original']['density']
    has_gotetra = gotetra_density is not None

    def plot_density(ax, density, title, stats=None):
        """Helper to plot a density panel."""
        d_norm = density / np.mean(density)
        d_norm = np.maximum(d_norm, 1e-3)

        im = ax.imshow(d_norm.T, extent=extent, origin='lower',
                       norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')

        title_text = title
        if stats:
            title_text += f"\nCorr={stats['correlation']:.4f}"
        ax.set_title(title_text, fontsize=11)
        ax.set_xlabel('x [cMpc/h]')
        ax.set_ylabel('y [cMpc/h]')

        return im

    # Row 1: Raw data and original methods
    # Particle positions (scatter plot)
    ax = fig.add_subplot(gs[0, 0])
    if positions is not None and subbox_config is not None:
        origin = np.array(subbox_config['origin'])
        width = subbox_config['width']

        # Filter particles in subbox (with z-slice for visibility)
        z_center = origin[2] + width / 2
        z_thickness = width * 0.1  # 10% slice thickness

        mask = (
            (positions[:, 0] >= origin[0]) &
            (positions[:, 0] < origin[0] + width) &
            (positions[:, 1] >= origin[1]) &
            (positions[:, 1] < origin[1] + width) &
            (positions[:, 2] >= z_center - z_thickness/2) &
            (positions[:, 2] < z_center + z_thickness/2)
        )

        local_pos = positions[mask]
        n_particles_shown = len(local_pos)

        # Plot particles
        ax.scatter(local_pos[:, 0], local_pos[:, 1], s=0.1, c='white', alpha=0.5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor('black')
        ax.set_title(f'Raw Particles (z-slice)\n{n_particles_shown:,} particles in 10% slice', fontsize=11)
        ax.set_xlabel('x [cMpc/h]')
        ax.set_ylabel('y [cMpc/h]')
    else:
        ax.text(0.5, 0.5, 'Particle data\nnot available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Raw Particles', fontsize=11)

    # Gotetra reference (or placeholder)
    ax = fig.add_subplot(gs[0, 1])
    if has_gotetra:
        im = plot_density(ax, gotetra_density, 'Gotetra (Reference)')
        plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'Gotetra reference\nnot available\n(different subbox)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Gotetra (Reference)', fontsize=11)
        ax.axis('off')

    # Original tessellation
    ax = fig.add_subplot(gs[0, 2])
    im = plot_density(ax, results['original']['density'], '1. Original Tessellation')
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # Particle counts
    ax = fig.add_subplot(gs[0, 3])
    counts = results['original']['counts']
    im = ax.imshow(counts.T, extent=extent, origin='lower', cmap='plasma')
    ax.set_title(f'Sample Counts\nZero: {(counts==0).sum()}, Low(<2): {(counts<2).sum()}', fontsize=11)
    ax.set_xlabel('x [cMpc/h]')
    ax.set_ylabel('y [cMpc/h]')
    plt.colorbar(im, ax=ax, label='Samples per pixel', shrink=0.8)

    # Row 2: Post-processing methods
    # Gaussian smoothing
    ax = fig.add_subplot(gs[1, 0])
    stats = compute_statistics(results['gaussian']['density'], reference, 'Gaussian')
    im = plot_density(ax, results['gaussian']['density'], '2. Gaussian Smoothing (σ=1.5)', stats if has_gotetra else None)
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # Adaptive smoothing
    ax = fig.add_subplot(gs[1, 1])
    stats = compute_statistics(results['adaptive']['density'], reference, 'Adaptive')
    im = plot_density(ax, results['adaptive']['density'], '3. Adaptive Smoothing', stats if has_gotetra else None)
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # Void masking (from earlier implementation)
    ax = fig.add_subplot(gs[1, 2])
    stats = compute_statistics(results['void_masked']['density'], reference, 'Void Masked')
    im = plot_density(ax, results['void_masked']['density'], 'Void Masking (counts<2)', stats if has_gotetra else None)
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # Bilateral smoothing
    ax = fig.add_subplot(gs[1, 3])
    if 'bilateral' in results:
        stats = compute_statistics(results['bilateral']['density'], reference, 'Bilateral')
        im = plot_density(ax, results['bilateral']['density'], 'Bilateral Filter', stats if has_gotetra else None)
        plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)
    else:
        ax.axis('off')

    # Row 3: Alternative methods
    # SPH density
    ax = fig.add_subplot(gs[2, 0])
    stats = compute_statistics(results['sph']['density'], reference, 'SPH')
    im = plot_density(ax, results['sph']['density'], '4. SPH Density (n=32)', stats if has_gotetra else None)
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # CIC density
    ax = fig.add_subplot(gs[2, 1])
    stats = compute_statistics(results['cic']['density'], reference, 'CIC')
    im = plot_density(ax, results['cic']['density'], 'CIC Density (simple)', stats if has_gotetra else None)
    plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)

    # NGP (Nearest Grid Point) - even simpler
    ax = fig.add_subplot(gs[2, 2])
    if 'ngp' in results:
        stats = compute_statistics(results['ngp']['density'], reference, 'NGP')
        im = plot_density(ax, results['ngp']['density'], 'NGP Density (nearest)', stats if has_gotetra else None)
        plt.colorbar(im, ax=ax, label=r'$\rho / \bar{\rho}$', shrink=0.8)
    else:
        ax.axis('off')

    # Summary statistics
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')

    # Create summary table
    summary_text = f"Scale factor: a = {scale_factor:.1f}\n"
    summary_text += f"Snapshot: {SNAP_ID}\n"
    summary_text += f"Subbox: {SUBBOX_CONFIG['width']:.2f} cMpc/h\n"
    summary_text += f"Centered on Group 0 halo\n"
    summary_text += "\n" + "="*30 + "\n"

    if has_gotetra:
        methods = ['Original', 'Gaussian', 'Adaptive', 'Void Masked', 'SPH', 'CIC']
        correlations = [
            compute_statistics(results['original']['density'], reference)['correlation'],
            compute_statistics(results['gaussian']['density'], reference)['correlation'],
            compute_statistics(results['adaptive']['density'], reference)['correlation'],
            compute_statistics(results['void_masked']['density'], reference)['correlation'],
            compute_statistics(results['sph']['density'], reference)['correlation'],
            compute_statistics(results['cic']['density'], reference)['correlation'],
        ]

        summary_text += "Correlation with Gotetra:\n"
        for method, corr in zip(methods, correlations):
            bar = "█" * int(max(0, corr) * 20)
            summary_text += f"{method:12s}: {corr:.4f} {bar}\n"
    else:
        summary_text += "Comparison of methods\n"
        summary_text += "(No gotetra reference\nfor this subbox)\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    title_suffix = "Tessera Density Methods" if not has_gotetra else "Tessera vs Gotetra Reference"
    plt.suptitle(f'Artifact Mitigation Comparison: Group 0 Halo at a={scale_factor:.0f}\n{title_suffix}',
                 fontsize=14, y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison figure: {output_file}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Artifact Mitigation Comparison")
    print("=" * 70)
    print(f"Snapshot: {SNAP_ID}")
    print(f"Simulation: {SIM_DIR}")
    print(f"Subbox centered on Group 0 halo")
    print()

    # Load gotetra reference (halo-centered version)
    gotetra_density = None
    output_cells = 101  # Default grid resolution

    if GOTETRA_REF.exists():
        try:
            print(f"Loading gotetra reference: {GOTETRA_REF}")
            gotetra_density, gt_header = load_gotetra_reference(GOTETRA_REF)
            output_cells = gotetra_density.shape[0]
            print(f"  Shape: {gotetra_density.shape}")
            print(f"  Range: [{gotetra_density.min():.2e}, {gotetra_density.max():.2e}]")
        except Exception as e:
            print(f"  Warning: Could not load gotetra reference: {e}")
            gotetra_density = None
    else:
        print(f"Gotetra reference not found: {GOTETRA_REF}")
        print(f"  Using default grid resolution: {output_cells}x{output_cells}")

    # Load particles
    print("\nLoading particles...")
    positions, ids, box_size, scale_factor = load_snapshot_particles(SIM_DIR, SNAP_ID)
    grid_size = int(round(len(positions) ** (1/3)))
    print(f"  Particles: {len(positions):,} (grid_size={grid_size})")
    print(f"  Box size: {box_size}")
    print(f"  Scale factor: {scale_factor}")

    # Store all results
    results = {}

    # 1. Original tessellation
    print("\n1. Computing original tessellation...")
    density, counts, sorted_positions = compute_tessellation_density(
        positions, ids, box_size, grid_size, SUBBOX_CONFIG, output_cells
    )
    results['original'] = {'density': density, 'counts': counts}
    print(f"   Range: [{density.min():.2e}, {density.max():.2e}]")
    print(f"   Zero-count pixels: {(counts == 0).sum()}")

    # 2. Gaussian smoothing
    print("\n2. Applying Gaussian smoothing (σ=1.5)...")
    smoothed = gaussian_smooth_density(density, sigma=1.5, particle_counts=counts)
    results['gaussian'] = {'density': smoothed}
    print(f"   Range: [{smoothed.min():.2e}, {smoothed.max():.2e}]")

    # 3. Adaptive smoothing
    print("\n3. Applying adaptive smoothing...")
    adaptive = adaptive_smooth_density(density, counts, sigma_min=0.5, sigma_max=3.0)
    results['adaptive'] = {'density': adaptive}
    print(f"   Range: [{adaptive.min():.2e}, {adaptive.max():.2e}]")

    # 4. Void masking (simple threshold)
    print("\n4. Applying void masking (counts < 2)...")
    void_masked = density.copy()
    sparse_mask = counts < 2
    if np.any(~sparse_mask):
        floor_value = np.percentile(density[~sparse_mask], 5)
    else:
        floor_value = np.percentile(density, 5)
    void_masked[sparse_mask] = floor_value
    results['void_masked'] = {'density': void_masked}
    print(f"   Masked {sparse_mask.sum()} pixels ({100*sparse_mask.sum()/sparse_mask.size:.1f}%)")

    # 5. SPH density
    print("\n5. Computing SPH density (n_neighbors=32)...")
    sph_density = compute_sph_density(positions, box_size, SUBBOX_CONFIG, output_cells)
    results['sph'] = {'density': sph_density}
    print(f"   Range: [{sph_density.min():.2e}, {sph_density.max():.2e}]")

    # 6. CIC density
    print("\n6. Computing CIC density...")
    cic_density = compute_cic_density(positions, box_size, SUBBOX_CONFIG, output_cells)
    results['cic'] = {'density': cic_density}
    print(f"   Range: [{cic_density.min():.2e}, {cic_density.max():.2e}]")

    # Compute extent for plotting
    origin = SUBBOX_CONFIG['origin']
    width = SUBBOX_CONFIG['width']
    extent = [origin[0], origin[0] + width, origin[1], origin[1] + width]

    # Create comparison figure
    print("\nCreating comparison figure...")
    create_comparison_figure(results, gotetra_density, extent, scale_factor, OUTPUT_FILE,
                             positions=positions, subbox_config=SUBBOX_CONFIG)

    # Print summary statistics
    print("\n" + "=" * 70)
    if gotetra_density is not None:
        print("Summary: Correlation with Gotetra Reference")
        print("(Note: Low correlation expected if subbox positions differ)")
    else:
        print("Summary: Method Comparison (no gotetra reference)")
    print("=" * 70)

    methods = [
        ('Original Tessellation', 'original'),
        ('Gaussian Smoothing', 'gaussian'),
        ('Adaptive Smoothing', 'adaptive'),
        ('Void Masking', 'void_masked'),
        ('SPH Density', 'sph'),
        ('CIC Density', 'cic'),
    ]

    # Use original tessellation as reference if no gotetra
    reference = gotetra_density if gotetra_density is not None else results['original']['density']
    ref_name = "Gotetra" if gotetra_density is not None else "Original Tessellation"

    for name, key in methods:
        if gotetra_density is None and key == 'original':
            print(f"  {name:22s}: (reference)")
            continue
        stats = compute_statistics(results[key]['density'], reference)
        bar = "█" * int(max(0, stats['correlation']) * 20)
        print(f"  {name:22s}: {stats['correlation']:.6f} {bar}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
