#!/usr/bin/env python3
"""
Example: ORIGAMI Morphology Classification

This script demonstrates how to:
1. Load a GADGET-4 snapshot
2. Compute ORIGAMI morphology classification (void/wall/filament/halo)
3. Compute density field and sample at particle positions
4. Generate conditional PDFs P(1+delta | morphology class)
5. Save results to HDF5 and create visualizations

Reference: Falck, Neyrinck, & Szalay 2012, ApJ 754, 126

Usage:
    python origami_morphology.py snapshot_034.hdf5 -o origami_snap034.h5 --plot origami_snap034.png
"""

import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Try to find the tessera module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

# Add parent directory to path to import utils
sys.path.insert(0, str(_script_dir.parent))

# Import tessera
try:
    import _tessera as ts
    HAS_AT = True
except ImportError:
    try:
        import tessera as ts
        HAS_AT = True
    except ImportError:
        HAS_AT = False
        print("Error: tessera not found. Build the project first.")
        sys.exit(1)

# Import utils
from tessera.utils import infer_grid_size
from tessera.utils import load_snapshot as utils_load_snapshot


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot.

    Uses the utils load_snapshot function which handles multi-file snapshots
    via the C++ reader. For ORIGAMI, we need particle IDs for Lagrangian sorting.
    """
    print(f"Loading snapshot from {snapshot_path}...")

    # Use utils load_snapshot (handles single/multi-file via C++ reader)
    positions, particle_ids, header = utils_load_snapshot(
        snapshot_path, particle_type=particle_type, read_ids=True
    )

    box_size = header.box_size
    redshift = header.redshift
    time = header.time
    particle_mass = header.mass_table[particle_type]

    print(f"  Box size: {box_size}")
    print(f"  Redshift: {redshift:.4f}")
    print(f"  Scale factor: {time:.4f}")
    print(f"  Total particles: {len(positions):,}")

    return positions, particle_ids, box_size, redshift, time, particle_mass


def run_origami_pipeline(positions, particle_ids, box_size, output_cells=128,
                          n_samples=100, n_threads=0):
    """Run unified ORIGAMI pipeline with density and grid outputs.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions.
    particle_ids : ndarray, shape (N,)
        Particle IDs (1-indexed Lagrangian grid positions).
    box_size : float
        Simulation box size.
    output_cells : int
        Grid resolution for density and morphology grids.
    n_samples : int
        Monte Carlo samples for density computation.
    n_threads : int
        Number of OpenMP threads (0 = auto).

    Returns
    -------
    result : PipelineResult
        Complete pipeline result with morphology, density, and grid outputs.
    density_3d : ndarray
        3D density field.
    grid_size : int
        Lagrangian grid size.
    """
    print("\nRunning unified ORIGAMI pipeline...")

    # Determine grid size from particle count using utils
    grid_size = infer_grid_size(len(positions))

    print(f"  Lagrangian grid: {grid_size}^3")

    # Validate particle IDs
    print("  Validating particle IDs...")
    min_id, max_id = particle_ids.min(), particle_ids.max()
    unique_ids = len(np.unique(particle_ids))
    print(f"    ID range: [{min_id:,}, {max_id:,}]")
    print(f"    Unique IDs: {unique_ids:,}")

    if min_id != 1 or max_id != n_particles:
        print(f"    WARNING: IDs should range from 1 to {n_particles:,}")

    if unique_ids != n_particles:
        raise ValueError(f"Found {unique_ids:,} unique IDs for {n_particles:,} particles. "
                         "IDs must be unique for ORIGAMI.")

    # Configure unified pipeline
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = output_cells
    config.density_n_samples = n_samples
    config.sample_density_at_particles = True
    config.grid_cells = output_cells
    config.n_threads = n_threads
    config.n_split = 1  # Single domain to avoid OpenMP issues on macOS

    # Run pipeline
    print("  Running pipeline (ID detection, sorting, morphology, density, grid)...")
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"  Detected ID ordering: {ts.origami.id_ordering_name(result.detected_ordering)}")
    print(f"  Pipeline completed in {result.total_time_ms:.1f} ms")

    print(f"\n  Morphology Results:")
    print(f"    Void:     {result.n_void:>10,} particles ({result.f_void:>6.2%})")
    print(f"    Wall:     {result.n_wall:>10,} particles ({result.f_wall:>6.2%})")
    print(f"    Filament: {result.n_filament:>10,} particles ({result.f_filament:>6.2%})")
    print(f"    Halo:     {result.n_halo:>10,} particles ({result.f_halo:>6.2%})")

    # Get density as 3D array
    density_3d = np.array(result.density_3d).reshape(
        output_cells, output_cells, output_cells
    )
    print(f"\n  Density field ({output_cells}^3):")
    print(f"    Range: [{density_3d.min():.4e}, {density_3d.max():.4e}]")
    print(f"    Mean: {result.mean_density:.4e}")

    print(f"\n  Grid outputs ({output_cells}^3):")
    print(f"    Halo fraction range: [{np.array(result.halo_fraction_grid).min():.3f}, "
          f"{np.array(result.halo_fraction_grid).max():.3f}]")

    return result, density_3d, grid_size


def save_results(output_path, result, density_3d, positions, box_size, redshift,
                 scale_factor, grid_size, particle_mass):
    """Save results to HDF5."""
    print(f"\nSaving results to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Header
        hdr = f.create_group('Header')
        hdr.attrs['box_size'] = box_size
        hdr.attrs['redshift'] = redshift
        hdr.attrs['scale_factor'] = scale_factor
        hdr.attrs['n_particles'] = len(positions)
        hdr.attrs['lagrangian_grid_size'] = grid_size
        hdr.attrs['particle_mass'] = particle_mass

        # Linear regime detection
        # In the linear regime (no shell-crossing), ORIGAMI classifies all
        # particles as voids, so the classification is not physically meaningful.
        hdr.attrs['is_linear_regime'] = result.is_linear_regime
        hdr.attrs['linear_regime_threshold'] = result.linear_regime_threshold

        # Per-particle data
        per_part = f.create_group('PerParticle')
        per_part.create_dataset('morphology', data=np.array(result.morphology),
                                 compression='gzip')
        if len(result.particle_density) > 0:
            per_part.create_dataset('density', data=np.array(result.particle_density),
                                     compression='gzip')

        # Grid data
        if len(result.morphology_grid) > 0:
            grid = f.create_group('Grid')
            grid.attrs['cells'] = result.morph_grid_cells
            grid.create_dataset('morphology', data=np.array(result.morphology_grid),
                                 compression='gzip')
            grid.create_dataset('void_fraction', data=np.array(result.void_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('wall_fraction', data=np.array(result.wall_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('filament_fraction', data=np.array(result.filament_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('halo_fraction', data=np.array(result.halo_fraction_grid),
                                 compression='gzip')

        # Density field
        if density_3d is not None:
            f.create_dataset('density_3d', data=density_3d, compression='gzip')

        # Statistics
        stats = f.create_group('Statistics')
        stats.create_dataset('counts', data=np.array(result.counts))
        stats.create_dataset('mass_fractions', data=np.array(result.fractions))

        # Compute mean density per class
        if len(result.particle_density) > 0:
            morph = np.array(result.morphology)
            density = np.array(result.particle_density)
            mean_densities = np.array([
                density[morph == i].mean() if np.sum(morph == i) > 0 else 0.0
                for i in range(4)
            ])
            stats.create_dataset('mean_density_per_class', data=mean_densities)

    print("  Saved successfully")


def create_visualization(output_path, result, density_3d, box_size):
    """Create visualization of ORIGAMI results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    print(f"\nCreating visualization...")

    fig = plt.figure(figsize=(16, 12))

    # Get middle slice index
    mid = density_3d.shape[0] // 2

    # 1. Density slice
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(density_3d[mid, :, :], origin='lower', cmap='magma',
                      norm=LogNorm(), extent=[0, box_size, 0, box_size])
    ax1.set_title('Density (z-slice)')
    ax1.set_xlabel('x [Mpc/h]')
    ax1.set_ylabel('y [Mpc/h]')
    plt.colorbar(im1, ax=ax1, label='Density')

    # 2. Morphology grid slice
    ax2 = fig.add_subplot(2, 3, 2)
    morph_grid = np.array(result.morphology_grid)
    im2 = ax2.imshow(morph_grid[mid, :, :], origin='lower', cmap='viridis',
                      vmin=0, vmax=3, extent=[0, box_size, 0, box_size])
    ax2.set_title('Dominant Morphology (z-slice)')
    ax2.set_xlabel('x [Mpc/h]')
    ax2.set_ylabel('y [Mpc/h]')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
    cbar2.set_ticklabels(['Void', 'Wall', 'Filament', 'Halo'])

    # 3. Halo fraction slice
    ax3 = fig.add_subplot(2, 3, 3)
    halo_frac = np.array(result.halo_fraction_grid)
    im3 = ax3.imshow(halo_frac[mid, :, :], origin='lower', cmap='Reds',
                      vmin=0, vmax=1, extent=[0, box_size, 0, box_size])
    ax3.set_title('Halo Fraction (z-slice)')
    ax3.set_xlabel('x [Mpc/h]')
    ax3.set_ylabel('y [Mpc/h]')
    plt.colorbar(im3, ax=ax3, label='Halo fraction')

    # 4. Mass fractions bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    classes = ['Void', 'Wall', 'Filament', 'Halo']
    fractions = [result.f_void, result.f_wall, result.f_filament, result.f_halo]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax4.bar(classes, fractions, color=colors)
    ax4.set_ylabel('Mass Fraction')
    ax4.set_title('Morphology Distribution')
    ax4.set_ylim(0, max(fractions) * 1.2)
    for bar, frac in zip(bars, fractions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{frac:.1%}', ha='center', va='bottom', fontsize=10)

    # 5. Conditional density PDFs
    ax5 = fig.add_subplot(2, 3, 5)
    morph = np.array(result.morphology)
    density = np.array(result.particle_density)

    # Normalize by mean
    mean_density = density.mean()
    overdensity = density / mean_density

    bins = np.logspace(-2, 4, 100)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    for i, (name, color) in enumerate(zip(classes, colors)):
        mask = morph == i
        if np.sum(mask) > 0:
            hist, _ = np.histogram(overdensity[mask], bins=bins, density=True)
            ax5.plot(bin_centers, hist, label=name, color=color, linewidth=2)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel(r'$1 + \delta$')
    ax5.set_ylabel('PDF')
    ax5.set_title(r'Conditional PDF: $P(1+\delta | \mathrm{class})$')
    ax5.legend()
    ax5.set_xlim(1e-2, 1e4)
    ax5.grid(True, alpha=0.3)

    # 6. Mean overdensity per class
    ax6 = fig.add_subplot(2, 3, 6)
    mean_overdensities = []
    for i in range(4):
        mask = morph == i
        if np.sum(mask) > 0:
            mean_overdensities.append(overdensity[mask].mean())
        else:
            mean_overdensities.append(0)

    bars = ax6.bar(classes, mean_overdensities, color=colors)
    ax6.axhline(y=1.0, color='black', linestyle='--', label='Mean')
    ax6.set_ylabel(r'Mean $1 + \delta$')
    ax6.set_title('Mean Overdensity by Class')
    ax6.set_yscale('log')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute ORIGAMI morphology classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 HDF5 snapshot')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output plot path (optional)')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Density/morphology grid resolution (default: 128)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples for density (default: 100)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='GADGET particle type (default: 1 for DM)')

    args = parser.parse_args()

    # Load snapshot
    positions, particle_ids, box_size, redshift, scale_factor, particle_mass = \
        load_snapshot(args.snapshot, args.particle_type)

    # Run unified ORIGAMI pipeline (handles detection, sorting, morphology, density, grid)
    result, density_3d, grid_size = run_origami_pipeline(
        positions, particle_ids, box_size,
        output_cells=args.resolution,
        n_samples=args.samples,
        n_threads=args.threads
    )

    # Save results (positions are now sorted in-place)
    save_results(args.output, result, density_3d, positions, box_size,
                 redshift, scale_factor, grid_size, particle_mass)

    # Create visualization
    if args.plot:
        create_visualization(args.plot, result, density_3d, box_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
