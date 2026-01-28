#!/usr/bin/env python3
"""
Diagnostic: scatter/hexbin of local density vs ORIGAMI morphology class.

Creates a 1x2 figure comparing two snapshots (e.g., a=1 and a=100) showing
the distribution of particle density for each morphology class. This helps
diagnose whether M=3 (halo) particles at late times have low local density
(indicating extended multistream regions rather than dense cores).

Usage:
    python plot_density_vs_morphology.py /path/to/snap1.hdf5 /path/to/snap2.hdf5 [options]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import h5py
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add tessera to path if needed
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


ORIGAMI_CLASSES = {
    0: {'name': 'Void',     'color': '#3b82f6'},
    1: {'name': 'Wall',     'color': '#22c55e'},
    2: {'name': 'Filament', 'color': '#f59e0b'},
    3: {'name': 'Halo',     'color': '#ef4444'},
}


def run_origami_with_density(snapshot_path, cartesian_only=False):
    """Run ORIGAMI pipeline with per-particle density sampling."""
    print(f"\nProcessing: {snapshot_path}")

    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])
        redshift = float(f['Header'].attrs['Redshift'])

    n = len(positions)
    grid_size = int(round(n ** (1/3)))
    print(f"  {n:,} particles ({grid_size}^3), a={scale_factor:.4f}, z={redshift:.4f}")

    positions = np.ascontiguousarray(positions, dtype=np.float64)
    particle_ids = np.ascontiguousarray(particle_ids, dtype=np.int64)

    # Run pipeline with density
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.density_output_cells = grid_size
    config.sample_density_at_particles = True
    config.n_threads = 1
    config.pdf_n_bins = 50
    config.cartesian_only = cartesian_only

    result = ts.origami.run_pipeline(positions, particle_ids, config)

    morphology = np.array(result.morphology)
    density = np.array(result.particle_density)

    # Compute overdensity (1+delta)
    mean_density = result.mean_density
    if mean_density > 0:
        overdensity = density / mean_density
    else:
        overdensity = density

    print(f"  Void: {result.f_void:.1%}, Wall: {result.f_wall:.1%}, "
          f"Fil: {result.f_filament:.1%}, Halo: {result.f_halo:.1%}")
    print(f"  Mean density: {mean_density:.4e}")
    print(f"  Overdensity range: [{overdensity.min():.4e}, {overdensity.max():.4e}]")

    return {
        'morphology': morphology,
        'overdensity': overdensity,
        'scale_factor': scale_factor,
        'redshift': redshift,
        'fractions': {
            0: result.f_void, 1: result.f_wall,
            2: result.f_filament, 3: result.f_halo,
        },
    }


def create_diagnostic_figure(data_list, output_path, dpi=200):
    """Create 1xN diagnostic figure with density histograms per morphology class."""

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 11

    n_panels = len(data_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    plt.subplots_adjust(wspace=0.05, left=0.08, right=0.98, bottom=0.12, top=0.85)

    for ax, data in zip(axes, data_list):
        overdensity = data['overdensity']
        morphology = data['morphology']
        a = data['scale_factor']
        z = data['redshift']
        fracs = data['fractions']

        # Log bins for overdensity
        od_pos = overdensity[overdensity > 0]
        log_min = np.log10(max(od_pos.min(), 1e-3))
        log_max = np.log10(min(od_pos.max(), 1e5))
        bins = np.logspace(log_min, log_max, 80)

        # Plot histogram for each class (stacked style)
        for i in [0, 1, 2, 3]:
            mask = morphology == i
            cls = ORIGAMI_CLASSES[i]
            od_cls = overdensity[mask]
            od_cls = od_cls[od_cls > 0]

            ax.hist(od_cls, bins=bins, alpha=0.5, color=cls['color'],
                    label=f"{cls['name']} ({fracs[i]:.0%})",
                    density=True, histtype='stepfilled')
            ax.hist(od_cls, bins=bins, color=cls['color'],
                    density=True, histtype='step', linewidth=1.2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$1 + \delta$', fontsize=13)
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        z_str = f'z = {z:.2f}' if abs(z) < 100 else f'z = {z:.0f}'
        ax.set_title(f'$a = {a:.1f}$  ({z_str})', fontsize=14, fontweight='bold')

        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    axes[0].set_ylabel('Probability density', fontsize=13)

    fig.suptitle(
        'Local Density Distribution by ORIGAMI Morphology Class',
        fontsize=16, fontweight='bold', y=0.95,
    )

    plt.savefig(output_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: density distribution by ORIGAMI morphology class'
    )
    parser.add_argument('snapshots', type=str, nargs='+',
                        help='Path(s) to GADGET-4 HDF5 snapshot(s)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--cartesian-only', action='store_true',
                        help='Use Cartesian axes only (no diagonal checks)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output DPI (default: 200)')

    args = parser.parse_args()

    # Process each snapshot
    data_list = []
    for snap_str in args.snapshots:
        snap_path = Path(snap_str)
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")
        data = run_origami_with_density(snap_path, cartesian_only=args.cartesian_only)
        data_list.append(data)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_cartesian" if args.cartesian_only else ""
        snap_stems = "_".join(Path(s).stem for s in args.snapshots)
        output_path = Path(args.snapshots[0]).parent / f"density_vs_morphology{suffix}_{snap_stems}.png"

    create_diagnostic_figure(data_list, output_path, dpi=args.dpi)


if __name__ == '__main__':
    main()
