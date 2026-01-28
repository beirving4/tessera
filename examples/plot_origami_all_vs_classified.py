#!/usr/bin/env python3
"""
Create 1x2 figure: all particles vs ORIGAMI-classified particles.

Left panel: All particles in thin slice (single color)
Right panel: All particles colored by their ORIGAMI morphology class

Usage:
    python plot_origami_all_vs_classified.py /path/to/snapshot.hdf5 [options]
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


def main():
    parser = argparse.ArgumentParser(
        description='Create 1x2 all particles vs ORIGAMI-classified particles'
    )
    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 HDF5 snapshot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--slice-thickness', type=float, default=5.0,
                        help='Slice thickness in cMpc/h (default: 5.0)')
    parser.add_argument('--point-size', type=float, default=None,
                        help='Scatter point size (default: auto)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output DPI (default: 200)')

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    # Load snapshot
    print(f"Loading snapshot: {snapshot_path}")
    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])
        redshift = float(f['Header'].attrs['Redshift'])

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))
    print(f"  Particles: {n_particles:,} ({grid_size}^3)")
    print(f"  Box size: {box_size:.1f} Mpc/h, a={scale_factor:.4f}, z={redshift:.4f}")

    positions = np.ascontiguousarray(positions, dtype=np.float64)
    particle_ids = np.ascontiguousarray(particle_ids, dtype=np.int64)

    # Run ORIGAMI on full box
    print("\nRunning ORIGAMI on full box...")
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.density_output_cells = grid_size
    config.sample_density_at_particles = False
    config.n_threads = 1
    config.pdf_n_bins = 50

    result = ts.origami.run_pipeline(positions, particle_ids, config)
    morphology = np.array(result.morphology)

    print(f"  Void: {result.f_void:.1%}, Wall: {result.f_wall:.1%}, "
          f"Filament: {result.f_filament:.1%}, Halo: {result.f_halo:.1%}")

    # Sort positions to Lagrangian order to match morphology
    sorted_positions, _, _ = ts.origami.sort_particles_to_lagrangian(
        positions, particle_ids, grid_size, box_size
    )

    # Select thin slice
    slice_center = box_size / 2
    slice_thickness = args.slice_thickness
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2

    z_coords = sorted_positions[:, 2]
    in_slice = (z_coords >= slice_min) & (z_coords < slice_max)

    slice_x = sorted_positions[in_slice, 0]
    slice_y = sorted_positions[in_slice, 1]
    slice_morph = morphology[in_slice]
    n_in_slice = in_slice.sum()

    print(f"\n  Particles in slice: {n_in_slice:,}")
    slice_fracs = {}
    for i in range(4):
        cls = ORIGAMI_CLASSES[i]
        count = (slice_morph == i).sum()
        frac = count / n_in_slice
        slice_fracs[i] = frac
        print(f"    {cls['name']:>10}: {count:>10,} ({frac:.1%})")

    # Auto-scale point size
    if args.point_size is None:
        n_layers = max(1, slice_thickness / (box_size / grid_size))
        n_est = grid_size * grid_size * n_layers
        point_size = max(0.05, min(5.0, 500_000 / n_est))
        print(f"  Auto point size: {point_size:.2f}")
    else:
        point_size = args.point_size

    # Build per-particle color array for classified panel
    colors_classified = np.empty((len(slice_morph), 3), dtype=np.float64)
    for i in range(4):
        mask = slice_morph == i
        hex_color = ORIGAMI_CLASSES[i]['color']
        h = hex_color.lstrip('#')
        rgb = [int(h[j:j+2], 16) / 255.0 for j in (0, 2, 4)]
        colors_classified[mask] = rgb

    # Create figure
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, left=0.07, right=0.98, bottom=0.10, top=0.88)

    # Left panel: all particles in single color
    ax_all = axes[0]
    ax_all.set_facecolor('white')
    ax_all.scatter(
        slice_x, slice_y,
        s=point_size, c='#333333', marker='.', linewidths=0,
        alpha=0.8, rasterized=True,
    )
    ax_all.set_xlim(0, box_size)
    ax_all.set_ylim(0, box_size)
    ax_all.set_aspect('equal')
    ax_all.set_title(f'All Particles ({n_in_slice:,})', fontsize=14, fontweight='bold')
    ax_all.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)
    ax_all.set_ylabel(r'$y$ [Mpc/$h$]', fontsize=12)

    # Right panel: classified particles
    ax_cls = axes[1]
    ax_cls.set_facecolor('white')

    # Plot each class separately so dense regions overlay correctly
    # Draw void first (background), then wall, filament, halo on top
    for i in [0, 1, 2, 3]:
        mask = slice_morph == i
        ax_cls.scatter(
            slice_x[mask], slice_y[mask],
            s=point_size,
            c=ORIGAMI_CLASSES[i]['color'],
            marker='.', linewidths=0, alpha=0.8, rasterized=True,
        )

    ax_cls.set_xlim(0, box_size)
    ax_cls.set_ylim(0, box_size)
    ax_cls.set_aspect('equal')
    ax_cls.set_title('ORIGAMI Classification', fontsize=14, fontweight='bold')
    ax_cls.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)
    ax_cls.tick_params(axis='y', labelleft=False, length=0)

    # Legend on right panel
    legend_elements = [
        Patch(facecolor=ORIGAMI_CLASSES[i]['color'], edgecolor='gray', linewidth=0.5,
              label=f"{ORIGAMI_CLASSES[i]['name']} ({slice_fracs[i]:.0%})")
        for i in range(4)
    ]
    ax_cls.legend(
        handles=legend_elements, loc='upper right',
        framealpha=0.9, fontsize=10,
    )

    # Suptitle
    z_str = f'z = {redshift:.2f}' if abs(redshift) < 100 else f'z = {redshift:.0f}'
    fig.suptitle(
        f'ORIGAMI Morphological Classification  —  $a = {scale_factor:.1f}$ ({z_str})\n'
        f'Thin slice: {slice_thickness:.1f} cMpc/$h$ at center',
        fontsize=16, fontweight='bold', y=0.98,
    )

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = snapshot_path.parent / f"origami_classified_{snapshot_path.stem}.png"

    plt.savefig(output_path, dpi=args.dpi, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure to {output_path}")


if __name__ == '__main__':
    main()
