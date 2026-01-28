#!/usr/bin/env python3
"""
Create 1x5 figure showing ORIGAMI morphological particle breakdown.

Panels (left to right):
1. All particles in thin slice
2. Void particles
3. Wall particles
4. Filament particles
5. Halo particles

Each panel shows the actual particle positions (scatter plot, no tessellation).
ORIGAMI is run on the full box for correct classification, then only thin
slice particles are plotted.

Usage:
    python plot_origami_particle_breakdown.py /path/to/snapshot.hdf5 [options]
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

# Add tessera to path if needed
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


# ORIGAMI class definitions with colors
ORIGAMI_CLASSES = {
    0: {'name': 'Void',     'color': '#3b82f6'},  # Blue
    1: {'name': 'Wall',     'color': '#22c55e'},  # Green
    2: {'name': 'Filament', 'color': '#f59e0b'},  # Amber/Orange
    3: {'name': 'Halo',     'color': '#ef4444'},  # Red
}


def load_and_classify(snapshot_path: Path, box_size_mpc: float) -> tuple:
    """Load snapshot, run ORIGAMI on full box, return positions and morphology."""
    print(f"Loading snapshot: {snapshot_path}")

    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = f['Header'].attrs['BoxSize']
        scale_factor = f['Header'].attrs['Time']
        redshift = f['Header'].attrs['Redshift']

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    print(f"  Particles: {n_particles:,} ({grid_size}^3)")
    print(f"  Box size: {box_size:.1f} Mpc/h")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Redshift: {redshift:.4f}")

    positions = np.ascontiguousarray(positions, dtype=np.float64)
    particle_ids = np.ascontiguousarray(particle_ids, dtype=np.int64)

    # Run ORIGAMI on full box
    print("\nRunning ORIGAMI on full box...")
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = grid_size
    config.sample_density_at_particles = False
    config.n_threads = 1
    config.pdf_n_bins = 50

    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"  Global ORIGAMI fractions:")
    print(f"    Void:     {result.f_void:.1%}")
    print(f"    Wall:     {result.f_wall:.1%}")
    print(f"    Filament: {result.f_filament:.1%}")
    print(f"    Halo:     {result.f_halo:.1%}")

    morphology = np.array(result.morphology)

    # Sort positions to Lagrangian order to match morphology
    sorted_positions, _, _ = ts.origami.sort_particles_to_lagrangian(
        positions, particle_ids, grid_size, float(box_size)
    )

    return sorted_positions, morphology, float(box_size), scale_factor, redshift


def create_breakdown_figure(
    positions: np.ndarray,
    morphology: np.ndarray,
    box_size: float,
    scale_factor: float,
    redshift: float,
    slice_center: float,
    slice_thickness: float,
    output_path: Path,
    point_size: float = 0.01,
    dpi: int = 200,
):
    """Create 1x5 figure with all particles + per-class breakdown."""

    # Select particles in the thin slice (z-axis)
    z_coords = positions[:, 2]
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2
    in_slice = (z_coords >= slice_min) & (z_coords < slice_max)

    slice_x = positions[in_slice, 0]
    slice_y = positions[in_slice, 1]
    slice_morph = morphology[in_slice]

    n_in_slice = in_slice.sum()
    print(f"\n  Particles in slice: {n_in_slice:,}")

    # Compute per-class counts in slice
    for i in range(4):
        cls = ORIGAMI_CLASSES[i]
        count = (slice_morph == i).sum()
        frac = count / n_in_slice if n_in_slice > 0 else 0
        print(f"    {cls['name']:>10}: {count:>10,} ({frac:.1%})")

    # Setup figure - LaTeX fonts, white background
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(
        1, 5, figsize=(25, 5.5),
        sharex=True, sharey=True,
    )
    plt.subplots_adjust(wspace=0, hspace=0, left=0.04, right=0.99, bottom=0.10, top=0.88)

    panel_configs = [
        {'label': 'All Particles', 'mask': np.ones(len(slice_morph), dtype=bool), 'color': '#333333'},
        {'label': 'Void',          'mask': slice_morph == 0, 'color': ORIGAMI_CLASSES[0]['color']},
        {'label': 'Wall',          'mask': slice_morph == 1, 'color': ORIGAMI_CLASSES[1]['color']},
        {'label': 'Filament',      'mask': slice_morph == 2, 'color': ORIGAMI_CLASSES[2]['color']},
        {'label': 'Halo',          'mask': slice_morph == 3, 'color': ORIGAMI_CLASSES[3]['color']},
    ]

    for i, (ax, cfg) in enumerate(zip(axes, panel_configs)):
        ax.set_facecolor('white')
        mask = cfg['mask']
        n_pts = mask.sum()
        frac = n_pts / n_in_slice if n_in_slice > 0 else 0

        ax.scatter(
            slice_x[mask], slice_y[mask],
            s=point_size,
            c=cfg['color'],
            marker='.',
            linewidths=0,
            alpha=0.8,
            rasterized=True,
        )

        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal')

        # Title with count/fraction
        if i == 0:
            title = f"{cfg['label']}\n({n_pts:,})"
        else:
            title = f"{cfg['label']}\n({frac:.1%})"
        ax.set_title(title, fontsize=14, fontweight='bold', color=cfg['color'])

        # X label on all panels
        ax.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)

        # Y label only on leftmost
        if i == 0:
            ax.set_ylabel(r'$y$ [Mpc/$h$]', fontsize=12)
        else:
            ax.tick_params(axis='y', labelleft=False, length=0)

    # Suptitle
    z_str = f'z = {redshift:.2f}' if abs(redshift) < 100 else f'z = {redshift:.0f}'
    fig.suptitle(
        f'ORIGAMI Morphological Classification  —  $a = {scale_factor:.1f}$ ({z_str})\n'
        f'Thin slice: {slice_thickness:.1f} cMpc/$h$ at center',
        fontsize=16, fontweight='bold', y=0.98,
    )

    plt.savefig(output_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"\n  Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create 1x5 ORIGAMI particle breakdown figure'
    )
    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 HDF5 snapshot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--slice-thickness', type=float, default=5.0,
                        help='Slice thickness in cMpc/h (default: 5.0)')
    parser.add_argument('--point-size', type=float, default=None,
                        help='Scatter point size (default: auto-scaled based on box/particle count)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output DPI (default: 200)')

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    # Load and classify
    positions, morphology, box_size, scale_factor, redshift = load_and_classify(
        snapshot_path, box_size_mpc=0  # unused, read from file
    )

    # Slice parameters
    slice_center = box_size / 2
    slice_thickness = args.slice_thickness

    print(f"\nSlice: center={slice_center:.1f}, thickness={slice_thickness:.1f} cMpc/h")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = snapshot_path.parent / f"origami_breakdown_{snapshot_path.stem}.png"

    # Auto-scale point size if not specified
    # Aim for each particle to be ~1-2 pixels at the output resolution
    if args.point_size is None:
        # Estimate particles in slice
        grid_size = int(round(len(positions) ** (1/3)))
        n_layers = max(1, slice_thickness / (box_size / grid_size))
        n_in_slice_est = grid_size * grid_size * n_layers
        # Scale: fewer particles -> larger points
        point_size = max(0.05, min(5.0, 500_000 / n_in_slice_est))
        print(f"  Auto point size: {point_size:.2f} (est. {n_in_slice_est:.0f} particles in slice)")
    else:
        point_size = args.point_size

    create_breakdown_figure(
        positions, morphology,
        box_size, scale_factor, redshift,
        slice_center, slice_thickness,
        output_path,
        point_size=point_size,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
