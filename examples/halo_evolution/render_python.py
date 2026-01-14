#!/usr/bin/env python3
"""
Density rendering using pure Python implementation (fallback for C++ issues).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np

# Add parent directory to path for tetra_density
sys.path.insert(0, str(Path(__file__).parent.parent))

from tetra_density import (
    compute_tetra_density_field_python,
    project_density_slice,
    load_gadget4_for_tessellation,
)


@dataclass
class HaloInfo:
    """Halo information from branch file."""
    tree_index: int
    snap_num: int
    subhalo_nr: int
    group_nr: int
    scale_factor: float
    redshift: float
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    m200c: float
    r200c: float


def load_branch(branch_file: Path) -> tuple:
    """Load branch data from JSON file."""
    with open(branch_file, 'r') as f:
        data = json.load(f)

    branch = []
    for h in data['halos']:
        branch.append(HaloInfo(
            tree_index=h['tree_index'],
            snap_num=h['snap_num'],
            subhalo_nr=h['subhalo_nr'],
            group_nr=h['group_nr'],
            scale_factor=h['scale_factor'],
            redshift=h['redshift'],
            position=np.array(h['position']),
            velocity=np.array(h['velocity']),
            mass=h['mass'],
            m200c=h['m200c'],
            r200c=h['r200c'],
        ))

    return branch, data['box_size']


def render_density_python(
    snap_num: int,
    halo_position: np.ndarray,
    snapshot_dir: Path,
    subbox_size: float = 1.5,
    output_cells: int = 128,
    n_samples: int = 50,
) -> tuple:
    """Render 2D density field around a halo position using Python implementation."""

    snap_path = snapshot_dir / f'snapshot_{snap_num:03d}.hdf5'
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")

    print(f"    Loading snapshot...")
    positions, particle_ids, header, grid_size = load_gadget4_for_tessellation(
        str(snap_path), particle_type=1
    )
    sim_box_size = header.box_size

    print(f"    Sorting {len(positions):,} particles by Lagrangian ID...")
    from tetra_density import sort_particles_by_lagrangian_id
    sorted_positions = sort_particles_by_lagrangian_id(positions, particle_ids, grid_size)

    # For subbox, we compute full density field then extract the region
    # This is simpler than implementing subbox in Python
    print(f"    Computing density field ({output_cells}^3 cells)...")
    density_3d = compute_tetra_density_field_python(
        sorted_positions,
        box_size=sim_box_size,
        grid_size=grid_size,
        output_cells=output_cells,
        n_samples=n_samples,
        particle_mass=1.0,
        verbose=False,
        batch_size=50000,
    )

    # Extract region around halo and project
    cell_width = sim_box_size / output_cells
    half_subbox = subbox_size / 2

    # Find cell indices for the subbox
    center_idx = (halo_position / cell_width).astype(int) % output_cells

    # Calculate slice range
    n_cells_subbox = int(subbox_size / cell_width)
    half_cells = n_cells_subbox // 2

    # Extract and project a slab centered on halo (along z)
    z_min = int((halo_position[2] - half_subbox) / cell_width) % output_cells
    z_max = int((halo_position[2] + half_subbox) / cell_width) % output_cells

    # Handle periodic wrapping for slice extraction
    if z_min < z_max:
        slice_density = density_3d[z_min:z_max, :, :].sum(axis=0)
    else:
        slice_density = density_3d[z_min:, :, :].sum(axis=0) + density_3d[:z_max, :, :].sum(axis=0)

    # Extract x-y region around halo
    x_min = int((halo_position[0] - half_subbox) / cell_width) % output_cells
    x_max = int((halo_position[0] + half_subbox) / cell_width) % output_cells
    y_min = int((halo_position[1] - half_subbox) / cell_width) % output_cells
    y_max = int((halo_position[1] + half_subbox) / cell_width) % output_cells

    # Handle periodic wrapping for region extraction
    if x_min < x_max and y_min < y_max:
        region = slice_density[y_min:y_max, x_min:x_max]
    else:
        # Need to handle wrapping - just use the full slice for simplicity
        region = slice_density

    # Multiply by cell width to get surface density
    region = region * cell_width

    return region, sim_box_size


def create_figure(
    densities: Dict[float, np.ndarray],
    branch: List[HaloInfo],
    output_path: Path,
    epochs: List[float],
    box_size: float,
    mean_density: float,
):
    """Create multi-panel figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Find nearest available epochs
    available = sorted(densities.keys())
    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)

    # Create figure
    fig_width = 3.5 * n_panels + 1
    fig_height = 4.3

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), dpi=150)
    if n_panels == 1:
        axes = [axes]

    for ax, epoch in zip(axes, epochs_to_plot):
        density = densities[epoch]

        # Convert to overdensity
        plot_data = density / mean_density

        # Handle zeros for log scale
        positive_mask = plot_data > 0
        if positive_mask.any():
            min_positive = plot_data[positive_mask].min()
            plot_data = np.maximum(plot_data, min_positive * 0.1)

        vmin = np.percentile(plot_data[plot_data > 0], 0.5)
        vmax = np.percentile(plot_data, 99.9)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Get extent from density shape
        n_cells = density.shape[0]
        half_size = box_size / 2
        extent = (-half_size, half_size, -half_size, half_size)

        im = ax.imshow(
            plot_data,
            extent=extent,
            origin='lower',
            cmap='viridis',
            norm=norm,
            aspect='equal',
            interpolation='nearest',
        )

        cbar = fig.colorbar(im, ax=ax, location='top', pad=0.02, shrink=0.9)
        cbar.set_label(r'$1 + \delta(a)$', fontsize=10)

        ax.text(
            0.05, 0.95, f'$a = {epoch:.2f}$',
            transform=ax.transAxes,
            fontsize=11,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
        )

        ax.set_xlabel(r'$x$ [cMpc/h]', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel(r'$y$ [cMpc/h]', fontsize=10)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Render density fields (Python implementation)')
    parser.add_argument('--branch-file', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=float, nargs='+', default=[1.0, 10.0, 100.0])
    parser.add_argument('--box-size', type=float, default=1.5)
    parser.add_argument('--output-cells', type=int, default=128)
    parser.add_argument('--n-samples', type=int, default=50)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DENSITY RENDERING (Python implementation)")
    print("=" * 60)
    print()

    # Load branch
    print(f"Loading branch from {args.branch_file}")
    branch, tree_box_size = load_branch(args.branch_file)
    print(f"  Loaded {len(branch)} halos")
    print()

    # Find which snapshots to render
    halo_lookup = {h.scale_factor: h for h in branch}
    available_epochs = sorted(halo_lookup.keys())

    to_render = []
    for target in args.epochs:
        nearest = min(available_epochs, key=lambda x: abs(x - target))
        if nearest not in [h.scale_factor for h in to_render]:
            to_render.append(halo_lookup[nearest])

    print(f"Rendering {len(to_render)} epochs: {[f'a={h.scale_factor:.2f}' for h in to_render]}")
    print()

    # Render densities
    densities = {}
    sim_box_size = None

    for i, halo in enumerate(to_render):
        print(f"  [{i+1}/{len(to_render)}] Rendering a={halo.scale_factor:.2f} (snap {halo.snap_num})...")
        density, sim_box_size = render_density_python(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=args.snapshot_dir,
            subbox_size=args.box_size,
            output_cells=args.output_cells,
            n_samples=args.n_samples,
        )
        densities[halo.scale_factor] = density
        print(f"    Done! Shape: {density.shape}")

    print()

    # Compute mean density
    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3) * args.box_size  # Surface density

    # Create figure
    print("Creating multi-panel figure...")
    output_path = args.output_dir / 'halo_evolution_multipanel.png'
    create_figure(
        densities=densities,
        branch=branch,
        output_path=output_path,
        epochs=args.epochs,
        box_size=args.box_size,
        mean_density=mean_density,
    )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
