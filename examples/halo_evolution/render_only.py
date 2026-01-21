#!/usr/bin/env python3
"""
Standalone density rendering script (avoids h5py for macOS compatibility).

.. deprecated::
    This script is superseded by halo_evolution_pipeline.py which uses
    the consolidated utils and visualize modules:

        python halo_evolution_pipeline.py \\
            --branch-file branch_data.json \\
            --snapshot-dir /path/to/snapshots \\
            --output-dir ./output

    See also:
    - utils.DensityRenderer: High-level density rendering class
    - visualize.create_evolution_multipanel: Multi-panel figure generation

This script loads branch data from JSON and renders density fields using
only tessera, avoiding the HDF5 library conflict on macOS.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tessera (handles platform-specific configuration automatically)
try:
    import tessera as ts
except ImportError:
    # Fallback for development: try to import native module directly
    _script_dir = Path(__file__).parent.resolve()
    _repo_root = _script_dir.parent.parent
    _build_dir = _repo_root / 'build'
    if _build_dir.exists():
        sys.path.insert(0, str(_build_dir))
    import _tessera as ts


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


def render_density(
    snap_num: int,
    halo_position: np.ndarray,
    snapshot_dir: Path,
    box_size: float = 1.5,
    output_cells: int = 256,
    n_samples: int = 100,
) -> np.ndarray:
    """Render 2D density field around a halo position."""

    # Load snapshot
    snap_path = snapshot_dir / f'snapshot_{snap_num:03d}.hdf5'
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")

    snapshot = ts.io.read_gadget4_snapshot(
        str(snap_path),
        particle_types=2,  # Dark matter (type 1)
        read_velocities=False,
        read_ids=True,
    )

    particles = snapshot.particles[1]
    positions = np.array(particles.coordinates, dtype=np.float64)
    particle_ids = np.array(particles.particle_ids, dtype=np.int64)
    sim_box_size = snapshot.header.box_size

    # Infer grid size
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    # Sort by Lagrangian ID
    sorted_positions = ts.density.sort_by_lagrangian_id(
        positions, particle_ids, grid_size, id_offset=1
    )

    # Compute subbox origin
    half_width = box_size / 2.0
    origin = halo_position - half_width

    # Handle periodic wrapping
    for i in range(3):
        if origin[i] < 0:
            origin[i] += sim_box_size
        elif origin[i] >= sim_box_size:
            origin[i] -= sim_box_size

    # Configure tessera
    config = ts.density.TetraDensityConfig()
    config.box_size = sim_box_size
    config.lagrangian_grid_size = grid_size
    config.output_cells = output_cells
    config.n_samples = n_samples
    config.n_threads = 0
    config.periodic = True
    config.particle_mass = 1.0
    config.subbox_enabled = True
    config.subbox_origin = tuple(origin)
    config.subbox_width = (box_size, box_size, box_size)

    # Compute 2D projection
    result = ts.density.compute_tetra_density_2d_projection(
        sorted_positions, config, 2  # Project along z
    )

    density = np.array(result.density).reshape(output_cells, output_cells)
    return density, sim_box_size


def setup_latex_style():
    """Configure matplotlib for LaTeX-style fonts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'text.usetex': False,  # Use mathtext (doesn't require LaTeX installation)
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',  # Computer Modern for math
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def create_figure(
    densities: Dict[float, np.ndarray],
    branch: List[HaloInfo],
    output_path: Path,
    epochs: List[float],
    box_size: float,
    mean_density: float,
):
    """Create multi-panel figure with white background and LaTeX fonts."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_latex_style()

    # Find nearest available epochs
    available = sorted(densities.keys())
    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)

    # Create figure with white background
    fig_width = 3.8 * n_panels + 0.5
    fig_height = 4.2

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor('white')

    if n_panels == 1:
        axes = [axes]

    half_size = box_size / 2
    extent = (-half_size, half_size, -half_size, half_size)

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
        cbar.set_label(r'$1 + \delta$', fontsize=11)

        # Scale factor label
        ax.text(
            0.05, 0.95, f'$a = {epoch:.2f}$',
            transform=ax.transAxes,
            fontsize=12,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        )

        ax.set_xlabel(r'$x$ [cMpc/$h$]', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel(r'$y$ [cMpc/$h$]', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figure to {output_path}")


def create_animation(
    densities: Dict[float, np.ndarray],
    output_path: Path,
    box_size: float,
    mean_density: float,
    fps: int = 10,
):
    """Create MP4 animation of halo evolution."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    setup_latex_style()

    # Sort epochs
    epochs = sorted(densities.keys())
    if len(epochs) == 0:
        print("No epochs available for animation")
        return

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)
    fig.patch.set_facecolor('white')

    half_size = box_size / 2
    extent = (-half_size, half_size, -half_size, half_size)

    # Initial frame
    first_density = densities[epochs[0]]
    first_data = first_density / mean_density

    positive_mask = first_data > 0
    if positive_mask.any():
        min_pos = first_data[positive_mask].min()
        first_data = np.maximum(first_data, min_pos * 0.1)
        vmin = np.percentile(first_data[positive_mask], 0.5)
        vmax = np.percentile(first_data, 99.9)
    else:
        vmin, vmax = 1, 100

    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        first_data,
        extent=extent,
        origin='lower',
        cmap='viridis',
        norm=norm,
        aspect='equal',
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'$1 + \delta$', fontsize=12)

    # Scale factor text
    text = ax.text(
        0.05, 0.95, '',
        transform=ax.transAxes,
        fontsize=14,
        color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
    )

    ax.set_xlabel(r'$x$ [cMpc/$h$]', fontsize=12)
    ax.set_ylabel(r'$y$ [cMpc/$h$]', fontsize=12)
    ax.set_title(r'Dark Matter Halo Evolution', fontsize=14)

    def update(frame_idx):
        epoch = epochs[frame_idx]
        density = densities[epoch]
        plot_data = density / mean_density

        positive_mask = plot_data > 0
        if positive_mask.any():
            min_pos = plot_data[positive_mask].min()
            plot_data = np.maximum(plot_data, min_pos * 0.1)
            vmin = np.percentile(plot_data[positive_mask], 0.5)
            vmax = np.percentile(plot_data, 99.9)
            im.set_norm(LogNorm(vmin=vmin, vmax=vmax))

        im.set_data(plot_data)
        text.set_text(f'$a = {epoch:.2f}$')
        return [im, text]

    anim = FuncAnimation(
        fig, update,
        frames=len(epochs),
        interval=1000 // fps,
        blit=True,
    )

    plt.tight_layout()

    # Save animation
    print(f"Saving animation ({len(epochs)} frames at {fps} fps)...")
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved animation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Render density fields from branch data')
    parser.add_argument('--branch-file', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=float, nargs='+', default=[1.0, 10.0, 100.0])
    parser.add_argument('--box-size', type=float, default=1.5)
    parser.add_argument('--output-cells', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--animation', action='store_true', help='Generate animation')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DENSITY RENDERING (tessera only)")
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

    if args.animation:
        # Render all epochs for animation
        to_render = branch
        print(f"Rendering all {len(to_render)} epochs for animation...")
    else:
        # Render only selected epochs for figure
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
        density, sim_box_size = render_density(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=args.snapshot_dir,
            box_size=args.box_size,
            output_cells=args.output_cells,
            n_samples=args.n_samples,
        )
        densities[halo.scale_factor] = density

    print()

    # Compute mean density
    n_particles = 256 ** 3  # From branch file name hint
    mean_density = n_particles / (sim_box_size ** 3)

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

    # Create animation if requested
    if args.animation:
        print()
        print("Creating animation...")
        anim_path = args.output_dir / 'halo_evolution.mp4'
        create_animation(
            densities=densities,
            output_path=anim_path,
            box_size=args.box_size,
            mean_density=mean_density,
            fps=args.fps,
        )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
