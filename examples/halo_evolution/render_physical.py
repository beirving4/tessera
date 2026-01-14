#!/usr/bin/env python3
"""
Physical vs Comoving Coordinate Visualization for Halo Evolution

Demonstrates the freeze-out/isolation of halos by showing:
1. Fixed physical box: shows halo stabilization and isolation at late times
2. Fixed comoving box: shows apparent shrinking (what simulations show directly)

The key insight: in physical coordinates, halos reach a stable size and become
isolated as the universe expands. In comoving coordinates, this appears as
continuous shrinking.

Usage:
    # For thesis text (fixed physical box, multi-panel):
    python render_physical.py --branch-file branch.json --snapshot-dir /path/to/snaps \\
        --output-dir /path/to/output --mode physical --physical-box-size 15.0

    # For thesis defense (side-by-side animation):
    python render_physical.py --branch-file branch.json --snapshot-dir /path/to/snaps \\
        --output-dir /path/to/output --mode comparison --animation
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tessera
try:
    import tessera as ts
except ImportError:
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


def render_density_2d(
    snap_num: int,
    halo_position: np.ndarray,
    snapshot_dir: Path,
    comoving_box_size: float,
    output_cells: int = 256,
    n_samples: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Render 2D density field around a halo position.

    Parameters
    ----------
    comoving_box_size : float
        Box size in comoving Mpc/h
    """
    snap_path = snapshot_dir / f'snapshot_{snap_num:03d}.hdf5'
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")

    snapshot = ts.io.read_gadget4_snapshot(
        str(snap_path),
        particle_types=2,
        read_velocities=False,
        read_ids=True,
    )

    particles = snapshot.particles[1]
    positions = np.array(particles.coordinates, dtype=np.float64)
    particle_ids = np.array(particles.particle_ids, dtype=np.int64)
    sim_box_size = snapshot.header.box_size

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    sorted_positions = ts.density.sort_by_lagrangian_id(
        positions, particle_ids, grid_size, id_offset=1
    )

    # Compute subbox origin (comoving coordinates)
    half_width = comoving_box_size / 2.0
    origin = halo_position - half_width

    # Handle periodic wrapping
    for i in range(3):
        if origin[i] < 0:
            origin[i] += sim_box_size
        elif origin[i] >= sim_box_size:
            origin[i] -= sim_box_size

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
    config.subbox_width = (comoving_box_size, comoving_box_size, comoving_box_size)

    result = ts.density.compute_tetra_density_2d_projection(
        sorted_positions, config, 2
    )

    density = np.array(result.density).reshape(output_cells, output_cells)
    return density, sim_box_size


def setup_latex_style():
    """Configure matplotlib for LaTeX-style fonts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def create_physical_box_figure(
    branch: List[HaloInfo],
    snapshot_dir: Path,
    output_path: Path,
    physical_box_size: float,
    epochs: List[float],
    output_cells: int = 256,
    n_samples: int = 100,
):
    """
    Create multi-panel figure with FIXED PHYSICAL BOX size.

    This shows the freeze-out: at early times, we see surrounding structure.
    At late times, the halo is isolated as the universe has expanded.

    Parameters
    ----------
    physical_box_size : float
        Box size in physical Mpc (NOT comoving)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_latex_style()

    # Find nearest available epochs
    halo_lookup = {h.scale_factor: h for h in branch}
    available = sorted(halo_lookup.keys())

    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)

    print(f"Creating fixed PHYSICAL box figure ({physical_box_size} Mpc)")
    print(f"Epochs: {epochs_to_plot}")

    # Compute densities for each epoch
    densities = {}
    sim_box_size = None

    for i, epoch in enumerate(epochs_to_plot):
        halo = halo_lookup[epoch]

        # Convert physical box to comoving: comoving = physical / a
        comoving_box = physical_box_size / epoch

        print(f"  [{i+1}/{n_panels}] a={epoch:.2f}: physical={physical_box_size:.1f} Mpc -> comoving={comoving_box:.3f} cMpc/h")

        # Check if comoving box is reasonable
        if comoving_box < 0.01:
            print(f"    Warning: Very small comoving box ({comoving_box:.4f} cMpc/h), may have few particles")

        density, sim_box_size = render_density_2d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=snapshot_dir,
            comoving_box_size=comoving_box,
            output_cells=output_cells,
            n_samples=n_samples,
        )
        densities[epoch] = density

    # Compute mean density for normalization
    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3)

    # Create figure
    fig_width = 3.8 * n_panels + 0.5
    fig_height = 4.5

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor('white')

    if n_panels == 1:
        axes = [axes]

    # Use PHYSICAL coordinates for extent (same for all panels)
    half_phys = physical_box_size / 2
    extent_phys = (-half_phys, half_phys, -half_phys, half_phys)

    # Find global color scale
    all_overdensities = []
    for epoch, density in densities.items():
        od = density / mean_density
        positive = od[od > 0]
        if len(positive) > 0:
            all_overdensities.extend(positive.flatten())

    if len(all_overdensities) > 0:
        vmin = np.percentile(all_overdensities, 1)
        vmax = np.percentile(all_overdensities, 99.5)
    else:
        vmin, vmax = 0.1, 1000

    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, epoch in zip(axes, epochs_to_plot):
        density = densities[epoch]
        overdensity = density / mean_density

        # Handle zeros for log scale
        positive_mask = overdensity > 0
        if positive_mask.any():
            min_pos = overdensity[positive_mask].min()
            overdensity = np.maximum(overdensity, min_pos * 0.1)

        im = ax.imshow(
            overdensity,
            extent=extent_phys,
            origin='lower',
            cmap='viridis',
            norm=norm,
            aspect='equal',
            interpolation='nearest',
        )

        # Add colorbar on top
        cbar = fig.colorbar(im, ax=ax, location='top', pad=0.02, shrink=0.9)
        cbar.set_label(r'$1 + \delta$', fontsize=11)

        # Scale factor label
        ax.text(
            0.05, 0.95, f'$a = {epoch:.1f}$',
            transform=ax.transAxes,
            fontsize=12,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        )

        # Physical coordinate labels
        ax.set_xlabel(r'$x$ [Mpc]', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel(r'$y$ [Mpc]', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    # Add title indicating fixed physical scale
    fig.suptitle(
        f'Fixed Physical Box: {physical_box_size:.0f} Mpc',
        fontsize=14, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved fixed-physical figure to {output_path}")


def create_comoving_box_figure(
    branch: List[HaloInfo],
    snapshot_dir: Path,
    output_path: Path,
    comoving_box_size: float,
    epochs: List[float],
    output_cells: int = 256,
    n_samples: int = 100,
):
    """
    Create multi-panel figure with FIXED COMOVING BOX size.

    This shows what simulations directly output: the halo appears to
    shrink over time as it collapses in comoving coordinates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_latex_style()

    halo_lookup = {h.scale_factor: h for h in branch}
    available = sorted(halo_lookup.keys())

    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)

    print(f"Creating fixed COMOVING box figure ({comoving_box_size} cMpc/h)")

    densities = {}
    sim_box_size = None

    for i, epoch in enumerate(epochs_to_plot):
        halo = halo_lookup[epoch]
        print(f"  [{i+1}/{n_panels}] a={epoch:.2f}: comoving={comoving_box_size:.2f} cMpc/h")

        density, sim_box_size = render_density_2d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=snapshot_dir,
            comoving_box_size=comoving_box_size,
            output_cells=output_cells,
            n_samples=n_samples,
        )
        densities[epoch] = density

    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3)

    fig_width = 3.8 * n_panels + 0.5
    fig_height = 4.5

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor('white')

    if n_panels == 1:
        axes = [axes]

    half_com = comoving_box_size / 2
    extent_com = (-half_com, half_com, -half_com, half_com)

    # Global color scale
    all_overdensities = []
    for density in densities.values():
        od = density / mean_density
        positive = od[od > 0]
        if len(positive) > 0:
            all_overdensities.extend(positive.flatten())

    vmin = np.percentile(all_overdensities, 1) if all_overdensities else 0.1
    vmax = np.percentile(all_overdensities, 99.5) if all_overdensities else 1000
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, epoch in zip(axes, epochs_to_plot):
        density = densities[epoch]
        overdensity = density / mean_density

        positive_mask = overdensity > 0
        if positive_mask.any():
            min_pos = overdensity[positive_mask].min()
            overdensity = np.maximum(overdensity, min_pos * 0.1)

        im = ax.imshow(
            overdensity,
            extent=extent_com,
            origin='lower',
            cmap='viridis',
            norm=norm,
            aspect='equal',
            interpolation='nearest',
        )

        cbar = fig.colorbar(im, ax=ax, location='top', pad=0.02, shrink=0.9)
        cbar.set_label(r'$1 + \delta$', fontsize=11)

        ax.text(
            0.05, 0.95, f'$a = {epoch:.1f}$',
            transform=ax.transAxes,
            fontsize=12,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        )

        # Comoving coordinate labels
        ax.set_xlabel(r'$x$ [cMpc/$h$]', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel(r'$y$ [cMpc/$h$]', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    fig.suptitle(
        f'Fixed Comoving Box: {comoving_box_size:.1f} cMpc/$h$',
        fontsize=14, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved fixed-comoving figure to {output_path}")


def create_comparison_figure(
    branch: List[HaloInfo],
    snapshot_dir: Path,
    output_path: Path,
    physical_box_size: float,
    comoving_box_size: float,
    epochs: List[float],
    output_cells: int = 256,
    n_samples: int = 100,
):
    """
    Create side-by-side comparison: comoving vs physical at each epoch.

    Top row: Fixed comoving box (shows apparent shrinking)
    Bottom row: Fixed physical box (shows isolation/freeze-out)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_latex_style()

    halo_lookup = {h.scale_factor: h for h in branch}
    available = sorted(halo_lookup.keys())

    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_cols = len(epochs_to_plot)

    print(f"Creating comparison figure")
    print(f"  Physical box: {physical_box_size} Mpc")
    print(f"  Comoving box: {comoving_box_size} cMpc/h")

    # Render both comoving and physical for each epoch
    comoving_densities = {}
    physical_densities = {}
    sim_box_size = None

    for i, epoch in enumerate(epochs_to_plot):
        halo = halo_lookup[epoch]

        # Comoving (fixed box)
        print(f"  [{i+1}/{n_cols}] a={epoch:.2f} comoving...")
        density, sim_box_size = render_density_2d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=snapshot_dir,
            comoving_box_size=comoving_box_size,
            output_cells=output_cells,
            n_samples=n_samples,
        )
        comoving_densities[epoch] = density

        # Physical (comoving box = physical / a)
        phys_to_com = physical_box_size / epoch
        print(f"  [{i+1}/{n_cols}] a={epoch:.2f} physical -> comoving={phys_to_com:.3f}...")

        density, _ = render_density_2d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=snapshot_dir,
            comoving_box_size=phys_to_com,
            output_cells=output_cells,
            n_samples=n_samples,
        )
        physical_densities[epoch] = density

    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3)

    # Create 2-row figure
    fig_width = 3.5 * n_cols + 0.8
    fig_height = 7.5

    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor('white')

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Extents
    half_com = comoving_box_size / 2
    extent_com = (-half_com, half_com, -half_com, half_com)

    half_phys = physical_box_size / 2
    extent_phys = (-half_phys, half_phys, -half_phys, half_phys)

    # Global color scales (separate for comoving and physical)
    def get_color_scale(densities_dict):
        all_od = []
        for d in densities_dict.values():
            od = d / mean_density
            positive = od[od > 0]
            if len(positive) > 0:
                all_od.extend(positive.flatten())
        if len(all_od) > 0:
            return np.percentile(all_od, 1), np.percentile(all_od, 99.5)
        return 0.1, 1000

    vmin_com, vmax_com = get_color_scale(comoving_densities)
    vmin_phys, vmax_phys = get_color_scale(physical_densities)

    norm_com = LogNorm(vmin=vmin_com, vmax=vmax_com)
    norm_phys = LogNorm(vmin=vmin_phys, vmax=vmax_phys)

    # Top row: Comoving
    for j, epoch in enumerate(epochs_to_plot):
        ax = axes[0, j]
        density = comoving_densities[epoch]
        overdensity = density / mean_density

        positive_mask = overdensity > 0
        if positive_mask.any():
            min_pos = overdensity[positive_mask].min()
            overdensity = np.maximum(overdensity, min_pos * 0.1)

        im = ax.imshow(
            overdensity,
            extent=extent_com,
            origin='lower',
            cmap='viridis',
            norm=norm_com,
            aspect='equal',
            interpolation='nearest',
        )

        ax.text(
            0.05, 0.95, f'$a = {epoch:.1f}$',
            transform=ax.transAxes,
            fontsize=11,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        )

        if j == 0:
            ax.set_ylabel(r'$y$ [cMpc/$h$]', fontsize=11)
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)

        if j == 0:
            ax.text(
                -0.25, 0.5, 'Comoving\n(Fixed cMpc/$h$)',
                transform=ax.transAxes,
                fontsize=12,
                rotation=90,
                verticalalignment='center',
                horizontalalignment='center',
                fontweight='bold',
            )

    # Add colorbar for top row
    cbar_ax_com = fig.add_axes([0.92, 0.53, 0.02, 0.35])
    cbar_com = fig.colorbar(im, cax=cbar_ax_com)
    cbar_com.set_label(r'$1 + \delta$', fontsize=11)

    # Bottom row: Physical
    for j, epoch in enumerate(epochs_to_plot):
        ax = axes[1, j]
        density = physical_densities[epoch]
        overdensity = density / mean_density

        positive_mask = overdensity > 0
        if positive_mask.any():
            min_pos = overdensity[positive_mask].min()
            overdensity = np.maximum(overdensity, min_pos * 0.1)

        im = ax.imshow(
            overdensity,
            extent=extent_phys,
            origin='lower',
            cmap='viridis',
            norm=norm_phys,
            aspect='equal',
            interpolation='nearest',
        )

        ax.text(
            0.05, 0.95, f'$a = {epoch:.1f}$',
            transform=ax.transAxes,
            fontsize=11,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        )

        ax.set_xlabel(r'$x$ [Mpc]', fontsize=11)
        if j == 0:
            ax.set_ylabel(r'$y$ [Mpc]', fontsize=11)
        else:
            ax.tick_params(labelleft=False)

        if j == 0:
            ax.text(
                -0.25, 0.5, 'Physical\n(Fixed Mpc)',
                transform=ax.transAxes,
                fontsize=12,
                rotation=90,
                verticalalignment='center',
                horizontalalignment='center',
                fontweight='bold',
            )

    # Add colorbar for bottom row
    cbar_ax_phys = fig.add_axes([0.92, 0.12, 0.02, 0.35])
    cbar_phys = fig.colorbar(im, cax=cbar_ax_phys)
    cbar_phys.set_label(r'$1 + \delta$', fontsize=11)

    plt.subplots_adjust(left=0.12, right=0.90, top=0.95, bottom=0.08, hspace=0.1, wspace=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison figure to {output_path}")


def create_comparison_animation(
    branch: List[HaloInfo],
    snapshot_dir: Path,
    output_path: Path,
    physical_box_size: float,
    comoving_box_size: float,
    output_cells: int = 256,
    n_samples: int = 100,
    fps: int = 5,
):
    """
    Create side-by-side animation: comoving vs physical over time.

    Left panel: Fixed comoving box (structure appears to shrink)
    Right panel: Fixed physical box (structure stabilizes, becomes isolated)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    setup_latex_style()

    # Sort branch by scale factor
    sorted_branch = sorted(branch, key=lambda h: h.scale_factor)

    print(f"Creating comparison animation ({len(sorted_branch)} frames)")
    print(f"  Physical box: {physical_box_size} Mpc")
    print(f"  Comoving box: {comoving_box_size} cMpc/h")

    # Pre-render all frames
    comoving_densities = {}
    physical_densities = {}
    sim_box_size = None

    for i, halo in enumerate(sorted_branch):
        epoch = halo.scale_factor
        print(f"  [{i+1}/{len(sorted_branch)}] Rendering a={epoch:.2f}...")

        # Comoving
        density, sim_box_size = render_density_2d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=snapshot_dir,
            comoving_box_size=comoving_box_size,
            output_cells=output_cells,
            n_samples=n_samples,
        )
        comoving_densities[epoch] = density

        # Physical
        phys_to_com = physical_box_size / epoch
        if phys_to_com > 0.005:  # Only render if box is large enough
            density, _ = render_density_2d(
                snap_num=halo.snap_num,
                halo_position=halo.position,
                snapshot_dir=snapshot_dir,
                comoving_box_size=phys_to_com,
                output_cells=output_cells,
                n_samples=n_samples,
            )
            physical_densities[epoch] = density
        else:
            physical_densities[epoch] = np.zeros((output_cells, output_cells))

    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3)

    # Compute global color scales
    def get_color_scale(densities_dict):
        all_od = []
        for d in densities_dict.values():
            od = d / mean_density
            positive = od[od > 0]
            if len(positive) > 0:
                all_od.extend(positive.flatten())
        if len(all_od) > 0:
            return np.percentile(all_od, 0.5), np.percentile(all_od, 99.9)
        return 0.1, 1000

    vmin_com, vmax_com = get_color_scale(comoving_densities)
    vmin_phys, vmax_phys = get_color_scale(physical_densities)

    # Create figure
    fig, (ax_com, ax_phys) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=120)
    fig.patch.set_facecolor('white')

    half_com = comoving_box_size / 2
    extent_com = (-half_com, half_com, -half_com, half_com)

    half_phys = physical_box_size / 2
    extent_phys = (-half_phys, half_phys, -half_phys, half_phys)

    # Initial frame
    epochs = sorted(comoving_densities.keys())
    first_epoch = epochs[0]

    # Comoving panel
    first_com = comoving_densities[first_epoch] / mean_density
    first_com = np.maximum(first_com, 1e-6)
    im_com = ax_com.imshow(
        first_com,
        extent=extent_com,
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=vmin_com, vmax=vmax_com),
        aspect='equal',
        interpolation='nearest',
    )
    ax_com.set_xlabel(r'$x$ [cMpc/$h$]', fontsize=12)
    ax_com.set_ylabel(r'$y$ [cMpc/$h$]', fontsize=12)
    ax_com.set_title('Comoving Coordinates', fontsize=14, fontweight='bold')

    text_com = ax_com.text(
        0.05, 0.95, f'$a = {first_epoch:.2f}$',
        transform=ax_com.transAxes,
        fontsize=12,
        color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
    )

    cbar_com = fig.colorbar(im_com, ax=ax_com, shrink=0.85)
    cbar_com.set_label(r'$1 + \delta$', fontsize=11)

    # Physical panel
    first_phys = physical_densities[first_epoch] / mean_density
    first_phys = np.maximum(first_phys, 1e-6)
    im_phys = ax_phys.imshow(
        first_phys,
        extent=extent_phys,
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=vmin_phys, vmax=vmax_phys),
        aspect='equal',
        interpolation='nearest',
    )
    ax_phys.set_xlabel(r'$x$ [Mpc]', fontsize=12)
    ax_phys.set_ylabel(r'$y$ [Mpc]', fontsize=12)
    ax_phys.set_title('Physical Coordinates', fontsize=14, fontweight='bold')

    text_phys = ax_phys.text(
        0.05, 0.95, f'$a = {first_epoch:.2f}$',
        transform=ax_phys.transAxes,
        fontsize=12,
        color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
    )

    cbar_phys = fig.colorbar(im_phys, ax=ax_phys, shrink=0.85)
    cbar_phys.set_label(r'$1 + \delta$', fontsize=11)

    plt.tight_layout()

    def update(frame_idx):
        epoch = epochs[frame_idx]

        # Update comoving
        com_data = comoving_densities[epoch] / mean_density
        com_data = np.maximum(com_data, 1e-6)
        im_com.set_data(com_data)
        text_com.set_text(f'$a = {epoch:.2f}$')

        # Update physical
        phys_data = physical_densities[epoch] / mean_density
        phys_data = np.maximum(phys_data, 1e-6)
        im_phys.set_data(phys_data)
        text_phys.set_text(f'$a = {epoch:.2f}$')

        return [im_com, im_phys, text_com, text_phys]

    print(f"Creating animation ({len(epochs)} frames at {fps} fps)...")

    anim = FuncAnimation(
        fig, update,
        frames=len(epochs),
        interval=1000 // fps,
        blit=True,
    )

    # Save animation
    output_path = Path(output_path)
    if output_path.suffix == '.gif':
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2000)

    anim.save(str(output_path), writer=writer)
    plt.close()

    print(f"Saved animation to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Physical vs Comoving visualization for halo evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed physical box (thesis text):
  python render_physical.py --branch-file branch.json --snapshot-dir snaps/ \\
      --output-dir output/ --mode physical --physical-box-size 15.0

  # Comparison figure (comoving vs physical):
  python render_physical.py --branch-file branch.json --snapshot-dir snaps/ \\
      --output-dir output/ --mode comparison

  # Comparison animation (thesis defense):
  python render_physical.py --branch-file branch.json --snapshot-dir snaps/ \\
      --output-dir output/ --mode comparison --animation
        """
    )

    parser.add_argument('--branch-file', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['physical', 'comoving', 'comparison'],
                        help='Visualization mode')

    parser.add_argument('--epochs', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0],
                        help='Scale factors for still frames')

    parser.add_argument('--physical-box-size', type=float, default=15.0,
                        help='Physical box size in Mpc')
    parser.add_argument('--comoving-box-size', type=float, default=1.5,
                        help='Comoving box size in cMpc/h')

    parser.add_argument('--output-cells', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=100)

    parser.add_argument('--animation', action='store_true',
                        help='Create animation instead of still frame')
    parser.add_argument('--fps', type=int, default=5)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHYSICAL vs COMOVING VISUALIZATION")
    print("=" * 60)
    print()

    # Load branch
    print(f"Loading branch from {args.branch_file}")
    branch, tree_box_size = load_branch(args.branch_file)
    print(f"  Loaded {len(branch)} halos")
    print()

    if args.mode == 'physical':
        output_path = args.output_dir / 'halo_evolution_physical.png'
        create_physical_box_figure(
            branch=branch,
            snapshot_dir=args.snapshot_dir,
            output_path=output_path,
            physical_box_size=args.physical_box_size,
            epochs=args.epochs,
            output_cells=args.output_cells,
            n_samples=args.n_samples,
        )

    elif args.mode == 'comoving':
        output_path = args.output_dir / 'halo_evolution_comoving.png'
        create_comoving_box_figure(
            branch=branch,
            snapshot_dir=args.snapshot_dir,
            output_path=output_path,
            comoving_box_size=args.comoving_box_size,
            epochs=args.epochs,
            output_cells=args.output_cells,
            n_samples=args.n_samples,
        )

    elif args.mode == 'comparison':
        if args.animation:
            output_path = args.output_dir / 'halo_evolution_comparison.mp4'
            create_comparison_animation(
                branch=branch,
                snapshot_dir=args.snapshot_dir,
                output_path=output_path,
                physical_box_size=args.physical_box_size,
                comoving_box_size=args.comoving_box_size,
                output_cells=args.output_cells,
                n_samples=args.n_samples,
                fps=args.fps,
            )
        else:
            output_path = args.output_dir / 'halo_evolution_comparison.png'
            create_comparison_figure(
                branch=branch,
                snapshot_dir=args.snapshot_dir,
                output_path=output_path,
                physical_box_size=args.physical_box_size,
                comoving_box_size=args.comoving_box_size,
                epochs=args.epochs,
                output_cells=args.output_cells,
                n_samples=args.n_samples,
            )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
