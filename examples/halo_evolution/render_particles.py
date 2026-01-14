#!/usr/bin/env python3
"""
Particle-based 3D Visualization for Halo Evolution

Renders dark matter particles as semi-transparent scatter points with
brightness based on local density, similar to observational visualizations.

Usage:
    python render_particles.py --branch-file branch_data.json \
        --snapshot-dir /path/to/snapshots --output-dir /path/to/output
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

# Import tessera (handles platform-specific configuration automatically)
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


def load_particles_in_box(
    snap_num: int,
    center: np.ndarray,
    snapshot_dir: Path,
    box_size: float = 1.5,
    max_particles: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load particles within a box centered on a position.

    Returns particles relative to the box center.
    """
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
    sim_box_size = snapshot.header.box_size

    # Apply periodic boundary conditions and select particles in box
    half_width = box_size / 2.0

    # Compute distances with periodic wrapping
    rel_pos = positions - center

    # Wrap to nearest image
    for i in range(3):
        rel_pos[:, i] = np.where(
            rel_pos[:, i] > sim_box_size / 2,
            rel_pos[:, i] - sim_box_size,
            rel_pos[:, i]
        )
        rel_pos[:, i] = np.where(
            rel_pos[:, i] < -sim_box_size / 2,
            rel_pos[:, i] + sim_box_size,
            rel_pos[:, i]
        )

    # Select particles within box
    in_box = (
        (np.abs(rel_pos[:, 0]) <= half_width) &
        (np.abs(rel_pos[:, 1]) <= half_width) &
        (np.abs(rel_pos[:, 2]) <= half_width)
    )

    selected_pos = rel_pos[in_box]

    # Subsample if too many particles
    if len(selected_pos) > max_particles:
        indices = np.random.choice(len(selected_pos), max_particles, replace=False)
        selected_pos = selected_pos[indices]

    return selected_pos, sim_box_size


def compute_local_density(
    positions: np.ndarray,
    n_neighbors: int = 32,
) -> np.ndarray:
    """
    Compute local density estimate using k-nearest neighbors.

    Density ~ n_neighbors / (4/3 * pi * r_k^3) where r_k is distance to kth neighbor.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=n_neighbors + 1)

    # Distance to nth neighbor (excluding self)
    r_n = distances[:, -1]

    # Avoid division by zero
    r_n = np.maximum(r_n, 1e-10)

    # Density proportional to 1/r^3
    density = n_neighbors / (4/3 * np.pi * r_n**3)

    return density


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


def render_particles_3d(
    positions: np.ndarray,
    densities: np.ndarray,
    output_path: Path,
    box_size: float,
    title: str = "Dark Matter",
    point_size: float = 0.5,
    alpha_scale: float = 0.3,
    cmap: str = 'Greys',
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    elev: float = 20,
    azim: float = -60,
    invert_colors: bool = True,
):
    """
    Render particles as 3D scatter plot with density-based coloring.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N, 3)
    densities : np.ndarray
        Local density at each particle
    output_path : Path
        Output image path
    box_size : float
        Box size in Mpc/h
    title : str
        Plot title
    point_size : float
        Base point size
    alpha_scale : float
        Maximum alpha value
    cmap : str
        Colormap name
    figsize : tuple
        Figure size in inches
    dpi : int
        Output resolution
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view
    invert_colors : bool
        If True, denser regions are darker (like the reference image)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    setup_latex_style()

    # Normalize densities to [0, 1]
    log_density = np.log10(densities + 1e-10)
    norm_density = (log_density - log_density.min()) / (log_density.max() - log_density.min() + 1e-10)

    # Compute alpha and color values
    if invert_colors:
        # Denser = darker and more opaque
        colors = 1 - norm_density  # Invert for grayscale (0=black, 1=white)
        alphas = norm_density * alpha_scale + 0.05  # Higher density = more opaque
    else:
        colors = norm_density
        alphas = norm_density * alpha_scale + 0.05

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Sort by density (render low density first, high density on top)
    sort_idx = np.argsort(densities)
    positions = positions[sort_idx]
    colors = colors[sort_idx]
    alphas = alphas[sort_idx]

    # Create RGBA colors
    cmap_obj = plt.get_cmap(cmap)
    rgba = cmap_obj(colors)
    rgba[:, 3] = alphas

    # Scatter plot
    half = box_size / 2
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=rgba,
        s=point_size,
        marker='.',
        depthshade=True,
    )

    # Set axis limits
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)

    # Labels with units
    ax.set_xlabel(r'$x$ [$h^{-1}$kpc]', labelpad=10)
    ax.set_ylabel(r'$y$ [$h^{-1}$kpc]', labelpad=10)
    ax.set_zlabel(r'$z$ [$h^{-1}$kpc]', labelpad=10)

    # Convert to kpc for axis labels
    tick_vals = np.linspace(-half, half, 5)
    tick_labels = [f'{int(v * 1000)}' for v in tick_vals]  # Mpc/h to kpc/h
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_zticks(tick_vals)
    ax.set_zticklabels(tick_labels)

    # Title
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
              fontweight='bold', va='top')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap if not invert_colors else cmap + '_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Relative Local Density', fontsize=11)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Adjust pane colors (make them lighter)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved particle rendering to {output_path}")


def create_rotation_animation(
    positions: np.ndarray,
    densities: np.ndarray,
    output_path: Path,
    box_size: float,
    title: str = "Dark Matter",
    n_frames: int = 36,
    fps: int = 10,
    point_size: float = 0.5,
    alpha_scale: float = 0.3,
    cmap: str = 'Greys',
):
    """Create a rotating animation of the particle visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D

    setup_latex_style()

    # Normalize densities
    log_density = np.log10(densities + 1e-10)
    norm_density = (log_density - log_density.min()) / (log_density.max() - log_density.min() + 1e-10)

    # Compute colors and alphas (inverted for grayscale)
    colors = 1 - norm_density
    alphas = norm_density * alpha_scale + 0.05

    # Sort by density
    sort_idx = np.argsort(densities)
    positions = positions[sort_idx]
    colors = colors[sort_idx]
    alphas = alphas[sort_idx]

    # Create RGBA
    cmap_obj = plt.get_cmap(cmap)
    rgba = cmap_obj(colors)
    rgba[:, 3] = alphas

    # Create figure
    fig = plt.figure(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    half = box_size / 2

    # Initial scatter
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=rgba,
        s=point_size,
        marker='.',
        depthshade=True,
    )

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)

    ax.set_xlabel(r'$x$ [$h^{-1}$kpc]', labelpad=10)
    ax.set_ylabel(r'$y$ [$h^{-1}$kpc]', labelpad=10)
    ax.set_zlabel(r'$z$ [$h^{-1}$kpc]', labelpad=10)

    # Convert to kpc
    tick_vals = np.linspace(-half, half, 5)
    tick_labels = [f'{int(v * 1000)}' for v in tick_vals]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_zticks(tick_vals)
    ax.set_zticklabels(tick_labels)

    ax.text2D(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
              fontweight='bold', va='top')

    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    def update(frame):
        azim = -60 + 360 * frame / n_frames
        ax.view_init(elev=20, azim=azim)
        return [scatter]

    print(f"Creating rotation animation ({n_frames} frames)...")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Saved animation to {output_path}")


def create_multipanel_figure(
    particle_data: Dict[float, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    box_size: float,
    epochs: List[float],
    point_size: float = 0.3,
    alpha_scale: float = 0.3,
    cmap: str = 'Greys',
):
    """Create a multi-panel figure showing evolution across epochs."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    setup_latex_style()

    # Find nearest available epochs
    available = sorted(particle_data.keys())
    epochs_to_plot = []
    for target in epochs:
        nearest = min(available, key=lambda x: abs(x - target))
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)

    fig = plt.figure(figsize=(5 * n_panels, 5), dpi=150)
    fig.patch.set_facecolor('white')

    half = box_size / 2

    for i, epoch in enumerate(epochs_to_plot):
        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        ax.set_facecolor('white')

        positions, densities = particle_data[epoch]

        # Normalize densities
        log_density = np.log10(densities + 1e-10)
        norm_density = (log_density - log_density.min()) / (log_density.max() - log_density.min() + 1e-10)

        colors = 1 - norm_density
        alphas = norm_density * alpha_scale + 0.05

        # Sort by density
        sort_idx = np.argsort(densities)
        positions = positions[sort_idx]
        colors = colors[sort_idx]
        alphas = alphas[sort_idx]

        # RGBA
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(colors)
        rgba[:, 3] = alphas

        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=rgba,
            s=point_size,
            marker='.',
            depthshade=True,
        )

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)

        ax.set_xlabel(r'$x$ [$h^{-1}$kpc]', labelpad=5, fontsize=9)
        ax.set_ylabel(r'$y$ [$h^{-1}$kpc]', labelpad=5, fontsize=9)
        ax.set_zlabel(r'$z$ [$h^{-1}$kpc]', labelpad=5, fontsize=9)

        # Tick labels
        tick_vals = np.linspace(-half, half, 3)
        tick_labels = [f'{int(v * 1000)}' for v in tick_vals]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_zticks(tick_vals)
        ax.set_zticklabels(tick_labels, fontsize=8)

        ax.set_title(f'$a = {epoch:.2f}$', fontsize=12, pad=10)

        ax.view_init(elev=20, azim=-60)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved multi-panel figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Particle-based 3D visualization for halo evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--branch-file', type=Path, required=True,
                        help='Branch data JSON file')
    parser.add_argument('--snapshot-dir', type=Path, required=True,
                        help='Directory containing snapshots')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory')

    # Epoch selection
    parser.add_argument('--epochs', type=float, nargs='+', default=[1.0, 10.0, 100.0],
                        help='Scale factors to render')
    parser.add_argument('--all-epochs', action='store_true',
                        help='Render all epochs in branch')

    # Box settings
    parser.add_argument('--box-size', type=float, default=1.5,
                        help='Box size in Mpc/h')
    parser.add_argument('--max-particles', type=int, default=50000,
                        help='Maximum particles to render')
    parser.add_argument('--n-neighbors', type=int, default=32,
                        help='Neighbors for density estimation')

    # Rendering settings
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Particle point size')
    parser.add_argument('--alpha-scale', type=float, default=0.3,
                        help='Maximum alpha value')
    parser.add_argument('--cmap', type=str, default='Greys',
                        help='Colormap')

    # Output options
    parser.add_argument('--multipanel', action='store_true',
                        help='Create multi-panel figure')
    parser.add_argument('--animation', action='store_true',
                        help='Create rotation animation')
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation FPS')
    parser.add_argument('--n-frames', type=int, default=36,
                        help='Animation frames')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PARTICLE-BASED 3D VISUALIZATION")
    print("=" * 60)
    print()

    # Load branch
    print(f"Loading branch from {args.branch_file}")
    branch, tree_box_size = load_branch(args.branch_file)
    print(f"  Loaded {len(branch)} halos")
    print()

    # Select epochs
    halo_lookup = {h.scale_factor: h for h in branch}
    available_epochs = sorted(halo_lookup.keys())

    if args.all_epochs:
        to_render = branch
    else:
        to_render = []
        for target in args.epochs:
            nearest = min(available_epochs, key=lambda x: abs(x - target))
            if nearest not in [h.scale_factor for h in to_render]:
                to_render.append(halo_lookup[nearest])

    print(f"Rendering {len(to_render)} epochs: {[f'a={h.scale_factor:.2f}' for h in to_render]}")
    print()

    # Load particles and compute densities
    particle_data = {}

    for i, halo in enumerate(to_render):
        print(f"  [{i+1}/{len(to_render)}] Loading particles for a={halo.scale_factor:.2f} (snap {halo.snap_num})...")

        positions, _ = load_particles_in_box(
            snap_num=halo.snap_num,
            center=halo.position,
            snapshot_dir=args.snapshot_dir,
            box_size=args.box_size,
            max_particles=args.max_particles,
        )

        print(f"      Loaded {len(positions)} particles")
        print(f"      Computing local density with {args.n_neighbors} neighbors...")

        densities = compute_local_density(positions, n_neighbors=args.n_neighbors)

        particle_data[halo.scale_factor] = (positions, densities)
        print(f"      Density range: [{densities.min():.2e}, {densities.max():.2e}]")

    print()

    # Create individual renderings
    print("Creating 3D particle renderings...")
    for epoch, (positions, densities) in particle_data.items():
        output_path = args.output_dir / f'halo_particles_a{epoch:.2f}.png'
        render_particles_3d(
            positions=positions,
            densities=densities,
            output_path=output_path,
            box_size=args.box_size,
            title=f'Dark Matter ($a = {epoch:.2f}$)',
            point_size=args.point_size,
            alpha_scale=args.alpha_scale,
            cmap=args.cmap,
        )

    # Multi-panel figure
    if args.multipanel:
        print()
        print("Creating multi-panel figure...")
        output_path = args.output_dir / 'halo_particles_evolution.png'
        create_multipanel_figure(
            particle_data=particle_data,
            output_path=output_path,
            box_size=args.box_size,
            epochs=args.epochs,
            point_size=args.point_size * 0.6,
            alpha_scale=args.alpha_scale,
            cmap=args.cmap,
        )

    # Rotation animations
    if args.animation:
        print()
        print("Creating rotation animations...")
        for epoch, (positions, densities) in particle_data.items():
            output_path = args.output_dir / f'halo_particles_rotation_a{epoch:.2f}.gif'
            create_rotation_animation(
                positions=positions,
                densities=densities,
                output_path=output_path,
                box_size=args.box_size,
                title=f'Dark Matter ($a = {epoch:.2f}$)',
                n_frames=args.n_frames,
                fps=args.fps,
                point_size=args.point_size,
                alpha_scale=args.alpha_scale,
                cmap=args.cmap,
            )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
