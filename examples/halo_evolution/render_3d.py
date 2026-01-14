#!/usr/bin/env python3
"""
3D Density Rendering for Halo Evolution

Renders 3D density fields around traced halos using volume rendering
or isosurface visualization.

Requires: pip install pyvista
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
    _script_dir = Path(__file__).parent.resolve()
    _repo_root = _script_dir.parent.parent
    _build_dir = _repo_root / 'build'
    if _build_dir.exists():
        sys.path.insert(0, str(_build_dir))
    import _tessera as ts

from halo_evolution.visualize_3d import (
    Render3DConfig,
    render_3d,
    create_multiview_figure,
    create_rotation_animation,
    create_evolution_3d_figure,
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


def render_density_3d(
    snap_num: int,
    halo_position: np.ndarray,
    snapshot_dir: Path,
    box_size: float = 1.5,
    output_cells: int = 128,
    n_samples: int = 50,
) -> tuple:
    """Render 3D density field around a halo position."""

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

    # Configure tessera for 3D
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

    # Compute 3D density
    result = ts.density.compute_tetra_density_3d(sorted_positions, config)
    density = np.array(result.density).reshape(
        output_cells, output_cells, output_cells
    )

    return density, sim_box_size


def main():
    parser = argparse.ArgumentParser(
        description='3D density rendering for halo evolution',
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

    # Resolution
    parser.add_argument('--box-size', type=float, default=1.5,
                        help='Box size in Mpc/h')
    parser.add_argument('--output-cells', type=int, default=128,
                        help='3D grid resolution')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Monte Carlo samples per tetrahedron')

    # Rendering mode
    parser.add_argument('--mode', type=str, default='isosurface',
                        choices=['volume', 'isosurface', 'both'],
                        help='Rendering mode')

    # Isosurface settings
    parser.add_argument('--iso-levels', type=float, nargs='+', default=[10, 100, 1000],
                        help='Overdensity levels for isosurfaces')

    # Output options
    parser.add_argument('--multiview', action='store_true',
                        help='Create multi-view figure')
    parser.add_argument('--rotation', action='store_true',
                        help='Create rotation animation')
    parser.add_argument('--evolution', action='store_true',
                        help='Create evolution figure')

    # Visualization settings
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap for volume rendering')
    parser.add_argument('--background', type=str, default='white',
                        help='Background color')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3D DENSITY RENDERING")
    print("=" * 60)
    print()

    # Load branch
    print(f"Loading branch from {args.branch_file}")
    branch, tree_box_size = load_branch(args.branch_file)
    print(f"  Loaded {len(branch)} halos")
    print()

    # Select epochs to render
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

    # Configure rendering
    render_config = Render3DConfig(
        mode=args.mode,
        cmap=args.cmap,
        background=args.background,
        isosurface_levels=args.iso_levels,
        isosurface_opacity=[0.3, 0.5, 0.8][:len(args.iso_levels)],
        isosurface_colors=['blue', 'green', 'yellow', 'red'][:len(args.iso_levels)],
    )

    # Render 3D densities
    densities = {}
    sim_box_size = None

    for i, halo in enumerate(to_render):
        print(f"  [{i+1}/{len(to_render)}] Computing 3D density for a={halo.scale_factor:.2f} (snap {halo.snap_num})...")
        density, sim_box_size = render_density_3d(
            snap_num=halo.snap_num,
            halo_position=halo.position,
            snapshot_dir=args.snapshot_dir,
            box_size=args.box_size,
            output_cells=args.output_cells,
            n_samples=args.n_samples,
        )
        densities[halo.scale_factor] = density
        print(f"      Shape: {density.shape}, range: [{density.min():.2e}, {density.max():.2e}]")

    print()

    # Compute mean density
    n_particles = 256 ** 3
    mean_density = n_particles / (sim_box_size ** 3)

    # Create individual renderings
    print("Creating 3D renderings...")
    for epoch, density in densities.items():
        output_path = args.output_dir / f'halo_3d_a{epoch:.2f}.png'
        render_3d(
            density=density,
            output_path=output_path,
            box_size=args.box_size,
            mean_density=mean_density,
            config=render_config,
            title=f'$a = {epoch:.2f}$',
        )

    # Multi-view figure
    if args.multiview:
        print()
        print("Creating multi-view figures...")
        for epoch, density in densities.items():
            output_path = args.output_dir / f'halo_3d_multiview_a{epoch:.2f}.png'
            create_multiview_figure(
                density=density,
                output_path=output_path,
                box_size=args.box_size,
                mean_density=mean_density,
                config=render_config,
                title=f'$a = {epoch:.2f}$',
            )

    # Rotation animation
    if args.rotation:
        print()
        print("Creating rotation animations...")
        for epoch, density in densities.items():
            output_path = args.output_dir / f'halo_3d_rotation_a{epoch:.2f}.gif'
            create_rotation_animation(
                density=density,
                output_path=output_path,
                box_size=args.box_size,
                mean_density=mean_density,
                config=render_config,
                title=f'$a = {epoch:.2f}$',
            )

    # Evolution figure
    if args.evolution and len(densities) > 1:
        print()
        print("Creating 3D evolution figure...")
        output_path = args.output_dir / 'halo_3d_evolution.png'
        create_evolution_3d_figure(
            densities=densities,
            output_path=output_path,
            box_size=args.box_size,
            mean_density=mean_density,
            epochs=args.epochs,
            config=render_config,
        )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
