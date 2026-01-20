#!/usr/bin/env python3
"""
ORIGAMI Morphology Thin Slice Renderer

Creates publication-quality 2D slice visualizations of ORIGAMI morphology
classifications, similar to the "Dominant Morphology (z-slice)" panel.

Usage:
    python origami_slice_render.py --snapshot /path/to/snapshot.hdf5 --output slice.png
    python origami_slice_render.py --origami-file /path/to/origami_result.h5 --output slice.png

The colormap matches the standard ORIGAMI visualization:
- Void (0): Dark purple/blue
- Wall (1): Blue/cyan
- Filament (2): Green/yellow
- Halo (3): Yellow/bright
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Try to find the tessera module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

try:
    import _tessera as ts
except ImportError:
    try:
        import tessera as ts
    except ImportError:
        print("Error: tessera not found. Build the project first.")
        sys.exit(1)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Morphology class names and colors
MORPHOLOGY_NAMES = ['Void', 'Wall', 'Filament', 'Halo']

# Custom discrete colormap for morphology (matches viridis-style)
MORPHOLOGY_CMAP_COLORS = [
    '#440154',  # Void: dark purple
    '#31688e',  # Wall: blue
    '#35b779',  # Filament: green
    '#fde725',  # Halo: yellow
]


def create_morphology_cmap():
    """Create a discrete colormap for ORIGAMI morphology classes."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(MORPHOLOGY_CMAP_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot."""
    if not HAS_H5PY:
        snapshot = ts.io.read_gadget4_snapshot(
            str(snapshot_path),
            particle_types=(1 << particle_type),
            read_velocities=False,
            read_ids=True,
        )
        particles = snapshot.particles[particle_type]
        positions = np.array(particles.coordinates, dtype=np.float64)
        particle_ids = np.array(particles.particle_ids, dtype=np.int64)
        box_size = snapshot.header.box_size
        scale_factor = snapshot.header.time
        return positions, particle_ids, box_size, scale_factor

    with h5py.File(snapshot_path, 'r') as f:
        part_key = f'PartType{particle_type}'
        positions = np.ascontiguousarray(f[f'{part_key}/Coordinates'][:], dtype=np.float64)
        particle_ids = np.ascontiguousarray(f[f'{part_key}/ParticleIDs'][:], dtype=np.int64)
        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])

    return positions, particle_ids, box_size, scale_factor


def compute_origami_with_grid(positions, particle_ids, box_size, grid_cells=128):
    """Compute ORIGAMI morphology and deposit to grid using unified pipeline."""
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    # Use unified pipeline with grid output
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.grid_cells = grid_cells
    config.n_threads = 1
    config.n_split = 1

    print("  Computing ORIGAMI morphology...")
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    if result.is_linear_regime:
        print(f"  WARNING: Field is in LINEAR REGIME ({result.f_void*100:.1f}% void)")

    print(f"  Deposited to {grid_cells}^3 grid")

    return result, positions


def load_origami_result(origami_path):
    """Load pre-computed ORIGAMI results from HDF5."""
    with h5py.File(origami_path, 'r') as f:
        box_size = f['Header'].attrs['box_size']
        scale_factor = f['Header'].attrs['scale_factor']

        # Load grid data
        morphology_grid = f['Grid/morphology'][:]

        # Check for linear regime flag
        is_linear_regime = f['Header'].attrs.get('is_linear_regime', False)

    return morphology_grid, box_size, scale_factor, is_linear_regime


def render_morphology_slice(
    morphology_grid,
    box_size,
    output_path,
    slice_axis=0,
    slice_index=None,
    scale_factor=None,
    title=None,
    figsize=(8, 7),
    dpi=150,
    show_colorbar=True,
    colorbar_orientation='vertical',
):
    """
    Render a 2D slice of the ORIGAMI morphology grid.

    Parameters
    ----------
    morphology_grid : ndarray, shape (cells, cells, cells)
        3D morphology grid with values 0-3.
    box_size : float
        Physical box size for axis labels.
    output_path : str or Path
        Output file path.
    slice_axis : int
        Axis perpendicular to slice (0=z, 1=y, 2=x).
    slice_index : int, optional
        Index along slice_axis. Default: middle slice.
    scale_factor : float, optional
        Scale factor for title.
    title : str, optional
        Custom title. If None, auto-generated.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    show_colorbar : bool
        Whether to show the morphology colorbar.
    colorbar_orientation : str
        'vertical' or 'horizontal'.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Get slice
    grid_cells = morphology_grid.shape[0]
    if slice_index is None:
        slice_index = grid_cells // 2

    axis_names = ['z', 'y', 'x']
    slice_axis_name = axis_names[slice_axis]

    if slice_axis == 0:
        slice_data = morphology_grid[slice_index, :, :]
        xlabel, ylabel = 'x', 'y'
    elif slice_axis == 1:
        slice_data = morphology_grid[:, slice_index, :]
        xlabel, ylabel = 'x', 'z'
    else:
        slice_data = morphology_grid[:, :, slice_index]
        xlabel, ylabel = 'y', 'z'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create discrete colormap
    cmap, norm = create_morphology_cmap()

    # Plot
    im = ax.imshow(
        slice_data,
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[0, box_size, 0, box_size],
        interpolation='nearest',
    )

    # Labels
    ax.set_xlabel(f'{xlabel} [Mpc/h]', fontsize=12)
    ax.set_ylabel(f'{ylabel} [Mpc/h]', fontsize=12)

    # Title
    if title is None:
        if scale_factor is not None:
            redshift = 1.0 / scale_factor - 1.0
            if abs(redshift) < 0.01:
                title = f'Dominant Morphology ({slice_axis_name}-slice): z ≈ 0 (a = {scale_factor:.2f})'
            else:
                title = f'Dominant Morphology ({slice_axis_name}-slice): z = {redshift:.2f} (a = {scale_factor:.4f})'
        else:
            title = f'Dominant Morphology ({slice_axis_name}-slice)'
    ax.set_title(title, fontsize=14)

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation=colorbar_orientation,
                           ticks=[0, 1, 2, 3], pad=0.02)
        cbar.set_ticklabels(MORPHOLOGY_NAMES)
        cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved morphology slice to: {output_path}")


def render_morphology_slice_publication(
    morphology_grid,
    box_size,
    output_path,
    slice_axis=0,
    slice_index=None,
    scale_factor=None,
    figsize=(6, 5.5),
    dpi=300,
):
    """
    Render a publication-quality morphology slice (minimal styling).

    This version has cleaner styling suitable for papers/presentations.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Get slice
    grid_cells = morphology_grid.shape[0]
    if slice_index is None:
        slice_index = grid_cells // 2

    axis_names = ['z', 'y', 'x']
    slice_axis_name = axis_names[slice_axis]

    if slice_axis == 0:
        slice_data = morphology_grid[slice_index, :, :]
        xlabel, ylabel = 'x [Mpc/h]', 'y [Mpc/h]'
    elif slice_axis == 1:
        slice_data = morphology_grid[:, slice_index, :]
        xlabel, ylabel = 'x [Mpc/h]', 'z [Mpc/h]'
    else:
        slice_data = morphology_grid[:, :, slice_index]
        xlabel, ylabel = 'y [Mpc/h]', 'z [Mpc/h]'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create discrete colormap
    cmap, norm = create_morphology_cmap()

    # Plot
    im = ax.imshow(
        slice_data,
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[0, box_size, 0, box_size],
        interpolation='nearest',
    )

    # Labels
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)

    # Title
    if scale_factor is not None:
        redshift = 1.0 / scale_factor - 1.0
        if abs(redshift) < 0.01:
            title = f'Dominant Morphology ({slice_axis_name}-slice)'
        else:
            title = f'Dominant Morphology ({slice_axis_name}-slice)'
    else:
        title = f'Dominant Morphology ({slice_axis_name}-slice)'
    ax.set_title(title, fontsize=12, fontweight='medium')

    # Colorbar with class labels
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(MORPHOLOGY_NAMES, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved publication-quality morphology slice to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render ORIGAMI morphology thin slice visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--snapshot", type=Path,
                             help="Path to GADGET-4 HDF5 snapshot")
    input_group.add_argument("--origami-file", type=Path,
                             help="Path to pre-computed ORIGAMI HDF5 result")

    # Output
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output image path")

    # Slice options
    parser.add_argument("--slice-axis", type=int, default=0, choices=[0, 1, 2],
                        help="Axis perpendicular to slice (0=z, 1=y, 2=x)")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="Index along slice axis (default: middle)")

    # Grid options (for snapshot input)
    parser.add_argument("--grid-cells", type=int, default=128,
                        help="Grid cells for morphology deposition")

    # Style options
    parser.add_argument("--publication", action="store_true",
                        help="Use publication-quality styling")
    parser.add_argument("--figsize", type=float, nargs=2, default=None,
                        help="Figure size (width height) in inches")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom title")

    args = parser.parse_args()

    print("=" * 60)
    print("ORIGAMI MORPHOLOGY SLICE RENDERER")
    print("=" * 60)

    # Load data
    if args.snapshot:
        print(f"\nLoading snapshot: {args.snapshot}")
        positions, particle_ids, box_size, scale_factor = load_snapshot(args.snapshot)

        grid_size = int(round(len(positions) ** (1/3)))
        print(f"  Box size: {box_size} Mpc/h")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Particles: {len(positions)} ({grid_size}^3)")

        result, _ = compute_origami_with_grid(
            positions, particle_ids, box_size, args.grid_cells
        )
        morphology_grid = np.array(result.morphology_grid)

    else:  # --origami-file
        print(f"\nLoading ORIGAMI result: {args.origami_file}")
        morphology_grid, box_size, scale_factor, is_linear_regime = load_origami_result(args.origami_file)
        print(f"  Box size: {box_size} Mpc/h")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Grid: {morphology_grid.shape[0]}^3")
        if is_linear_regime:
            print("  WARNING: This snapshot is in the LINEAR REGIME")

    # Render
    print(f"\nRendering slice...")

    figsize = tuple(args.figsize) if args.figsize else None

    if args.publication:
        render_morphology_slice_publication(
            morphology_grid,
            box_size,
            args.output,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
            scale_factor=scale_factor,
            figsize=figsize or (6, 5.5),
            dpi=args.dpi or 300,
        )
    else:
        render_morphology_slice(
            morphology_grid,
            box_size,
            args.output,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
            scale_factor=scale_factor,
            title=args.title,
            figsize=figsize or (8, 7),
            dpi=args.dpi,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
