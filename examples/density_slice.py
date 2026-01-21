#!/usr/bin/env python3
"""
Example: Creating a Projected Density Slice from a GADGET-4 Snapshot

This script demonstrates how to:
1. Read a GADGET-4 snapshot (single file or distributed)
2. Compute a 2D projected density field using phase-space tessellation
3. Save the density field to an HDF5 file
4. Optionally generate a visualization

By default, this uses the gotetra phase-space tessellation method:
- Particles are connected into tetrahedra based on their Lagrangian IDs
- Monte Carlo sampling within tetrahedra deposits mass onto a grid
- A 2D slice is extracted by summing along the projection axis

This correctly handles stream crossing (caustics) in the density field.

Alternatively, use --histogram for a simple 2D particle histogram (faster
but does not handle caustics correctly).

Reference: gotetra (https://github.com/phil-mansfield/gotetra)
"""

import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared utilities
from utils import (
    load_snapshot,
    sort_positions_lagrangian,
    infer_grid_size,
    compute_mean_density,
    get_axis_labels,
    check_tessera_available,
)

# Import tessera
try:
    import tessera as ts
except ImportError:
    import _tessera as ts


def compute_density_slice_histogram(
    positions,
    box_size,
    slice_center,
    slice_thickness,
    projection_axis='z',
    grid_resolution=1024,
    mass_per_particle=1.0
):
    """
    Compute a 2D projected density field using simple histogram binning.

    This is a fast but approximate method that does NOT handle stream
    crossing (caustics) correctly. For accurate density fields, use
    compute_density_slice_tessellation() instead.
    """
    # Map axis name to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]

    # Get the two axes we'll histogram over
    plane_axes = [i for i in range(3) if i != proj_axis]

    # Select particles within the slice (with periodic boundary handling)
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2

    pos_along_axis = positions[:, proj_axis]

    if slice_min < 0:
        in_slice = (pos_along_axis >= (slice_min + box_size)) | (pos_along_axis < slice_max)
    elif slice_max > box_size:
        in_slice = (pos_along_axis >= slice_min) | (pos_along_axis < (slice_max - box_size))
    else:
        in_slice = (pos_along_axis >= slice_min) & (pos_along_axis < slice_max)

    pos_in_slice = positions[in_slice]
    x_coords = pos_in_slice[:, plane_axes[0]]
    y_coords = pos_in_slice[:, plane_axes[1]]

    print(f"Selected {len(pos_in_slice):,} particles in slice "
          f"({100*len(pos_in_slice)/len(positions):.1f}% of total)")

    # Create 2D histogram
    bins = np.linspace(0, box_size, grid_resolution + 1)
    density, x_edges, y_edges = np.histogram2d(
        x_coords, y_coords,
        bins=[bins, bins],
        weights=np.ones(len(x_coords)) * mass_per_particle
    )

    # Convert to surface density (mass per unit area)
    pixel_area = (box_size / grid_resolution) ** 2
    density = density / pixel_area

    extent = (0, box_size, 0, box_size)

    # Compute mean background surface density
    mean_3d_density = compute_mean_density(len(positions), mass_per_particle, box_size)
    mean_surface_density = mean_3d_density * slice_thickness

    slice_info = {
        'center': slice_center,
        'thickness': slice_thickness,
        'projection_axis': projection_axis,
        'n_particles_in_slice': len(pos_in_slice),
        'n_particles_total': len(positions),
        'grid_resolution': grid_resolution,
        'pixel_size': box_size / grid_resolution,
        'mean_surface_density': mean_surface_density,
        'mean_3d_density': mean_3d_density,
        'method': 'histogram'
    }

    return density, extent, slice_info


def compute_density_slice_tessellation(
    positions,
    particle_ids,
    box_size,
    slice_center,
    slice_thickness,
    projection_axis='z',
    grid_resolution=256,
    n_samples=100,
    mass_per_particle=1.0,
    n_threads=0,
    use_direct_2d=True
):
    """
    Compute a 2D projected density field using phase-space tessellation.

    This is the gotetra method that correctly handles stream crossing
    (caustics) by connecting particles into tetrahedra based on their
    Lagrangian IDs and using Monte Carlo sampling within each tetrahedron.

    Parameters
    ----------
    use_direct_2d : bool
        If True, use memory-efficient direct 2D computation.
        If False, compute full 3D field first then extract slice.
    """
    if not check_tessera_available():
        raise ImportError("C++ tessellation module not available.")

    # Infer Lagrangian grid size
    grid_size = infer_grid_size(len(positions))

    # Sort particles by Lagrangian ID
    print("Sorting particles by Lagrangian ID...")
    sorted_positions = sort_positions_lagrangian(positions, particle_ids, grid_size)

    # Set up tessellation config
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]

    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2

    # Create config
    config = ts.density.TetraDensityConfig()
    config.box_size = box_size
    config.lagrangian_grid_size = grid_size
    config.output_cells = grid_resolution
    config.n_samples = n_samples
    config.particle_mass = mass_per_particle
    config.n_threads = n_threads

    if use_direct_2d:
        # Memory-efficient direct 2D computation
        print(f"Computing 2D slice density directly ({grid_resolution}x{grid_resolution})...")
        result = ts.density.compute_tetra_density_2d_direct(
            sorted_positions, config, proj_axis, slice_min, slice_max
        )
        density_2d = np.array(result.density)
        cell_width = result.cell_width
        actual_thickness = slice_max - slice_min
    else:
        # Full 3D computation then slice extraction
        print(f"Computing 3D tessellation density field ({grid_resolution}^3)...")
        result = ts.density.compute_tetra_density_3d(sorted_positions, config)
        density_3d = np.array(result.density).reshape(
            (grid_resolution, grid_resolution, grid_resolution)
        )

        print(f"  Tetrahedra processed: {result.n_tetrahedra:,}")

        # Extract 2D slice
        cell_width = box_size / grid_resolution
        slice_start_idx = int(slice_min / cell_width)
        slice_end_idx = int(np.ceil(slice_max / cell_width))
        slice_start_idx = max(0, slice_start_idx)
        slice_end_idx = min(grid_resolution, slice_end_idx)

        print(f"Extracting slice from cells {slice_start_idx} to {slice_end_idx}...")

        # Sum density along projection axis
        if proj_axis == 0:  # x
            density_2d = density_3d[:, :, slice_start_idx:slice_end_idx].sum(axis=2) * cell_width
        elif proj_axis == 1:  # y
            density_2d = density_3d[:, slice_start_idx:slice_end_idx, :].sum(axis=1) * cell_width
        else:  # z
            density_2d = density_3d[slice_start_idx:slice_end_idx, :, :].sum(axis=0) * cell_width

        actual_thickness = (slice_end_idx - slice_start_idx) * cell_width

    extent = (0, box_size, 0, box_size)

    # Compute mean surface density
    mean_3d_density = compute_mean_density(len(positions), mass_per_particle, box_size)
    mean_surface_density = mean_3d_density * actual_thickness

    slice_info = {
        'center': slice_center,
        'thickness': actual_thickness,
        'projection_axis': projection_axis,
        'n_particles_total': len(positions),
        'grid_resolution': grid_resolution,
        'pixel_size': cell_width,
        'mean_surface_density': mean_surface_density,
        'mean_3d_density': mean_3d_density,
        'n_samples': n_samples,
        'method': 'tessellation_direct_2d' if use_direct_2d else 'tessellation_3d'
    }

    return density_2d, extent, slice_info


def save_density_hdf5(filename, density, header, slice_info, extent):
    """Save the density field to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('density', data=density, compression='gzip')
        dset.attrs['units'] = 'mass / area (code units)'

        hdr = f.create_group('Header')
        hdr.attrs['BoxSize'] = header.box_size
        hdr.attrs['Redshift'] = header.redshift
        hdr.attrs['Time'] = header.time
        hdr.attrs['NumPartTotal'] = list(header.num_part_total)
        hdr.attrs['MassTable'] = list(header.mass_table)

        slc = f.create_group('SliceInfo')
        for key, val in slice_info.items():
            slc.attrs[key] = val

        grid = f.create_group('Grid')
        grid.attrs['extent'] = extent
        grid.attrs['x_min'] = extent[0]
        grid.attrs['x_max'] = extent[1]
        grid.attrs['y_min'] = extent[2]
        grid.attrs['y_max'] = extent[3]

    print(f"Saved density field to {filename}")


def plot_density_slice(
    density,
    extent,
    slice_info,
    header,
    output_file=None,
    cmap='mako',
    log_scale=True,
    vmin_percentile=1,
    vmax_percentile=99.9,
    overdensity=True,
    vmin=None,
    vmax=None
):
    """Create a visualization of the density slice."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        try:
            import seaborn as sns
        except ImportError:
            pass
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    plot_data = density.T  # Transpose for correct orientation

    if overdensity:
        mean_density = slice_info['mean_surface_density']
        plot_data = plot_data / mean_density
        scale_factor = header.time
        density_label = rf'$1 + \delta(a={scale_factor:.2f})$'
    else:
        density_label = 'Surface Density'

    if log_scale:
        plot_data = np.maximum(plot_data, plot_data[plot_data > 0].min() * 0.1)
        vmin_val = vmin if vmin is not None else np.percentile(plot_data[plot_data > 0], vmin_percentile)
        vmax_val = vmax if vmax is not None else np.percentile(plot_data, vmax_percentile)
        norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
    else:
        vmin_val = vmin if vmin is not None else np.percentile(plot_data, vmin_percentile)
        vmax_val = vmax if vmax is not None else np.percentile(plot_data, vmax_percentile)
        norm = None
        plot_data = np.clip(plot_data, vmin_val, vmax_val)

    im = ax.imshow(
        plot_data,
        extent=extent,
        origin='lower',
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )

    # Labels and title
    ax1, ax2 = get_axis_labels(slice_info['projection_axis'])
    ax.set_xlabel(f'{ax1} [code units]', fontsize=12)
    ax.set_ylabel(f'{ax2} [code units]', fontsize=12)

    title = (f"Density Projection along {slice_info['projection_axis'].upper()}\n"
             f"z = {header.redshift:.2f}, "
             f"Slice: {slice_info['center']:.1f} ± {slice_info['thickness']/2:.1f}")
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(density_label + (' (log scale)' if log_scale else ''), fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create a projected density slice from a GADGET-4 snapshot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with tessellation (default)
  python density_slice.py snapshot_034.hdf5 -o density.h5

  # Use simple histogram method (faster, but less accurate)
  python density_slice.py snapshot_034.hdf5 -o density.h5 --histogram

  # Custom slice with visualization
  python density_slice.py snapshot.hdf5 -o density.h5 --center 128 --thickness 25 --plot density.png

  # Higher resolution tessellation with more Monte Carlo samples
  python density_slice.py snapshot.hdf5 -o density.h5 --resolution 512 --samples 200

  # Use full 3D computation (more memory, same result)
  python density_slice.py snapshot.hdf5 -o density.h5 --no-direct-2d
        """
    )

    parser.add_argument('snapshot', type=str,
                        help='Path to snapshot file or directory')
    parser.add_argument('-n', '--snapshot-num', type=int, default=None,
                        help='Snapshot number (for distributed files)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 filename')
    parser.add_argument('--center', type=float, default=None,
                        help='Slice center (default: box center)')
    parser.add_argument('--thickness', type=float, default=None,
                        help='Slice thickness (default: 10%% of box)')
    parser.add_argument('--axis', type=str, default='z', choices=['x', 'y', 'z'],
                        help='Projection axis (default: z)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution (default: 256 for tessellation, 1024 for histogram)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output image filename (optional)')
    parser.add_argument('--cmap', type=str, default='mako',
                        help='Colormap for visualization (default: mako)')
    parser.add_argument('--use_surface_density', action='store_true',
                        help='Plot raw surface density instead of 1+delta (density/mean)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colorbar (overrides percentile)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colorbar (overrides percentile)')

    # Method selection
    parser.add_argument('--histogram', action='store_true',
                        help='Use simple 2D histogram instead of tessellation (faster but less accurate)')
    parser.add_argument('--no-direct-2d', action='store_true',
                        help='Use full 3D computation instead of memory-efficient direct 2D')

    # Tessellation-specific options
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (tessellation only, default: 100)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto, tessellation only)')

    args = parser.parse_args()

    # Determine method
    use_tessellation = not args.histogram
    if use_tessellation and not check_tessera_available():
        print("Warning: C++ tessellation module not available, falling back to histogram method.")
        use_tessellation = False

    # Adjust default resolution based on method
    if args.resolution == 256 and args.histogram:
        args.resolution = 1024

    # Load snapshot
    print(f"Loading snapshot from {args.snapshot}...")
    positions, particle_ids, header = load_snapshot(
        args.snapshot,
        snapshot_num=args.snapshot_num,
        particle_type=args.particle_type,
        read_ids=use_tessellation
    )

    box_size = header.box_size
    print(f"Box size: {box_size}")
    print(f"Redshift: {header.redshift:.4f}")
    print(f"Loaded {len(positions):,} particles")

    # Set defaults based on box size
    slice_center = args.center if args.center is not None else box_size / 2
    slice_thickness = args.thickness if args.thickness is not None else box_size * 0.1

    # Get particle mass from header
    mass_per_particle = header.mass_table[args.particle_type]
    if mass_per_particle == 0:
        mass_per_particle = 1.0

    print(f"\nSlice configuration:")
    print(f"  Center: {slice_center}")
    print(f"  Thickness: {slice_thickness}")
    print(f"  Projection axis: {args.axis}")
    print(f"  Grid resolution: {args.resolution}")
    print(f"  Particle mass: {mass_per_particle:.6e}")
    print(f"  Method: {'tessellation' if use_tessellation else 'histogram'}")
    if use_tessellation:
        print(f"  Monte Carlo samples: {args.samples}")
        print(f"  Direct 2D mode: {not args.no_direct_2d}")

    # Compute density slice
    print("\nComputing density slice...")
    if use_tessellation:
        density, extent, slice_info = compute_density_slice_tessellation(
            positions,
            particle_ids,
            box_size,
            slice_center,
            slice_thickness,
            projection_axis=args.axis,
            grid_resolution=args.resolution,
            n_samples=args.samples,
            mass_per_particle=mass_per_particle,
            n_threads=args.threads,
            use_direct_2d=not args.no_direct_2d
        )
    else:
        density, extent, slice_info = compute_density_slice_histogram(
            positions,
            box_size,
            slice_center,
            slice_thickness,
            projection_axis=args.axis,
            grid_resolution=args.resolution,
            mass_per_particle=mass_per_particle
        )

    print(f"\nDensity statistics:")
    print(f"  Min: {density.min():.6e}")
    print(f"  Max: {density.max():.6e}")
    print(f"  Mean: {density.mean():.6e}")
    print(f"  Mean background: {slice_info['mean_surface_density']:.6e}")

    # Print overdensity stats
    if slice_info['mean_surface_density'] > 0:
        overdensity_field = density / slice_info['mean_surface_density']
        print("\nOverdensity (1+delta) statistics:")
        print(f"  Min: {overdensity_field.min():.4f}")
        print(f"  Max: {overdensity_field.max():.4f}")
        print(f"  Mean: {overdensity_field.mean():.4f}")

    # Save to HDF5
    save_density_hdf5(args.output, density, header, slice_info, extent)

    # Optional visualization
    if args.plot:
        print(f"\nGenerating visualization...")
        plot_density_slice(
            density, extent, slice_info, header,
            output_file=args.plot,
            cmap=args.cmap,
            overdensity=not args.use_surface_density,
            vmin=args.vmin,
            vmax=args.vmax
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
