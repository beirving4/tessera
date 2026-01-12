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
- Monte Carlo sampling within tetrahedra deposits mass onto a 3D grid
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

# Try to find the asymptotic_tetra module
# First check if we're in the examples directory and add build to path
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

# Try to import C++ module for tessellation
try:
    import _asymptotic_tetra as at
    HAS_CPP = True
except ImportError:
    try:
        import asymptotic_tetra as at
        HAS_CPP = hasattr(at, 'density')
    except ImportError:
        HAS_CPP = False
        at = None


def load_snapshot(path, snapshot_num=None, particle_type=1, read_ids=False):
    """
    Load particle positions from a GADGET-4 snapshot.
    
    Parameters
    ----------
    path : str
        Path to snapshot file or directory containing distributed files
    snapshot_num : int, optional
        Snapshot number for distributed files (auto-detected if None)
    particle_type : int
        Particle type to load (default: 1 for dark matter)
    read_ids : bool
        Whether to read particle IDs (needed for tessellation)
    
    Returns
    -------
    positions : ndarray, shape (N, 3)
        Particle positions
    particle_ids : ndarray or None
        Particle IDs if read_ids=True, else None
    header : Gadget4Header
        Snapshot header with box size, redshift, etc.
    """
    # Try importing the module in various ways depending on how it's installed
    try:
        # Direct import of compiled module (when build dir is in PYTHONPATH)
        import _asymptotic_tetra
        io = _asymptotic_tetra.io
    except ImportError:
        try:
            # Installed package
            import asymptotic_tetra as _at
            io = _at.io
        except (ImportError, AttributeError):
            raise ImportError(
                "Could not import asymptotic_tetra. Make sure the build directory "
                "is in your PYTHONPATH or the package is installed."
            )
    
    # Use unified interface - auto-detects single file vs distributed
    particle_types = 1 << particle_type  # Bitmask for particle type
    
    if snapshot_num is not None:
        snap = io.read_gadget4_snapshot(
            path, 
            snapshot_num=snapshot_num,
            particle_types=particle_types,
            read_velocities=False,
            read_ids=read_ids
        )
    else:
        snap = io.read_gadget4_snapshot(
            path,
            particle_types=particle_types,
            read_velocities=False,
            read_ids=read_ids
        )
    
    # Get positions as numpy array
    coords = snap.particles[particle_type].coordinates
    positions = np.array(coords, dtype=np.float64)
    
    # Get particle IDs if requested
    particle_ids = None
    if read_ids:
        ids = snap.particles[particle_type].particle_ids
        particle_ids = np.array(ids, dtype=np.int64)
    
    return positions, particle_ids, snap.header


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
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions
    box_size : float
        Size of the simulation box
    slice_center : float
        Center of the slice along the projection axis
    slice_thickness : float
        Thickness of the slice
    projection_axis : str
        Axis to project along ('x', 'y', or 'z')
    grid_resolution : int
        Number of pixels per dimension in the output grid
    mass_per_particle : float
        Mass of each particle (for proper density normalization)
    
    Returns
    -------
    density : ndarray, shape (grid_resolution, grid_resolution)
        2D projected density field (mass per unit area)
    extent : tuple
        (x_min, x_max, y_min, y_max) for plotting
    slice_info : dict
        Information about the slice
    """
    # Map axis name to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]
    
    # Get the two axes we'll histogram over
    # For z projection: use x and y (axes 0 and 1)
    plane_axes = [i for i in range(3) if i != proj_axis]
    
    # Select particles within the slice (with periodic boundary handling)
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2
    
    # Handle periodic boundaries
    pos_along_axis = positions[:, proj_axis]
    
    if slice_min < 0:
        # Slice wraps around the lower boundary
        in_slice = (pos_along_axis >= (slice_min + box_size)) | (pos_along_axis < slice_max)
    elif slice_max > box_size:
        # Slice wraps around the upper boundary
        in_slice = (pos_along_axis >= slice_min) | (pos_along_axis < (slice_max - box_size))
    else:
        # Normal case - no wrapping
        in_slice = (pos_along_axis >= slice_min) & (pos_along_axis < slice_max)
    
    # Get positions in the plane
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
    
    # Create extent for plotting
    extent = (0, box_size, 0, box_size)
    
    # Compute mean background surface density for this slice
    total_mass = len(positions) * mass_per_particle
    mean_3d_density = total_mass / (box_size ** 3)
    mean_surface_density = mean_3d_density * slice_thickness
    
    # Slice info for metadata
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
    n_threads=0
):
    """
    Compute a 2D projected density field using phase-space tessellation.
    
    This is the gotetra method that correctly handles stream crossing
    (caustics) by connecting particles into tetrahedra based on their
    Lagrangian IDs and using Monte Carlo sampling within each tetrahedron.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions (will be sorted by Lagrangian ID)
    particle_ids : ndarray, shape (N,)
        Particle IDs encoding Lagrangian grid positions
    box_size : float
        Size of the simulation box
    slice_center : float
        Center of the slice along the projection axis
    slice_thickness : float
        Thickness of the slice
    projection_axis : str
        Axis to project along ('x', 'y', or 'z')
    grid_resolution : int
        Number of cells per dimension in the 3D grid
    n_samples : int
        Monte Carlo samples per tetrahedron
    mass_per_particle : float
        Mass of each particle
    n_threads : int
        Number of threads (0 = auto)
    
    Returns
    -------
    density : ndarray, shape (grid_resolution, grid_resolution)
        2D projected density field (mass per unit area)
    extent : tuple
        (x_min, x_max, y_min, y_max) for plotting
    slice_info : dict
        Information about the slice
    """
    if not HAS_CPP:
        raise ImportError(
            "C++ tessellation module not available. "
            "Use --histogram for simple histogram method."
        )
    
    # Infer Lagrangian grid size from particle count
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))
    if grid_size ** 3 != n_particles:
        raise ValueError(
            f"Particle count {n_particles} is not a perfect cube. "
            f"Tessellation requires N^3 particles."
        )
    
    # Sort particles by Lagrangian ID
    print("Sorting particles by Lagrangian ID...")
    sorted_positions = at.density.sort_by_lagrangian_id(
        positions, particle_ids, grid_size, id_offset=1
    )
    
    # Set up tessellation config
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]
    
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2
    
    # Create config for 3D density computation
    config = at.density.TetraDensityConfig()
    config.box_size = box_size
    config.lagrangian_grid_size = grid_size
    config.output_cells = grid_resolution
    config.n_samples = n_samples
    config.particle_mass = mass_per_particle
    config.n_threads = n_threads
    
    # Compute 3D density field
    print(f"Computing 3D tessellation density field ({grid_resolution}^3)...")
    result = at.density.compute_tetra_density_3d(sorted_positions, config)
    # C++ uses idx = x + y*N + z*N*N (x varies fastest)
    # With NumPy C-order reshape, this gives density[z, y, x]
    density_3d = np.array(result.density).reshape(
        (grid_resolution, grid_resolution, grid_resolution)
    )
    
    print(f"  Tetrahedra processed: {result.n_tetrahedra:,}")
    
    # Extract 2D slice by summing along projection axis
    cell_width = box_size / grid_resolution
    slice_start_idx = int(slice_min / cell_width)
    slice_end_idx = int(np.ceil(slice_max / cell_width))
    
    # Clamp indices
    slice_start_idx = max(0, slice_start_idx)
    slice_end_idx = min(grid_resolution, slice_end_idx)
    
    print(f"Extracting slice from cells {slice_start_idx} to {slice_end_idx}...")
    
    # Sum density along projection axis and multiply by cell width to get surface density
    # Note: C++ uses idx = x + y*N + z*N*N, so after reshape the array is density[z, y, x]
    if proj_axis == 0:  # x - slice and sum on last axis (x)
        density_2d = density_3d[:, :, slice_start_idx:slice_end_idx].sum(axis=2) * cell_width
    elif proj_axis == 1:  # y - slice and sum on middle axis (y)
        density_2d = density_3d[:, slice_start_idx:slice_end_idx, :].sum(axis=1) * cell_width
    else:  # z - slice and sum on first axis (z)
        density_2d = density_3d[slice_start_idx:slice_end_idx, :, :].sum(axis=0) * cell_width
    
    extent = (0, box_size, 0, box_size)
    
    # Compute mean surface density
    total_mass = n_particles * mass_per_particle
    mean_3d_density = total_mass / (box_size ** 3)
    actual_thickness = (slice_end_idx - slice_start_idx) * cell_width
    mean_surface_density = mean_3d_density * actual_thickness
    
    slice_info = {
        'center': slice_center,
        'thickness': actual_thickness,
        'projection_axis': projection_axis,
        'n_particles_total': n_particles,
        'grid_resolution': grid_resolution,
        'pixel_size': cell_width,
        'mean_surface_density': mean_surface_density,
        'mean_3d_density': mean_3d_density,
        'n_samples': n_samples,
        'n_tetrahedra': result.n_tetrahedra,
        'method': 'tessellation'
    }
    
    return density_2d, extent, slice_info


def save_density_hdf5(
    filename,
    density,
    header,
    slice_info,
    extent
):
    """
    Save the density field to an HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output filename
    density : ndarray
        2D density array
    header : Gadget4Header
        Snapshot header with cosmological info
    slice_info : dict
        Information about the slice
    extent : tuple
        Spatial extent of the grid
    """
    with h5py.File(filename, 'w') as f:
        # Store the density field
        dset = f.create_dataset('density', data=density, compression='gzip')
        dset.attrs['units'] = 'mass / area (code units)'
        
        # Store header/cosmology info
        hdr = f.create_group('Header')
        hdr.attrs['BoxSize'] = header.box_size
        hdr.attrs['Redshift'] = header.redshift
        hdr.attrs['Time'] = header.time
        hdr.attrs['NumPartTotal'] = list(header.num_part_total)
        hdr.attrs['MassTable'] = list(header.mass_table)
        
        # Store slice info
        slc = f.create_group('SliceInfo')
        for key, val in slice_info.items():
            slc.attrs[key] = val
        
        # Store grid extent
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
    """
    Create a visualization of the density slice.
    
    Parameters
    ----------
    density : ndarray
        2D density array (surface density)
    extent : tuple
        Spatial extent for plotting
    slice_info : dict
        Slice information for title
    header : Gadget4Header
        Snapshot header
    output_file : str, optional
        If provided, save figure to this file
    cmap : str
        Colormap name
    log_scale : bool
        If True, plot log10(density)
    vmin_percentile, vmax_percentile : float
        Percentiles for color scaling (used if vmin/vmax not specified)
    overdensity : bool
        If True, plot 1+delta (density/mean_density) instead of raw density
    vmin, vmax : float, optional
        Explicit min/max values for colorbar (overrides percentiles)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        # Import seaborn to register additional colormaps (mako, rocket, etc.)
        try:
            import seaborn as sns
        except ImportError:
            pass
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Handle log scale
    plot_data = density.T  # Transpose for correct orientation
    
    # Convert to overdensity (1 + delta) if requested
    if overdensity:
        mean_density = slice_info['mean_surface_density']
        plot_data = plot_data / mean_density  # This gives 1 + delta
        scale_factor = header.time
        density_label = rf'$1 + \delta(a={scale_factor:.2f})$'
    else:
        density_label = 'Surface Density'
    
    if log_scale:
        # Avoid log(0)
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
    proj_axis = slice_info['projection_axis'].upper()
    other_axes = {'X': ('Y', 'Z'), 'Y': ('X', 'Z'), 'Z': ('X', 'Y')}
    ax1, ax2 = other_axes[proj_axis]
    
    ax.set_xlabel(f'{ax1} [code units]', fontsize=12)
    ax.set_ylabel(f'{ax2} [code units]', fontsize=12)
    
    title = (f"Density Projection along {proj_axis}\n"
             f"z = {header.redshift:.2f}, "
             f"Slice: {slice_info['center']:.1f} ± {slice_info['thickness']/2:.1f}")
    ax.set_title(title, fontsize=14)
    
    # Colorbar
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
    
    # Tessellation-specific options
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (tessellation only, default: 100)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto, tessellation only)')
    
    args = parser.parse_args()
    
    # Determine method
    use_tessellation = not args.histogram
    if use_tessellation and not HAS_CPP:
        print("Warning: C++ tessellation module not available, falling back to histogram method.")
        use_tessellation = False
    
    # Adjust default resolution based on method
    if args.resolution == 256 and args.histogram:
        args.resolution = 1024  # Higher default for histogram
    
    # Load snapshot
    print(f"Loading snapshot from {args.snapshot}...")
    positions, particle_ids, header = load_snapshot(
        args.snapshot,
        snapshot_num=args.snapshot_num,
        particle_type=args.particle_type,
        read_ids=use_tessellation  # Only need IDs for tessellation
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
            n_threads=args.threads
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
