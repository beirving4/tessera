#!/usr/bin/env python3
"""
Example: Halo-Centric Density Field Visualization

This script demonstrates how to:
1. Read a GADGET-4 snapshot and compute 3D density via phase-space tessellation
2. Load the associated FOF/Subfind halo catalog
3. Extract a 3D density sub-cube centered on a selected halo
4. Save the extracted density field to HDF5
5. Create tri-panel visualization (XY, XZ, YZ slices through halo center)

Usage:
  python halo_density.py snapshot_034.hdf5 -o halo_density.h5 --plot halo.png
  python halo_density.py snapshot_034.hdf5 -o halo_density.h5 --halo-index 0  # Most massive
  python halo_density.py snapshot_034.hdf5 -o halo_density.h5 --halo-index 5  # 6th most massive

Reference: gotetra (https://github.com/phil-mansfield/gotetra)
"""

import contextlib
import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Import shared utilities
from tessera.utils import (
    load_snapshot,
    find_halo_catalog,
    load_halo_catalog,
    infer_snapshot_number,
    sort_positions_lagrangian,
    infer_grid_size,
    compute_mean_density,
    check_tessera_available,
)

# Import tessera
try:
    import tessera as ts
except ImportError:
    import _tessera as ts


def compute_3d_density(positions, particle_ids, box_size, grid_resolution=256,
                       n_samples=100, mass_per_particle=1.0, n_threads=0,
                       subbox_origin=None, subbox_width=None):
    """
    Compute 3D density field using phase-space tessellation.

    Parameters
    ----------
    positions : ndarray
        Particle positions
    particle_ids : ndarray
        Particle IDs for Lagrangian ordering
    box_size : float
        Simulation box size
    grid_resolution : int
        Number of cells per dimension
    n_samples : int
        Monte Carlo samples per tetrahedron
    mass_per_particle : float
        Mass per particle
    n_threads : int
        Number of threads (0=auto)
    subbox_origin : tuple, optional
        If provided, render only a sub-region starting at (x, y, z)
    subbox_width : float or tuple, optional
        Width of sub-region (single value for cube, or tuple for different dimensions)

    Returns
    -------
    density_3d : ndarray, shape (grid_resolution, grid_resolution, grid_resolution)
        3D density field
    cell_width : float
        Width of each cell
    n_tetrahedra : int
        Number of tetrahedra processed
    """
    if not check_tessera_available():
        raise ImportError("C++ tessellation module not available.")

    # Infer Lagrangian grid size
    grid_size = infer_grid_size(len(positions))

    # Sort particles by Lagrangian ID
    print("Sorting particles by Lagrangian ID...")
    sorted_positions = sort_positions_lagrangian(positions, particle_ids, grid_size)

    # Create config
    config = ts.density.TetraDensityConfig()
    config.box_size = box_size
    config.lagrangian_grid_size = grid_size
    config.output_cells = grid_resolution
    config.n_samples = n_samples
    config.particle_mass = mass_per_particle
    config.n_threads = n_threads

    # Configure sub-box if requested
    if subbox_origin is not None and subbox_width is not None:
        config.subbox_enabled = True
        config.subbox_origin = tuple(subbox_origin)
        if isinstance(subbox_width, (int, float)):
            config.subbox_width = (subbox_width, subbox_width, subbox_width)
        else:
            config.subbox_width = tuple(subbox_width)
        render_width = max(config.subbox_width)
        print(f"Computing sub-box density field ({grid_resolution}^3 for {render_width:.2f} Mpc/h region)...")
    else:
        render_width = box_size
        print(f"Computing full-box density field ({grid_resolution}^3)...")

    # Compute 3D density
    result = ts.density.compute_tetra_density_3d(sorted_positions, config)
    # C++ uses idx = x + y*N + z*N*N (x varies fastest)
    # With NumPy C-order reshape, this gives density[z, y, x]
    density_3d = np.array(result.density).reshape(
        (grid_resolution, grid_resolution, grid_resolution)
    )

    print(f"  Tetrahedra processed: {result.n_tetrahedra:,}")

    cell_width = render_width / grid_resolution
    return density_3d, cell_width, result.n_tetrahedra


def save_halo_density_hdf5(filename, sub_density, full_density_3d, header,
                           halo_info, extraction_info, cell_width):
    """Save the extracted halo density field to HDF5."""
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('halo_density', data=sub_density, compression='gzip')
        dset.attrs['units'] = 'mass / volume (code units)'
        dset.attrs['cell_width'] = cell_width

        if full_density_3d is not None:
            f.create_dataset('full_density', data=full_density_3d, compression='gzip')

        hdr = f.create_group('Header')
        hdr.attrs['BoxSize'] = header.box_size
        hdr.attrs['Redshift'] = header.redshift
        hdr.attrs['Time'] = header.time
        hdr.attrs['NumPartTotal'] = list(header.num_part_total)
        hdr.attrs['MassTable'] = list(header.mass_table)

        halo = f.create_group('Halo')
        for key, val in halo_info.items():
            if isinstance(val, np.ndarray):
                halo.create_dataset(key, data=val)
            else:
                halo.attrs[key] = val

        ext = f.create_group('Extraction')
        for key, val in extraction_info.items():
            ext.attrs[key] = val

    print(f"Saved halo density field to {filename}")


def plot_halo_density(sub_density, halo_info, extraction_info, header,
                      output_file=None, cmap='mako', log_scale=True,
                      vmin=None, vmax=None, overdensity=True):
    """Create tri-panel visualization of density slices through halo center."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle
        with contextlib.suppress(ImportError):
            import seaborn as sns
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    center_idx = sub_density.shape[0] // 2

    # XY slice (z=0), XZ slice (y=0), YZ slice (x=0)
    slices = {
        'XY': sub_density[center_idx, :, :],
        'XZ': sub_density[:, center_idx, :],
        'YZ': sub_density[:, :, center_idx],
    }

    mean_density = extraction_info.get('mean_3d_density', 1.0)
    r200c = halo_info.get('r_crit200', 0)
    extraction_radius = extraction_info['extraction_radius']
    extent = (-extraction_radius, extraction_radius, -extraction_radius, extraction_radius)

    axis_labels = {
        'XY': ('X', 'Y'),
        'XZ': ('X', 'Z'),
        'YZ': ('Y', 'Z'),
    }

    for ax, (plane_name, slice_data) in zip(axes, slices.items()):
        plot_data = slice_data.copy()

        if overdensity:
            plot_data = plot_data / mean_density
            scale_factor = header.time
            label = rf'$1 + \delta(a={scale_factor:.2f})$'
        else:
            label = r'$\rho$ (code units)'

        if log_scale:
            plot_data = np.maximum(plot_data, plot_data[plot_data > 0].min() * 0.1)
            vmin_val = vmin if vmin is not None else np.percentile(plot_data[plot_data > 0], 1)
            vmax_val = vmax if vmax is not None else np.percentile(plot_data, 99.9)
            norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
        else:
            vmin_val = vmin if vmin is not None else np.percentile(plot_data, 1)
            vmax_val = vmax if vmax is not None else np.percentile(plot_data, 99)
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

        ax.plot(0, 0, 'w+', markersize=15, markeredgewidth=2)

        if r200c > 0:
            circle = Circle((0, 0), r200c, fill=False, color='white',
                           linestyle='--', linewidth=1.5, label=f'R200c = {r200c:.2f}')
            ax.add_patch(circle)

        ax.set_xlabel(f'{axis_labels[plane_name][0]} (code units)', fontsize=10)
        ax.set_ylabel(f'{axis_labels[plane_name][1]} (code units)', fontsize=10)
        ax.set_title(f'{plane_name} Plane', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=9)

    halo_idx = halo_info.get('index', '?')
    halo_mass = halo_info.get('mass', 0)
    mass_str = f"{halo_mass:.2e}" if halo_mass > 0 else "N/A"

    fig.suptitle(
        f"Halo {halo_idx} Density Field | z = {header.redshift:.3f} | "
        f"M = {mass_str} (10^10 M_sun/h)",
        fontsize=14, y=1.02
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create halo-centric density field visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize most massive halo
  python halo_density.py snapshot_034.hdf5 -o halo.h5 --plot halo.png

  # Specify halo by index (sorted by mass)
  python halo_density.py snapshot_034.hdf5 -o halo.h5 --halo-index 5

  # Custom extraction radius (multiple of R200c)
  python halo_density.py snapshot_034.hdf5 -o halo.h5 --radius-factor 5

  # Use specific halo catalog
  python halo_density.py snapshot_034.hdf5 -o halo.h5 --catalog fof_subhalo_tab_034.hdf5
        """
    )

    parser.add_argument('snapshot', type=str, help='Path to snapshot file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 filename for extracted density')
    parser.add_argument('--catalog', type=str, default=None,
                        help='Path to halo catalog (auto-detected if not specified)')
    parser.add_argument('--halo-index', type=int, default=0,
                        help='Halo index (0=most massive, 1=2nd most massive, etc.)')
    parser.add_argument('--radius-factor', type=float, default=3.0,
                        help='Extraction radius as multiple of R200 (default: 3.0)')
    parser.add_argument('--min-radius', type=float, default=5.0,
                        help='Minimum extraction radius in code units (default: 5.0)')
    parser.add_argument('--fixed-radius', type=float, default=None,
                        help='Fixed extraction radius in cMpc/h (overrides radius-factor)')
    parser.add_argument('--use-r-mean', action='store_true',
                        help='Use R_Mean200 instead of R_Crit200 (sometimes better units)')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Grid resolution for density field (default: 128)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (default: 100)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output image filename (optional)')
    parser.add_argument('--cmap', type=str, default='magma',
                        help='Colormap for visualization (default: magma)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    parser.add_argument('--save-full', action='store_true',
                        help='Also save full 3D density field (can be large)')

    args = parser.parse_args()

    if not check_tessera_available():
        print("Error: C++ module not available. Cannot compute tessellation.")
        sys.exit(1)

    # Find halo catalog
    catalog_path = args.catalog
    if catalog_path is None:
        catalog_path = find_halo_catalog(args.snapshot)
        if catalog_path is None:
            print("Error: Could not find halo catalog. Specify with --catalog.")
            sys.exit(1)
        print(f"Found halo catalog: {catalog_path}")
    else:
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            print(f"Error: Halo catalog not found: {catalog_path}")
            sys.exit(1)

    # Load halo catalog
    print(f"\nLoading halo catalog...")
    catalog = load_halo_catalog(catalog_path)
    n_groups = catalog.num_groups()
    n_subhalos = catalog.num_subhalos()
    print(f"  Groups: {n_groups:,}")
    print(f"  Subhalos: {n_subhalos:,}")
    print(f"  Redshift: {catalog.header.redshift:.4f}")

    if n_groups == 0:
        print("Error: No halos found in catalog.")
        sys.exit(1)

    # Sort groups by mass to find most massive
    groups = catalog.groups
    group_masses = np.array([g.mass for g in groups])
    sorted_indices = np.argsort(group_masses)[::-1]

    if args.halo_index >= n_groups:
        print(f"Error: Halo index {args.halo_index} out of range (0-{n_groups-1})")
        sys.exit(1)

    # Select halo
    selected_idx = sorted_indices[args.halo_index]
    halo = groups[selected_idx]

    # Check if R_Crit200 needs unit conversion
    r_crit200 = halo.r_crit200
    r_mean200 = halo.r_mean200

    if r_crit200 > 0 and r_mean200 > 0:
        if r_crit200 < 0.1 and r_mean200 > 0.5:
            r_crit200_physical = r_crit200 * catalog.header.box_size
            print(f"Note: R_Crit200 appears to be in box fraction, converting to physical units")
        else:
            r_crit200_physical = r_crit200
    elif r_crit200 > 0:
        if r_crit200 < 0.1:
            r_crit200_physical = r_crit200 * catalog.header.box_size
        else:
            r_crit200_physical = r_crit200
    else:
        r_crit200_physical = 0

    print(f"\nSelected halo (rank {args.halo_index} by mass):")
    print(f"  Catalog index: {selected_idx}")
    print(f"  Position: ({halo.pos[0]:.2f}, {halo.pos[1]:.2f}, {halo.pos[2]:.2f})")
    print(f"  Mass: {halo.mass:.4e} (10^10 M_sun/h)")
    print(f"  M200c: {halo.m_crit200:.4e} (10^10 M_sun/h)")
    print(f"  R200c (raw): {halo.r_crit200:.6f}")
    print(f"  R200c (physical): {r_crit200_physical:.4f} Mpc/h")
    print(f"  R200m: {r_mean200:.4f} Mpc/h")
    print(f"  N_particles: {halo.len:,}")

    # Load snapshot
    print(f"\nLoading snapshot from {args.snapshot}...")
    positions, particle_ids, header = load_snapshot(
        args.snapshot,
        particle_type=args.particle_type,
        read_ids=True
    )

    box_size = header.box_size
    print(f"  Box size: {box_size}")
    print(f"  Redshift: {header.redshift:.4f}")
    print(f"  Loaded {len(positions):,} particles")

    mass_per_particle = header.mass_table[args.particle_type]
    if mass_per_particle == 0:
        mass_per_particle = 1.0

    # Determine extraction radius
    if args.use_r_mean:
        r200 = r_mean200
        r200_label = "R200m"
    else:
        r200 = r_crit200_physical
        r200_label = "R200c"

    if args.fixed_radius is not None:
        extraction_radius = args.fixed_radius
        print(f"\nExtraction region (fixed):")
        print(f"  {r200_label}: {r200:.4f} Mpc/h")
        print(f"  Extraction radius: {extraction_radius:.4f} cMpc/h (fixed)")
    else:
        extraction_radius = max(r200 * args.radius_factor, args.min_radius)
        print(f"\nExtraction region:")
        print(f"  {r200_label}: {r200:.4f} Mpc/h")
        print(f"  Extraction radius: {extraction_radius:.4f} ({args.radius_factor}x {r200_label})")

    # Compute sub-box origin
    halo_pos = np.array(halo.pos)
    subbox_width = 2 * extraction_radius
    subbox_origin = halo_pos - extraction_radius

    # Handle periodic wrapping
    for i in range(3):
        if subbox_origin[i] < 0:
            subbox_origin[i] += box_size
        elif subbox_origin[i] >= box_size:
            subbox_origin[i] -= box_size

    # Compute density field
    print(f"\nComputing sub-box density field...")
    print(f"  Sub-box origin: ({subbox_origin[0]:.2f}, {subbox_origin[1]:.2f}, {subbox_origin[2]:.2f})")
    print(f"  Sub-box width: {subbox_width:.2f} Mpc/h")
    print(f"  Grid resolution: {args.resolution}^3")
    print(f"  Cell width: {subbox_width / args.resolution:.4f} Mpc/h")

    sub_density, cell_width, n_tetra = compute_3d_density(
        positions, particle_ids, box_size,
        grid_resolution=args.resolution,
        n_samples=args.samples,
        mass_per_particle=mass_per_particle,
        n_threads=args.threads,
        subbox_origin=subbox_origin,
        subbox_width=subbox_width
    )

    print(f"\nDensity field statistics:")
    print(f"  Shape: {sub_density.shape}")
    print(f"  Min: {sub_density.min():.6e}")
    print(f"  Max: {sub_density.max():.6e}")
    print(f"  Mean: {sub_density.mean():.6e}")

    # Prepare info dicts
    halo_info = {
        'index': int(selected_idx),
        'rank': args.halo_index,
        'pos': halo_pos,
        'mass': halo.mass,
        'm_crit200': halo.m_crit200,
        'r_crit200': r_crit200_physical,
        'r_crit200_raw': halo.r_crit200,
        'r_mean200': r_mean200,
        'n_particles': halo.len,
    }

    mean_3d_density = compute_mean_density(len(positions), mass_per_particle, box_size)

    extraction_info = {
        'extraction_radius': extraction_radius,
        'radius_factor': args.radius_factor,
        'cell_width': cell_width,
        'grid_resolution': args.resolution,
        'n_samples': args.samples,
        'mean_3d_density': mean_3d_density,
        'subbox_origin': tuple(subbox_origin),
        'subbox_width': subbox_width,
        'n_tetrahedra': n_tetra,
    }

    # Save to HDF5
    save_halo_density_hdf5(
        args.output, sub_density, None, header,
        halo_info, extraction_info, cell_width
    )

    # Optional visualization
    if args.plot:
        print(f"\nGenerating visualization...")
        plot_halo_density(
            sub_density, halo_info, extraction_info, header,
            output_file=args.plot,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            overdensity=True
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
