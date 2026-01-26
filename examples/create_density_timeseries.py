#!/usr/bin/env python3
"""
Create time series HDF5 files and visualizations from density slice maps.

This script:
1. Reads individual density slice HDF5 files (one per snapshot)
2. Combines them into a single time series HDF5 file
3. Creates a Diemer-style panoramic time evolution visualization

Usage:
    python create_density_timeseries.py /path/to/density_maps --output timeseries.h5

The input directory should contain files named: primary_density_slice_XXX.hdf5
where XXX is the snapshot number (000, 001, ..., 075).
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_density_slice(filepath: Path) -> dict:
    """Load a single density slice file."""
    with h5py.File(filepath, 'r') as f:
        data = {
            'density': f['density'][:],
            'box_size': f['Header'].attrs['BoxSize'],
            'redshift': f['Header'].attrs['Redshift'],
            'scale_factor': f['Header'].attrs['ScaleFactor'],
            'mass_per_particle': f['Header'].attrs['MassPerParticle'],
            'n_particles': f['Header'].attrs['NumParticlesTotal'],
            'grid_resolution': f['SliceInfo'].attrs['grid_resolution'],
            'mean_surface_density': f['SliceInfo'].attrs['mean_surface_density'],
            'mean_3d_density': f['SliceInfo'].attrs['mean_3d_density'],
            'slice_center': f['SliceInfo'].attrs['center'],
            'slice_thickness': f['SliceInfo'].attrs['thickness'],
            'projection_axis': f['SliceInfo'].attrs['projection_axis'],
            'method': f['SliceInfo'].attrs['method'],
        }
        if 'n_samples' in f['SliceInfo'].attrs:
            data['n_samples'] = f['SliceInfo'].attrs['n_samples']
        if 'n_tetrahedra' in f['SliceInfo'].attrs:
            data['n_tetrahedra'] = f['SliceInfo'].attrs['n_tetrahedra']
    return data


def create_timeseries_hdf5(
    input_dir: Path,
    output_file: Path,
    file_pattern: str = "primary_density_slice_{:03d}.hdf5",
    n_snapshots: int = 76,
) -> dict:
    """
    Combine individual density slice files into a time series HDF5.

    Returns metadata dict for use in plotting.
    """
    print(f"Creating time series from {input_dir}")
    print(f"Output: {output_file}")

    # Load first file to get dimensions
    first_file = input_dir / file_pattern.format(0)
    if not first_file.exists():
        raise FileNotFoundError(f"First file not found: {first_file}")

    first_data = load_density_slice(first_file)
    resolution = first_data['grid_resolution']
    box_size = first_data['box_size']

    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Box size: {box_size:.2f}")
    print(f"  Snapshots: {n_snapshots}")

    # Collect metadata
    scale_factors = []
    redshifts = []
    mean_densities = []

    with h5py.File(output_file, 'w') as f:
        # Create datasets with resizable first dimension
        density_dset = f.create_dataset(
            'density',
            shape=(n_snapshots, resolution, resolution),
            maxshape=(None, resolution, resolution),
            dtype=np.float64,
            compression='gzip',
            compression_opts=4
        )

        overdensity_dset = f.create_dataset(
            'overdensity',
            shape=(n_snapshots, resolution, resolution),
            maxshape=(None, resolution, resolution),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        # First pass: find which files exist (including ICs)
        existing_files = []

        # Check for ICs file first (a=0.01)
        ics_file = input_dir / "primary_density_slice_ics.hdf5"
        if ics_file.exists():
            existing_files.append((-1, ics_file))  # Use -1 as index for ICs
            print(f"  Found ICs file: {ics_file.name}")
        else:
            print(f"  WARNING: Missing ICs file {ics_file.name}")

        # Then check numbered snapshots
        for i in range(n_snapshots):
            filepath = input_dir / file_pattern.format(i)
            if filepath.exists():
                existing_files.append((i, filepath))
            else:
                print(f"  WARNING: Missing file {filepath.name}")

        n_existing = len(existing_files)
        print(f"  Found {n_existing}/{n_snapshots} files")

        # Resize datasets if needed
        if n_existing < n_snapshots:
            density_dset.resize((n_existing, resolution, resolution))
            overdensity_dset.resize((n_existing, resolution, resolution))

        # Load each existing snapshot
        for idx, (snap_idx, filepath) in enumerate(existing_files):
            data = load_density_slice(filepath)
            density_dset[idx] = data['density']

            # Compute overdensity (1 + delta)
            mean_density = data['mean_surface_density']
            if mean_density > 0:
                overdensity_dset[idx] = (data['density'] / mean_density).astype(np.float32)
            else:
                overdensity_dset[idx] = np.ones_like(data['density'], dtype=np.float32)

            scale_factors.append(data['scale_factor'])
            redshifts.append(data['redshift'])
            mean_densities.append(mean_density)

            if (idx + 1) % 10 == 0 or idx == n_existing - 1:
                print(f"  Loaded {idx+1}/{n_existing} (a={data['scale_factor']:.4f}, z={data['redshift']:.2f})")

        # Store metadata
        f.create_dataset('scale_factor', data=np.array(scale_factors))
        f.create_dataset('redshift', data=np.array(redshifts))
        f.create_dataset('mean_surface_density', data=np.array(mean_densities))

        # Store header info from first file
        hdr = f.create_group('Header')
        hdr.attrs['box_size'] = box_size
        hdr.attrs['grid_resolution'] = resolution
        hdr.attrs['n_snapshots'] = n_existing  # Actual number loaded
        hdr.attrs['n_snapshots_expected'] = n_snapshots
        hdr.attrs['n_particles'] = first_data['n_particles']
        hdr.attrs['mass_per_particle'] = first_data['mass_per_particle']
        hdr.attrs['slice_center'] = first_data['slice_center']
        hdr.attrs['slice_thickness'] = first_data['slice_thickness']
        hdr.attrs['projection_axis'] = first_data['projection_axis']
        hdr.attrs['method'] = first_data['method']
        if 'n_samples' in first_data:
            hdr.attrs['n_samples'] = first_data['n_samples']

    print(f"  Saved time series to {output_file}")

    return {
        'scale_factors': np.array(scale_factors),
        'redshifts': np.array(redshifts),
        'box_size': box_size,
        'resolution': resolution,
        'n_snapshots': n_snapshots,
    }


def create_cosmic_blue_colormap():
    """Create a colormap similar to Diemer's cosmic web visualizations."""
    colors = [
        (0.0, '#000510'),   # Nearly black
        (0.2, '#001a33'),   # Dark blue
        (0.4, '#003366'),   # Medium blue
        (0.6, '#0066aa'),   # Blue
        (0.8, '#00aacc'),   # Cyan
        (1.0, '#ffffff'),   # White for brightest
    ]

    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]

    rgb_colors = []
    for hex_color in hex_colors:
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)

    cmap = LinearSegmentedColormap.from_list(
        'cosmic_blue',
        list(zip(positions, rgb_colors))
    )
    return cmap


def build_panoramic_image(
    overdensity_stack: np.ndarray,
    scale_factors: np.ndarray,
    a_min: float = 0.02,
    a_max: float = 100.0,
    n_box_replications: int = 4,
) -> np.ndarray:
    """
    Build a Diemer-style panoramic time-series image.

    Parameters
    ----------
    overdensity_stack : np.ndarray
        Shape (n_snapshots, L_pix, L_pix) - overdensity maps for each snapshot
    scale_factors : np.ndarray
        Scale factors for each snapshot
    a_min, a_max : float
        Scale factor range for the image
    n_box_replications : int
        How many times to tile the box in x (time) direction

    Returns
    -------
    output : np.ndarray
        Panoramic time-series image
    """
    L_pix = overdensity_stack.shape[1]
    width = n_box_replications * L_pix
    height = L_pix
    output = np.zeros((height, width), dtype=np.float64)

    # Sort by scale factor
    sort_idx = np.argsort(scale_factors)
    a_values = scale_factors[sort_idx]
    density_sorted = overdensity_stack[sort_idx]
    log_a_values = np.log(a_values)

    for x_out in range(width):
        # Map x to scale factor (logarithmic)
        a = a_min * (a_max / a_min) ** (x_out / (width - 1))

        # Position within periodic box
        x_box = x_out % L_pix

        # Interpolate between snapshots (in log-space)
        log_a = np.log(a)

        # Find bracketing snapshots
        idx_after = np.searchsorted(log_a_values, log_a)
        idx_after = np.clip(idx_after, 1, len(a_values) - 1)
        idx_before = idx_after - 1

        # Interpolation weight
        denom = log_a_values[idx_after] - log_a_values[idx_before]
        if abs(denom) < 1e-10:
            t = 0.0
        else:
            t = (log_a - log_a_values[idx_before]) / denom
        t = np.clip(t, 0.0, 1.0)

        # Interpolate the column (in log-density space for smoother results)
        col_before = density_sorted[idx_before, :, x_box]
        col_after = density_sorted[idx_after, :, x_box]

        # Log-space interpolation
        eps = 1e-10
        log_col_before = np.log(col_before + eps)
        log_col_after = np.log(col_after + eps)
        log_col_interp = (1 - t) * log_col_before + t * log_col_after
        col_interp = np.exp(log_col_interp) - eps

        output[:, x_out] = col_interp

    return output


def apply_colormap_normalization(
    density: np.ndarray,
    log_scale: bool = True,
    percentile_clip: tuple = (0.5, 99.9)
) -> np.ndarray:
    """Apply log scaling and normalization to density field."""
    result = density.copy()

    if log_scale:
        min_positive = result[result > 0].min() if np.any(result > 0) else 1e-10
        result = np.log10(np.maximum(result, min_positive))

    vmin = np.percentile(result, percentile_clip[0])
    vmax = np.percentile(result, percentile_clip[1])

    if vmax > vmin:
        normalized = (result - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(result)

    return np.clip(normalized, 0, 1)


def plot_time_evolution(
    timeseries_file: Path,
    output_file: Path,
    a_min: float = 0.02,
    a_max: float = 100.0,
    n_box_replications: int = 4,
    cmap: str = 'cosmic_blue',
    figsize: tuple = None,
):
    """
    Create a Diemer-style panoramic plot showing continuous time evolution.
    """
    print(f"Creating panoramic time evolution plot: {output_file}")

    # Enable LaTeX-style rendering
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    with h5py.File(timeseries_file, 'r') as f:
        overdensity = f['overdensity'][:]
        scale_factors = f['scale_factor'][:]
        box_size = f['Header'].attrs['box_size']

    n_actual = len(scale_factors)
    print(f"  Found {n_actual} snapshots in file")
    print(f"  Scale factor range: {scale_factors.min():.4f} to {scale_factors.max():.4f}")

    # Build panoramic image
    print(f"  Building panoramic image ({n_box_replications} box replications)...")
    image = build_panoramic_image(
        overdensity,
        scale_factors,
        a_min=a_min,
        a_max=a_max,
        n_box_replications=n_box_replications,
    )

    # Apply colormap normalization
    image_normalized = apply_colormap_normalization(image, log_scale=True)

    # Determine figure size
    if figsize is None:
        aspect = image.shape[1] / image.shape[0]
        figsize = (min(24, 6 * aspect), 6)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')

    # Choose colormap
    if cmap == 'cosmic_blue':
        colormap = create_cosmic_blue_colormap()
    else:
        colormap = plt.get_cmap(cmap)

    # Display image
    im = ax.imshow(
        image_normalized,
        cmap=colormap,
        origin='lower',
        aspect='auto'
    )

    # Add scale factor labels on x-axis
    L_pix = overdensity.shape[1]
    width = n_box_replications * L_pix

    key_a_values = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    key_a_values = [a for a in key_a_values if a_min <= a <= a_max]

    tick_positions = []
    tick_labels = []
    for a in key_a_values:
        x = (width - 1) * np.log(a / a_min) / np.log(a_max / a_min)
        tick_positions.append(x)
        tick_labels.append(f'{a:.2g}')

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, color='white')
    ax.set_xlabel(r'${\rm Scale\ factor}\ a$', fontsize=12, color='white')

    ax.set_yticks([])
    ax.set_ylabel(r'$y\ {\rm [comoving]}$', fontsize=12, color='white')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'$\log_{10}(\rho / \bar{\rho})$', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Title
    sim_name = timeseries_file.stem.replace('_timeseries', '').replace('Uniform_', '').replace('_primary', '')
    ax.set_title(
        r'${\rm Cosmic\ Web\ Evolution:}\ a=' + f'{a_min:.2g}' + r' \rightarrow a=' + f'{a_max:.0f}' + r'$' +
        f'\n{sim_name}',
        fontsize=14, color='white', pad=10
    )

    # Style spines
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='x', colors='white')

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()

    print(f"  Saved panoramic plot to {output_file}")


def plot_single_snapshot_comparison(
    timeseries_files: list[Path],
    output_file: Path,
    snapshot_idx: int = -1,  # -1 for final snapshot
    cmap: str = 'magma',
    vmin: float = 0.01,
    vmax: float = 100,
):
    """
    Create a comparison plot of the same snapshot across different simulations.
    """
    print(f"Creating comparison plot: {output_file}")

    n_sims = len(timeseries_files)
    fig, axes = plt.subplots(1, n_sims, figsize=(5*n_sims, 5))
    if n_sims == 1:
        axes = [axes]

    for i, ts_file in enumerate(timeseries_files):
        ax = axes[i]

        with h5py.File(ts_file, 'r') as f:
            overdensity = f['overdensity'][snapshot_idx]
            scale_factor = f['scale_factor'][snapshot_idx]
            redshift = f['redshift'][snapshot_idx]
            box_size = f['Header'].attrs['box_size']

        im = ax.imshow(
            overdensity.T,
            extent=(0, box_size, 0, box_size),
            origin='lower',
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation='nearest'
        )

        sim_name = ts_file.stem.replace('_timeseries', '').replace('Uniform_', '').replace('_primary', '')
        ax.set_title(f'{sim_name}\na={scale_factor:.3f}, z={redshift:.2f}', fontsize=11)
        ax.set_xlabel('x [Mpc/h]', fontsize=10)
        if i == 0:
            ax.set_ylabel('y [Mpc/h]', fontsize=10)
        ax.tick_params(labelsize=8)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r'$1 + \delta$', fontsize=12)

    plt.suptitle('Final Snapshot Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved comparison to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create time series HDF5 and Diemer-style visualization from density slices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('input_dir', type=str,
                        help='Directory containing density slice HDF5 files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HDF5 filename (default: <input_dir>_timeseries.h5)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output plot filename (default: <output>.png)')
    parser.add_argument('--n-snapshots', type=int, default=76,
                        help='Number of snapshots (default: 76)')
    parser.add_argument('--a-min', type=float, default=0.02,
                        help='Minimum scale factor for visualization (default: 0.02)')
    parser.add_argument('--a-max', type=float, default=100.0,
                        help='Maximum scale factor for visualization (default: 100)')
    parser.add_argument('--n-replications', type=int, default=4,
                        help='Number of box replications in time direction (default: 4)')
    parser.add_argument('--cmap', type=str, default='cosmic_blue',
                        help='Colormap (default: cosmic_blue)')
    parser.add_argument('--skip-hdf5', action='store_true',
                        help='Skip HDF5 creation, only make plot from existing file')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create time_series subdirectory in input folder
    output_dir = input_dir / "time_series"
    output_dir.mkdir(exist_ok=True)

    # Determine output filenames
    if args.output:
        output_file = Path(args.output)
    else:
        # Use parent directory name for output
        sim_name = input_dir.parent.name
        output_file = output_dir / f"{sim_name}_timeseries.h5"

    plot_file = Path(args.plot) if args.plot else output_file.with_suffix('.png')

    # Create time series HDF5
    if not args.skip_hdf5:
        metadata = create_timeseries_hdf5(
            input_dir=input_dir,
            output_file=output_file,
            n_snapshots=args.n_snapshots,
        )

    # Create panoramic visualization
    if output_file.exists():
        plot_time_evolution(
            timeseries_file=output_file,
            output_file=plot_file,
            a_min=args.a_min,
            a_max=args.a_max,
            n_box_replications=args.n_replications,
            cmap=args.cmap,
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
