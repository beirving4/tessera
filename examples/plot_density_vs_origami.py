#!/usr/bin/env python3
"""
Create side-by-side comparison of density field and ORIGAMI morphology.

This script demonstrates that ORIGAMI correctly identifies cosmic web structures
by showing the density field alongside the morphological classification.

Left panel: Tessellation-based density slice (from pre-computed HDF5)
Right panel: ORIGAMI morphology (void, wall, filament, halo)

Usage:
    python plot_density_vs_origami.py /path/to/density_slice.hdf5 /path/to/snapshot.hdf5 [--output output.png]
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Try to import seaborn for mako colormap
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add tessera to path if needed
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


# ORIGAMI class definitions
ORIGAMI_CLASSES = {
    0: ('Void', '#1a1a2e'),      # Dark blue/black
    1: ('Wall', '#16697a'),       # Teal
    2: ('Filament', '#ffa62b'),   # Orange
    3: ('Halo', '#ff4757'),       # Red
}


def load_density_slice(slice_path: Path) -> tuple:
    """Load pre-computed density slice from HDF5 file."""
    print(f"Loading density slice: {slice_path}")

    with h5py.File(slice_path, 'r') as f:
        density_2d = f['density'][:]
        box_size = f['Header'].attrs['BoxSize']
        scale_factor = f['Header'].attrs['ScaleFactor']
        redshift = f['Header'].attrs['Redshift']
        extent = f['Grid'].attrs['extent']

        # Get slice parameters
        slice_center = f['SliceInfo'].attrs['center']
        slice_thickness = f['SliceInfo'].attrs['thickness']
        projection_axis = f['SliceInfo'].attrs['projection_axis']
        mean_density = f['SliceInfo'].attrs['mean_surface_density']

        # Check for reference density (z=0 normalization)
        if 'reference_mean_surface_density' in f['SliceInfo'].attrs:
            ref_mean_density = f['SliceInfo'].attrs['reference_mean_surface_density']
        else:
            ref_mean_density = mean_density

    print(f"  Box size: {box_size:.1f} Mpc/h")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Redshift: {redshift:.4f}")
    print(f"  Slice: center={slice_center:.1f}, thickness={slice_thickness:.1f}, axis={projection_axis}")
    print(f"  Grid resolution: {density_2d.shape}")

    return {
        'density_2d': density_2d,
        'box_size': box_size,
        'scale_factor': scale_factor,
        'redshift': redshift,
        'extent': extent,
        'slice_center': slice_center,
        'slice_thickness': slice_thickness,
        'projection_axis': projection_axis,
        'mean_density': mean_density,
        'ref_mean_density': ref_mean_density,
    }


def load_snapshot(snapshot_path: Path) -> tuple:
    """Load particle data from GADGET-4 HDF5 snapshot."""
    print(f"Loading snapshot: {snapshot_path}")

    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = f['Header'].attrs['BoxSize']
        scale_factor = f['Header'].attrs['Time']
        redshift = f['Header'].attrs['Redshift']

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    print(f"  Particles: {n_particles:,} ({grid_size}^3)")
    print(f"  Box size: {box_size:.1f} Mpc/h")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Redshift: {redshift:.4f}")

    return (
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(particle_ids, dtype=np.int64),
        float(box_size),
        grid_size,
        scale_factor,
        redshift,
    )


def compute_origami_morphology(
    positions: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
    grid_size: int,
    slice_center: float,
    slice_thickness: float,
    projection_axis: str = 'z',
    output_resolution: int = 256,
) -> tuple:
    """
    Compute ORIGAMI morphology and project onto 2D slice with density weighting.

    Runs ORIGAMI on the full box for correct classification, then projects
    only particles within the thin slice onto a 2D grid with density weighting.

    Returns:
        morphology_density: (N, N, 4) array of density per morphology class per pixel
        metadata: dict with slice info and global ORIGAMI fractions
    """
    print("\nRunning ORIGAMI classification on full box...")

    # Configure pipeline - need particle density for weighting
    config = ts.origami.PipelineConfig(grid_size, box_size)
    config.density_output_cells = output_resolution
    config.sample_density_at_particles = True  # Need particle density for weighting
    config.n_threads = 1
    config.pdf_n_bins = 50

    # Run pipeline on full box
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"  Global ORIGAMI fractions (full box):")
    print(f"    Void:     {result.f_void:.1%}")
    print(f"    Wall:     {result.f_wall:.1%}")
    print(f"    Filament: {result.f_filament:.1%}")
    print(f"    Halo:     {result.f_halo:.1%}")

    # Get morphology and particle density (in Lagrangian-sorted order)
    morphology = np.array(result.morphology)
    particle_density = np.array(result.particle_density)

    # Sort positions to Lagrangian order to match morphology output
    sorted_positions, _, _ = ts.origami.sort_particles_to_lagrangian(
        positions, particle_ids, grid_size, box_size
    )

    # Now compute 2D projection for the thin slice only
    print("\nProjecting ORIGAMI morphology onto 2D slice (density-weighted)...")

    # Map axis name to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]

    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2

    # Find particles in the slice
    coords_along_axis = sorted_positions[:, proj_axis]
    in_slice = (coords_along_axis >= slice_min) & (coords_along_axis < slice_max)
    n_in_slice = in_slice.sum()
    print(f"  Particles in slice: {n_in_slice:,} ({n_in_slice/len(coords_along_axis):.1%})")

    # Extract only particles in the thin slice
    slice_positions = sorted_positions[in_slice]
    slice_morphology = morphology[in_slice]
    slice_density = particle_density[in_slice]

    # Compute slice-specific ORIGAMI fractions
    slice_f_void = (slice_morphology == 0).sum() / n_in_slice
    slice_f_wall = (slice_morphology == 1).sum() / n_in_slice
    slice_f_filament = (slice_morphology == 2).sum() / n_in_slice
    slice_f_halo = (slice_morphology == 3).sum() / n_in_slice

    print(f"  Slice ORIGAMI fractions:")
    print(f"    Void:     {slice_f_void:.1%}")
    print(f"    Wall:     {slice_f_wall:.1%}")
    print(f"    Filament: {slice_f_filament:.1%}")
    print(f"    Halo:     {slice_f_halo:.1%}")

    # Determine which axes to use for x, y in the 2D projection
    if proj_axis == 0:  # x-axis projection -> y, z
        x_coords = slice_positions[:, 1]
        y_coords = slice_positions[:, 2]
    elif proj_axis == 1:  # y-axis projection -> x, z
        x_coords = slice_positions[:, 0]
        y_coords = slice_positions[:, 2]
    else:  # z-axis projection -> x, y
        x_coords = slice_positions[:, 0]
        y_coords = slice_positions[:, 1]

    # Create 2D grid for density-weighted morphology
    cell_size = box_size / output_resolution
    morphology_density = np.zeros((output_resolution, output_resolution, 4), dtype=np.float64)

    # Bin particles into grid cells
    x_idx = np.clip((x_coords / cell_size).astype(int), 0, output_resolution - 1)
    y_idx = np.clip((y_coords / cell_size).astype(int), 0, output_resolution - 1)

    # Accumulate density by morphology class
    for i in range(len(slice_positions)):
        xi, yi = x_idx[i], y_idx[i]
        morph_class = slice_morphology[i]
        morphology_density[xi, yi, morph_class] += slice_density[i]

    metadata = {
        'slice_center': slice_center,
        'slice_thickness': slice_thickness,
        'projection_axis': projection_axis,
        'n_particles_in_slice': n_in_slice,
        # Use slice fractions for the legend (what's actually shown)
        'f_void': slice_f_void,
        'f_wall': slice_f_wall,
        'f_filament': slice_f_filament,
        'f_halo': slice_f_halo,
        # Also store global fractions
        'f_void_global': result.f_void,
        'f_wall_global': result.f_wall,
        'f_filament_global': result.f_filament,
        'f_halo_global': result.f_halo,
    }

    return morphology_density, metadata


def create_morphology_rgb(morphology_density: np.ndarray) -> np.ndarray:
    """
    Create an RGB image from density-weighted morphology data.

    Each morphology class contributes its color weighted by its density.
    The brightness reflects the total density while color shows the
    morphology composition.

    Parameters
    ----------
    morphology_density : ndarray, shape (N, N, 4)
        Density contribution from each morphology class per pixel

    Returns
    -------
    rgb : ndarray, shape (N, N, 3)
        RGB image with values in [0, 1]
    """
    # Define RGB colors for each class (matching ORIGAMI_CLASSES)
    # Void: dark blue, Wall: teal, Filament: orange, Halo: red
    morph_colors = np.array([
        [0.10, 0.10, 0.18],   # Void - dark blue
        [0.09, 0.41, 0.48],   # Wall - teal
        [1.00, 0.65, 0.17],   # Filament - orange
        [1.00, 0.28, 0.34],   # Halo - red
    ])

    N = morphology_density.shape[0]
    rgb = np.zeros((N, N, 3), dtype=np.float64)

    # Total density per pixel
    total_density = morphology_density.sum(axis=2)

    # Normalize morphology density to get fractions
    mask = total_density > 0
    morph_fractions = np.zeros_like(morphology_density)
    morph_fractions[mask] = morphology_density[mask] / total_density[mask, np.newaxis]

    # Blend colors by morphology fraction
    for i in range(4):
        for c in range(3):
            rgb[:, :, c] += morph_fractions[:, :, i] * morph_colors[i, c]

    # Apply log-scale intensity based on total density
    # This makes the image look like the density slice
    log_density = np.zeros_like(total_density)
    log_density[mask] = np.log10(total_density[mask])

    # Normalize to [0, 1] range using percentiles
    valid_log = log_density[mask]
    if len(valid_log) > 0:
        vmin = np.percentile(valid_log, 1)
        vmax = np.percentile(valid_log, 99.5)
        intensity = np.clip((log_density - vmin) / (vmax - vmin), 0, 1)
    else:
        intensity = np.zeros_like(total_density)

    # Apply intensity to RGB (brightness modulation)
    # Use a gamma curve to enhance contrast
    gamma = 0.7
    intensity = np.power(intensity, gamma)

    # Modulate RGB by intensity
    for c in range(3):
        rgb[:, :, c] *= intensity

    # Ensure values are in [0, 1]
    rgb = np.clip(rgb, 0, 1)

    return rgb


def create_comparison_figure(
    density_2d: np.ndarray,
    morphology_density: np.ndarray,
    extent: tuple,
    scale_factor: float,
    redshift: float,
    mean_density: float,
    metadata: dict,
    output_path: Path,
    dpi: int = 200,
):
    """Create side-by-side density vs ORIGAMI morphology figure."""

    # Setup
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.subplots_adjust(wspace=0.05, left=0.06, right=0.94, bottom=0.08, top=0.92)

    # --- Left panel: Density ---
    ax_density = axes[0]

    # Get colormap
    if HAS_SEABORN:
        cmap_density = sns.color_palette("mako", as_cmap=True)
    else:
        cmap_density = plt.get_cmap('viridis')

    # Compute overdensity using mean density
    overdensity = density_2d / mean_density

    # Handle zeros for log scale
    overdensity_plot = np.maximum(overdensity, 1e-2)

    vmin = np.percentile(overdensity_plot[overdensity_plot > 0], 1)
    vmax = np.percentile(overdensity_plot, 99.5)

    im_density = ax_density.imshow(
        overdensity_plot.T,
        origin='lower',
        extent=extent,
        cmap=cmap_density,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect='equal',
    )

    ax_density.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)
    ax_density.set_ylabel(r'$y$ [Mpc/$h$]', fontsize=12)
    ax_density.set_title(r'Tessellation Density ($1+\delta$)', fontsize=14, fontweight='bold')

    # Colorbar for density
    divider = make_axes_locatable(ax_density)
    cax_density = divider.append_axes("right", size="4%", pad=0.08)
    cbar_density = fig.colorbar(im_density, cax=cax_density)
    cbar_density.set_label(r'$1 + \delta$', fontsize=11)

    # --- Right panel: ORIGAMI Morphology (density-weighted RGB) ---
    ax_morph = axes[1]

    # Create RGB image from density-weighted morphology
    morph_rgb = create_morphology_rgb(morphology_density)

    ax_morph.imshow(
        morph_rgb.transpose(1, 0, 2),  # Transpose x,y for imshow
        origin='lower',
        extent=extent,
        aspect='equal',
    )

    ax_morph.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)
    ax_morph.set_yticklabels([])
    ax_morph.tick_params(axis='y', length=0)
    ax_morph.set_title('ORIGAMI Morphology (Density-Weighted)', fontsize=14, fontweight='bold')

    # Legend for morphology classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ORIGAMI_CLASSES[i][1], edgecolor='white', linewidth=0.5,
              label=f"{ORIGAMI_CLASSES[i][0]} ({metadata[f'f_{ORIGAMI_CLASSES[i][0].lower()}']:.0%})")
        for i in range(4)
    ]
    ax_morph.legend(
        handles=legend_elements,
        loc='upper right',
        framealpha=0.9,
        fontsize=10,
    )

    # Overall title
    fig.suptitle(
        f'Cosmic Web Structure at $a = {scale_factor:.1f}$ ($z = {redshift:.2f}$)',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    # Save
    plt.savefig(output_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"\nSaved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create density vs ORIGAMI morphology comparison plot'
    )
    parser.add_argument('density_slice', type=str,
                        help='Path to pre-computed density slice HDF5 file')
    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 HDF5 snapshot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output DPI (default: 200)')

    args = parser.parse_args()

    # Load pre-computed density slice
    slice_path = Path(args.density_slice)
    if not slice_path.exists():
        raise FileNotFoundError(f"Density slice not found: {slice_path}")

    slice_data = load_density_slice(slice_path)

    # Load snapshot for ORIGAMI
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    positions, particle_ids, box_size, grid_size, scale_factor, redshift = load_snapshot(snapshot_path)

    # Use slice parameters from the density file
    slice_center = slice_data['slice_center']
    slice_thickness = slice_data['slice_thickness']
    projection_axis = slice_data['projection_axis']
    output_resolution = slice_data['density_2d'].shape[0]

    print(f"\nUsing slice parameters from density file:")
    print(f"  Center: {slice_center:.1f} Mpc/h")
    print(f"  Thickness: {slice_thickness:.1f} Mpc/h")
    print(f"  Projection axis: {projection_axis}")

    # Compute ORIGAMI morphology projection (density-weighted)
    morphology_density, metadata = compute_origami_morphology(
        positions, particle_ids, box_size, grid_size,
        slice_center, slice_thickness,
        projection_axis=projection_axis,
        output_resolution=output_resolution,
    )

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = snapshot_path.parent / f"density_vs_origami_{snapshot_path.stem}.png"

    # Create figure
    create_comparison_figure(
        density_2d=slice_data['density_2d'],
        morphology_density=morphology_density,
        extent=slice_data['extent'],
        scale_factor=slice_data['scale_factor'],
        redshift=slice_data['redshift'],
        mean_density=slice_data['mean_density'],
        metadata=metadata,
        output_path=output_path,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
