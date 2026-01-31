#!/usr/bin/env python3
"""
Build a panoramic time-series image from density projections.

This is Step 2 of the time-series visualization pipeline.

Usage:
    python build_time_series.py --projections projections.h5 --output time_series.png
"""

import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from time_series_config import TimeSeriesConfig, load_projections


def build_time_series_image(
    density_maps: dict,
    a_min: float,
    a_max: float,
    n_box_replications: int = 4,
    log_time: bool = True,
    use_physical_density: bool = False
) -> np.ndarray:
    """
    Assemble a panoramic time series image.

    Parameters
    ----------
    density_maps : dict
        Dictionary mapping scale factor to 2D density arrays (all same shape)
    a_min, a_max : float
        Scale factor range for the image
    n_box_replications : int
        How many times to tile the box in x (= time direction)
    log_time : bool
        If True, x maps logarithmically to scale factor
    use_physical_density : bool
        If True, scale density by a^-3 to show physical density evolution

    Returns
    -------
    output : np.ndarray
        Time-series image array
    """
    # Get image dimensions from first map
    a_values = np.array(sorted(density_maps.keys()))
    L_pix = density_maps[a_values[0]].shape[0]

    # Output image dimensions
    width = n_box_replications * L_pix
    height = L_pix
    output = np.zeros((height, width), dtype=np.float64)

    # Stack all density maps: shape (n_snapshots, L_pix, L_pix)
    density_stack = np.array([density_maps[a] for a in a_values])
    log_a_values = np.log(a_values)

    for x_out in range(width):
        # Map x to scale factor
        if log_time:
            a = a_min * (a_max / a_min) ** (x_out / (width - 1))
        else:
            a = a_min + (a_max - a_min) * (x_out / (width - 1))

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
        col_before = density_stack[idx_before, :, x_box]
        col_after = density_stack[idx_after, :, x_box]

        # Log-space interpolation
        eps = 1e-10
        log_col_before = np.log(col_before + eps)
        log_col_after = np.log(col_after + eps)
        log_col_interp = (1 - t) * log_col_before + t * log_col_after
        col_interp = np.exp(log_col_interp) - eps

        # Optional: convert to physical density
        if use_physical_density:
            col_interp = col_interp / (a ** 3)

        output[:, x_out] = col_interp

    return output


def apply_colormap(
    density: np.ndarray,
    log_scale: bool = True,
    percentile_clip: tuple = (0.5, 99.9)
) -> np.ndarray:
    """Apply log scaling and normalization to density field."""
    result = density.copy()

    if log_scale:
        # Handle zeros/negatives
        min_positive = result[result > 0].min() if np.any(result > 0) else 1e-10
        result = np.log10(np.maximum(result, min_positive))

    vmin = np.percentile(result, percentile_clip[0])
    vmax = np.percentile(result, percentile_clip[1])

    if vmax > vmin:
        normalized = (result - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(result)

    return np.clip(normalized, 0, 1)


def create_cosmic_blue_colormap():
    """Create a blue-to-white colormap for cosmic web visualizations."""
    # Dark blue to bright cyan/white
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

    # Convert hex to RGB
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


def render_time_series(
    config: TimeSeriesConfig,
    output_path: Path = None,
    show_axes: bool = True,
    figsize: tuple = None
):
    """
    Load projections and render the time-series image.

    Parameters
    ----------
    config : TimeSeriesConfig
        Pipeline configuration
    output_path : Path, optional
        Output file path (default: config.static_image_file)
    show_axes : bool
        Whether to show axis labels and colorbar
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    if output_path is None:
        output_path = config.static_image_file

    # Enable LaTeX-style rendering
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Load projections
    print(f"Loading projections from {config.projections_file}...")
    density_maps = load_projections(config.projections_file)

    if not density_maps:
        print("Error: No projections found")
        return

    a_values = sorted(density_maps.keys())
    print(f"Found {len(a_values)} snapshots: a = {a_values[0]:.4f} to {a_values[-1]:.4f}")

    # Build time-series image
    print("Building time-series image...")
    image = build_time_series_image(
        density_maps,
        a_min=config.a_min,
        a_max=config.a_max,
        n_box_replications=config.n_box_replications,
        log_time=config.log_time_mapping,
        use_physical_density=config.use_physical_density
    )

    # Apply colormap
    image_normalized = apply_colormap(
        image,
        log_scale=config.log_density_scale,
        percentile_clip=config.percentile_clip
    )

    # Determine figure size
    if figsize is None:
        aspect = image.shape[1] / image.shape[0]
        figsize = (min(24, 6 * aspect), 6)

    # Create figure
    if show_axes:
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
    else:
        fig = plt.figure(figsize=figsize, frameon=False, facecolor='black')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    # Choose colormap
    if config.colormap == 'cosmic_blue':
        cmap = create_cosmic_blue_colormap()
    else:
        cmap = plt.get_cmap(config.colormap)

    # Display
    im = ax.imshow(
        image_normalized,
        cmap=cmap,
        origin='lower',
        aspect='auto'
    )

    if show_axes:
        # Add scale factor labels on x-axis
        L_pix = density_maps[a_values[0]].shape[0]
        width = config.n_box_replications * L_pix

        # Generate tick positions for key scale factors
        key_a_values = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        key_a_values = [a for a in key_a_values if config.a_min <= a <= config.a_max]

        tick_positions = []
        tick_labels = []
        for a in key_a_values:
            if config.log_time_mapping:
                x = (width - 1) * np.log(a / config.a_min) / np.log(config.a_max / config.a_min)
            else:
                x = (width - 1) * (a - config.a_min) / (config.a_max - config.a_min)
            tick_positions.append(x)
            tick_labels.append(f'{a:.2g}')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='white')
        ax.set_xlabel(r'${\rm Scale\ factor}\ a$', fontsize=12, color='white')

        ax.set_yticks([])
        ax.set_ylabel(r'$y\ {\rm [comoving]}$', fontsize=12, color='white')

        # Colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(r'$\log_{10}(\rho / \bar{\rho})$', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        ax.set_title(r'${\rm Cosmic\ Web\ Evolution:}\ a=' + f'{config.a_min:.2g}' + r' \rightarrow a=' + f'{config.a_max:.0f}' + r'$',
                     fontsize=14, color='white', pad=10)

        # Style spines
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.tick_params(axis='x', colors='white')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()

    print(f"Saved time-series image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a time-series image from density projections"
    )
    parser.add_argument(
        "--projections", type=Path, required=True,
        help="HDF5 file containing density projections"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output image file (PNG)"
    )
    parser.add_argument(
        "--n-replications", type=int, default=4,
        help="Number of box replications (default: 4)"
    )
    parser.add_argument(
        "--a-min", type=float, default=0.02,
        help="Minimum scale factor (default: 0.02)"
    )
    parser.add_argument(
        "--a-max", type=float, default=100.0,
        help="Maximum scale factor (default: 100)"
    )
    parser.add_argument(
        "--linear-time", action="store_true",
        help="Use linear (not log) time mapping"
    )
    parser.add_argument(
        "--physical-density", action="store_true",
        help="Show physical (not comoving) density"
    )
    parser.add_argument(
        "--colormap", type=str, default="cosmic_blue",
        choices=["cosmic_blue", "Blues_r", "bone", "cividis", "viridis", "magma"],
        help="Colormap (default: cosmic_blue)"
    )
    parser.add_argument(
        "--no-axes", action="store_true",
        help="Remove axes and labels"
    )

    args = parser.parse_args()

    # Create config with projections file override
    config = TimeSeriesConfig(
        snapshot_dir=Path("."),  # Not used
        output_dir=args.output.parent,
        n_box_replications=args.n_replications,
        a_min=args.a_min,
        a_max=args.a_max,
        log_time_mapping=not args.linear_time,
        use_physical_density=args.physical_density,
        colormap=args.colormap,
        _projections_file_override=args.projections
    )

    render_time_series(
        config,
        output_path=args.output,
        show_axes=not args.no_axes
    )


if __name__ == "__main__":
    main()
