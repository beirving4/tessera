#!/usr/bin/env python3
"""
Create animations from density timeseries HDF5 files.

This script reads the timeseries HDF5 files created by create_density_timeseries.py
and generates MP4 animations showing the evolution of cosmic structure.

Usage:
    python animate_density_timeseries.py /path/to/density_maps

The script expects a time_series subdirectory containing *_timeseries.h5 files.
Output animations are saved to an animation subdirectory.
"""

import numpy as np
import h5py
import argparse
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def check_ffmpeg():
    """Check if ffmpeg is available for animation encoding."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg for animation creation.\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  conda: conda install ffmpeg"
        )


def create_cosmic_blue_colormap():
    """Create a blue-to-white colormap for cosmic web visualizations."""
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


def cosmic_time_gyr(a: float, H0: float = 70.0, Om: float = 0.3, OL: float = 0.7) -> float:
    """
    Compute cosmic time in Gyr for a given scale factor.
    Uses flat LCDM cosmology with default Planck-like parameters.
    """
    try:
        from scipy import integrate
    except ImportError:
        return np.nan

    # Convert H0 to 1/Gyr
    H0_per_gyr = H0 * 1.0227e-3  # km/s/Mpc to 1/Gyr

    def integrand(a_prime):
        return 1.0 / (a_prime * np.sqrt(Om / a_prime**3 + OL))

    result, _ = integrate.quad(integrand, 0, a)
    return result / H0_per_gyr


def interpolate_frames(
    overdensity: np.ndarray,
    scale_factors: np.ndarray,
    n_interp: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate additional frames between snapshots.

    Parameters
    ----------
    overdensity : np.ndarray
        Shape (n_snapshots, L_pix, L_pix)
    scale_factors : np.ndarray
        Scale factors for each snapshot
    n_interp : int
        Number of interpolated frames between each pair of snapshots

    Returns
    -------
    interp_overdensity, interp_scale_factors : tuple of np.ndarray
    """
    if n_interp <= 1:
        return overdensity, scale_factors

    n_orig = len(scale_factors)
    n_total = (n_orig - 1) * n_interp + 1
    L_pix = overdensity.shape[1]

    interp_overdensity = np.zeros((n_total, L_pix, L_pix), dtype=np.float32)
    interp_scale_factors = np.zeros(n_total)

    idx = 0
    for i in range(n_orig - 1):
        a_curr = scale_factors[i]
        a_next = scale_factors[i + 1]
        d_curr = overdensity[i]
        d_next = overdensity[i + 1]

        log_a_curr = np.log(a_curr)
        log_a_next = np.log(a_next)

        for j in range(n_interp):
            t = j / n_interp
            log_a_interp = log_a_curr + t * (log_a_next - log_a_curr)
            a_interp = np.exp(log_a_interp)

            # Interpolate density in log space
            eps = 1e-10
            log_d_curr = np.log(d_curr + eps)
            log_d_next = np.log(d_next + eps)
            log_d_interp = (1 - t) * log_d_curr + t * log_d_next
            d_interp = np.exp(log_d_interp) - eps

            interp_overdensity[idx] = d_interp
            interp_scale_factors[idx] = a_interp
            idx += 1

    # Add final frame
    interp_overdensity[idx] = overdensity[-1]
    interp_scale_factors[idx] = scale_factors[-1]

    return interp_overdensity, interp_scale_factors


def create_animation(
    timeseries_file: Path,
    output_file: Path,
    fps: int = 15,
    n_interp: int = 4,
    dpi: int = 150,
    cmap: str = 'cosmic_blue',
    show_time: bool = True,
):
    """
    Create animation from a timeseries HDF5 file.

    Parameters
    ----------
    timeseries_file : Path
        Input HDF5 file with overdensity and scale_factor datasets
    output_file : Path
        Output MP4 file
    fps : int
        Frames per second
    n_interp : int
        Number of interpolated frames between snapshots
    dpi : int
        Output DPI
    cmap : str
        Colormap name
    show_time : bool
        Whether to show time indicator overlay
    """
    check_ffmpeg()

    print(f"Loading timeseries from {timeseries_file}...")
    with h5py.File(timeseries_file, 'r') as f:
        overdensity = f['overdensity'][:]
        scale_factors = f['scale_factor'][:]
        box_size = f['Header'].attrs['box_size']

    n_snapshots = len(scale_factors)
    print(f"  Found {n_snapshots} snapshots (a={scale_factors.min():.4f} to {scale_factors.max():.4f})")

    # Interpolate frames
    if n_interp > 1:
        print(f"  Interpolating {n_interp}x frames...")
        overdensity, scale_factors = interpolate_frames(overdensity, scale_factors, n_interp)
        print(f"  Total frames: {len(scale_factors)}")

    # Enable LaTeX-style rendering
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Compute global normalization
    print("  Computing normalization...")
    all_data = overdensity.flatten()
    min_pos = all_data[all_data > 0].min() if np.any(all_data > 0) else 1e-10
    all_data_log = np.log10(np.maximum(all_data, min_pos))
    vmin = np.percentile(all_data_log, 0.5)
    vmax = np.percentile(all_data_log, 99.9)

    # Choose colormap
    if cmap == 'cosmic_blue':
        colormap = create_cosmic_blue_colormap()
    else:
        colormap = plt.get_cmap(cmap)

    # Create figure
    L_pix = overdensity.shape[1]
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')

    # Initial frame
    initial_data = overdensity[0]
    display_data = np.log10(np.maximum(initial_data, min_pos))

    im = ax.imshow(
        display_data,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal'
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Time indicator
    if show_time:
        a = scale_factors[0]
        z = 1.0 / a - 1.0 if a > 0 else np.inf

        text_a = ax.text(
            0.02, 0.98, f'a = {a:.4f}',
            transform=ax.transAxes,
            color='white',
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            fontfamily='monospace'
        )

        z_str = f'z = {z:.2f}' if z < 1000 else f'z = {z:.0f}'
        text_z = ax.text(
            0.02, 0.93, z_str,
            transform=ax.transAxes,
            color='white',
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace'
        )

        # Cosmic time
        t_gyr = cosmic_time_gyr(a)
        if not np.isnan(t_gyr):
            text_t = ax.text(
                0.02, 0.88, f't = {t_gyr:.2f} Gyr',
                transform=ax.transAxes,
                color='white',
                fontsize=12,
                verticalalignment='top',
                fontfamily='monospace'
            )
        else:
            text_t = None

        # Box size label
        sim_name = timeseries_file.stem.replace('_timeseries', '').replace('Uniform_', '').replace('_primary', '')
        text_sim = ax.text(
            0.98, 0.98, sim_name,
            transform=ax.transAxes,
            color='white',
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            fontfamily='monospace'
        )

        # Progress bar
        progress_bg = plt.Rectangle(
            (0.02, 0.02), 0.96, 0.015,
            transform=ax.transAxes,
            facecolor='gray',
            alpha=0.5
        )
        ax.add_patch(progress_bg)

        progress_bar = plt.Rectangle(
            (0.02, 0.02), 0.0, 0.015,
            transform=ax.transAxes,
            facecolor='white',
            alpha=0.8
        )
        ax.add_patch(progress_bar)
    else:
        text_a = text_z = text_t = text_sim = progress_bar = None

    plt.tight_layout()

    def update(frame):
        a = scale_factors[frame]
        density = overdensity[frame]

        display_data = np.log10(np.maximum(density, min_pos))
        im.set_array(display_data)

        artists = [im]

        if show_time:
            text_a.set_text(f'a = {a:.4f}')

            z = 1.0 / a - 1.0 if a > 0 else np.inf
            z_str = f'z = {z:.2f}' if z < 1000 else f'z = {z:.0f}'
            text_z.set_text(z_str)

            if text_t is not None:
                t_gyr = cosmic_time_gyr(a)
                if not np.isnan(t_gyr):
                    text_t.set_text(f't = {t_gyr:.2f} Gyr')

            # Update progress bar
            progress = frame / (len(scale_factors) - 1) if len(scale_factors) > 1 else 1.0
            progress_bar.set_width(0.96 * progress)

            artists.extend([text_a, text_z, progress_bar, text_sim])
            if text_t is not None:
                artists.append(text_t)

        return artists

    # Create animation
    n_frames = len(scale_factors)
    duration = n_frames / fps
    print(f"  Rendering animation ({n_frames} frames at {fps} fps = {duration:.1f}s)...")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True
    )

    # Save
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={'title': 'Cosmic Structure Evolution'},
        bitrate=5000
    )

    anim.save(str(output_file), writer=writer, dpi=dpi)
    plt.close()

    print(f"  Saved animation to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create animations from density timeseries HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('input_dir', type=str,
                        help='Directory containing density slice HDF5 files (with time_series subdir)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Frames per second (default: 15)')
    parser.add_argument('--interpolate', type=int, default=4,
                        help='Frame interpolation factor (default: 4)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--cmap', type=str, default='cosmic_blue',
                        help='Colormap (default: cosmic_blue)')
    parser.add_argument('--no-time', action='store_true',
                        help='Hide time indicator overlay')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find timeseries file
    timeseries_dir = input_dir / "time_series"
    if not timeseries_dir.exists():
        raise FileNotFoundError(f"time_series subdirectory not found: {timeseries_dir}")

    timeseries_files = list(timeseries_dir.glob("*_timeseries.h5"))
    if not timeseries_files:
        raise FileNotFoundError(f"No timeseries HDF5 files found in {timeseries_dir}")

    timeseries_file = timeseries_files[0]

    # Create animation output directory
    animation_dir = input_dir / "animation"
    animation_dir.mkdir(exist_ok=True)

    # Output filename
    output_file = animation_dir / timeseries_file.name.replace('.h5', '.mp4')

    create_animation(
        timeseries_file=timeseries_file,
        output_file=output_file,
        fps=args.fps,
        n_interp=args.interpolate,
        dpi=args.dpi,
        cmap=args.cmap,
        show_time=not args.no_time,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
