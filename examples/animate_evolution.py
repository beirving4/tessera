#!/usr/bin/env python3
"""
Create an animation of cosmic structure evolution from density projections.

This is Step 3 of the time-series visualization pipeline.

Usage:
    python animate_evolution.py --projections projections.h5 --output evolution.mp4
"""

import argparse
import shutil
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time_series_config import TimeSeriesConfig, load_projections, cosmic_time_gyr
from build_time_series import apply_colormap, create_cosmic_blue_colormap


def check_ffmpeg():
    """Check if ffmpeg is available for animation encoding."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg for animation creation.\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  conda: conda install ffmpeg"
        )


def interpolate_frames(
    density_maps: dict,
    n_interp: int = 4
) -> dict:
    """
    Interpolate additional frames between snapshots.

    Parameters
    ----------
    density_maps : dict
        Dictionary mapping scale factor to 2D density arrays
    n_interp : int
        Number of interpolated frames between each pair of snapshots

    Returns
    -------
    interpolated : dict
        New dictionary with original + interpolated frames
    """
    if n_interp <= 1:
        return density_maps

    a_values = np.array(sorted(density_maps.keys()))
    interpolated = {}

    for i in range(len(a_values)):
        a_curr = a_values[i]
        interpolated[a_curr] = density_maps[a_curr]

        if i < len(a_values) - 1:
            a_next = a_values[i + 1]

            # Interpolate in log(a) space
            log_a_curr = np.log(a_curr)
            log_a_next = np.log(a_next)

            for j in range(1, n_interp):
                t = j / n_interp
                log_a_interp = log_a_curr + t * (log_a_next - log_a_curr)
                a_interp = np.exp(log_a_interp)

                # Interpolate density in log space
                d_curr = density_maps[a_curr]
                d_next = density_maps[a_next]

                eps = 1e-10
                log_d_curr = np.log(d_curr + eps)
                log_d_next = np.log(d_next + eps)
                log_d_interp = (1 - t) * log_d_curr + t * log_d_next
                d_interp = np.exp(log_d_interp) - eps

                interpolated[a_interp] = d_interp

    return interpolated


def create_animation(
    config: TimeSeriesConfig,
    output_path: Path = None,
    show_time_indicator: bool = True
):
    """
    Create animation from density projections.

    Parameters
    ----------
    config : TimeSeriesConfig
        Pipeline configuration
    output_path : Path, optional
        Output file path (default: config.animation_file)
    show_time_indicator : bool
        Whether to show scale factor / redshift / cosmic time labels
    """
    # Check for ffmpeg
    check_ffmpeg()

    if output_path is None:
        output_path = config.animation_file

    # Enable LaTeX-style rendering
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Load projections
    print(f"Loading projections from {config.projections_file}...")
    density_maps = load_projections(config.projections_file)

    if not density_maps:
        print("Error: No projections found")
        return

    # Interpolate frames if requested
    if config.interpolate_frames > 1:
        print(f"Interpolating {config.interpolate_frames}x frames...")
        density_maps = interpolate_frames(density_maps, config.interpolate_frames)

    a_values = np.array(sorted(density_maps.keys()))
    print(f"Total frames: {len(a_values)}")

    # Pre-compute normalized frames for consistent colormap
    print("Normalizing frames...")

    # Find global percentiles across all frames for consistent scaling
    all_densities = np.concatenate([d.flatten() for d in density_maps.values()])
    if config.log_density_scale:
        min_pos = all_densities[all_densities > 0].min()
        all_densities_log = np.log10(np.maximum(all_densities, min_pos))
        vmin = np.percentile(all_densities_log, config.percentile_clip[0])
        vmax = np.percentile(all_densities_log, config.percentile_clip[1])
    else:
        vmin = np.percentile(all_densities, config.percentile_clip[0])
        vmax = np.percentile(all_densities, config.percentile_clip[1])

    # Choose colormap
    if config.colormap == 'cosmic_blue':
        cmap = create_cosmic_blue_colormap()
    else:
        cmap = plt.get_cmap(config.colormap)

    # Create figure
    L_pix = density_maps[a_values[0]].shape[0]
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')

    # Initial frame
    initial_density = density_maps[a_values[0]]
    if config.log_density_scale:
        min_pos = initial_density[initial_density > 0].min() if np.any(initial_density > 0) else 1e-10
        display_data = np.log10(np.maximum(initial_density, min_pos))
    else:
        display_data = initial_density

    im = ax.imshow(
        display_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal'
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add time indicator
    if show_time_indicator:
        a = a_values[0]
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

        if z >= 0 and z < 1000:
            z_str = f'z = {z:.2f}'
        elif z >= 0:
            z_str = f'z = {z:.0f}'
        else:
            z_str = f'z = {z:.2f}'

        text_z = ax.text(
            0.02, 0.93, z_str,
            transform=ax.transAxes,
            color='white',
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace'
        )

        # Try to add cosmic time (requires scipy)
        try:
            t_gyr = cosmic_time_gyr(a)
            text_t = ax.text(
                0.02, 0.88, f't = {t_gyr:.2f} Gyr',
                transform=ax.transAxes,
                color='white',
                fontsize=12,
                verticalalignment='top',
                fontfamily='monospace'
            )
        except ImportError:
            text_t = None

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
        text_a = text_z = text_t = progress_bar = None

    plt.tight_layout()

    def update(frame):
        a = a_values[frame]
        density = density_maps[a]

        if config.log_density_scale:
            min_pos = density[density > 0].min() if np.any(density > 0) else 1e-10
            display_data = np.log10(np.maximum(density, min_pos))
        else:
            display_data = density

        im.set_array(display_data)

        artists = [im]

        if show_time_indicator:
            text_a.set_text(f'a = {a:.4f}')

            z = 1.0 / a - 1.0 if a > 0 else np.inf
            if z >= 0 and z < 1000:
                z_str = f'z = {z:.2f}'
            elif z >= 0:
                z_str = f'z = {z:.0f}'
            else:
                z_str = f'z = {z:.2f}'
            text_z.set_text(z_str)

            if text_t is not None:
                try:
                    t_gyr = cosmic_time_gyr(a)
                    text_t.set_text(f't = {t_gyr:.2f} Gyr')
                except:
                    pass

            # Update progress bar
            progress = frame / (len(a_values) - 1) if len(a_values) > 1 else 1.0
            progress_bar.set_width(0.96 * progress)

            artists.extend([text_a, text_z, progress_bar])
            if text_t is not None:
                artists.append(text_t)

        return artists

    # Create animation
    print(f"Rendering animation ({len(a_values)} frames at {config.fps} fps)...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(a_values),
        interval=1000 / config.fps,
        blit=True
    )

    # Save
    writer = animation.FFMpegWriter(
        fps=config.fps,
        metadata={'title': 'Cosmic Structure Evolution'},
        bitrate=5000
    )

    anim.save(str(output_path), writer=writer, dpi=config.dpi)
    plt.close()

    duration = len(a_values) / config.fps
    print(f"Saved animation to: {output_path} ({duration:.1f} seconds)")


def main():
    parser = argparse.ArgumentParser(
        description="Create animation of cosmic structure evolution"
    )
    parser.add_argument(
        "--projections", type=Path, required=True,
        help="HDF5 file containing density projections"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output video file (MP4)"
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second (default: 15)"
    )
    parser.add_argument(
        "--interpolate", type=int, default=1,
        help="Interpolation factor between snapshots (default: 1 = none)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)"
    )
    parser.add_argument(
        "--colormap", type=str, default="cosmic_blue",
        help="Colormap (default: cosmic_blue)"
    )
    parser.add_argument(
        "--no-time", action="store_true",
        help="Hide time indicator overlay"
    )

    args = parser.parse_args()

    config = TimeSeriesConfig(
        snapshot_dir=Path("."),
        output_dir=args.output.parent,
        fps=args.fps,
        interpolate_frames=args.interpolate,
        dpi=args.dpi,
        colormap=args.colormap,
        _projections_file_override=args.projections
    )

    create_animation(
        config,
        output_path=args.output,
        show_time_indicator=not args.no_time
    )


if __name__ == "__main__":
    main()
