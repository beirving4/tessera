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
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from time_series_config import TimeSeriesConfig, load_projections, cosmic_time_gyr
from build_time_series import apply_colormap, create_cosmic_blue_colormap


# =============================================================================
# CALLOUT TEMPLATE SYSTEM
# =============================================================================
#
# Callouts are annotations that appear during specific time ranges in the
# animation. Use them to highlight cosmic events, point out features, or
# add explanatory text.
#
# To use callouts:
#   1. Define your callouts in the CALLOUT_TEMPLATE list below
#   2. Run the animation with --callouts flag
#
# Each callout is a dictionary with these fields:
#   - a_start: Scale factor when callout appears
#   - a_end: Scale factor when callout disappears
#   - text: The annotation text
#   - position: (x, y) in axes coordinates (0-1) or pixel coordinates
#   - position_type: 'axes' (default) or 'data'
#   - style: 'text', 'arrow', 'box', or 'circle'
#   - target: (x, y) target position for arrows (in same coords as position)
#   - color: Text/annotation color (default: 'white')
#   - fontsize: Font size (default: 12)
#   - alpha: Transparency 0-1 (default: 1.0)
#   - fade_in: Scale factor range for fade-in (default: 0, instant)
#   - fade_out: Scale factor range for fade-out (default: 0, instant)
#
# =============================================================================

@dataclass
class Callout:
    """A single callout annotation for the animation."""
    a_start: float                          # Scale factor when callout appears
    a_end: float                            # Scale factor when callout disappears
    text: str                               # Annotation text
    position: tuple = (0.5, 0.5)            # Position (x, y)
    position_type: str = 'axes'             # 'axes' (0-1) or 'data' (pixel coords)
    style: str = 'text'                     # 'text', 'arrow', 'box', 'circle'
    target: Optional[tuple] = None          # Target for arrows
    color: str = 'white'                    # Text/line color
    fontsize: int = 12                      # Font size
    alpha: float = 1.0                      # Base transparency
    fade_in: float = 0.0                    # Scale factor range for fade-in
    fade_out: float = 0.0                   # Scale factor range for fade-out
    bbox: bool = False                      # Add background box to text
    bbox_color: str = 'black'               # Background box color
    bbox_alpha: float = 0.7                 # Background box transparency


# =============================================================================
# CALLOUT TEMPLATE - EDIT THIS LIST TO ADD YOUR ANNOTATIONS
# =============================================================================
#
# Example callouts for a typical cosmological simulation.
# Modify these for your specific simulation and visualization!
#
CALLOUT_TEMPLATE: List[Callout] = [
    # Example: Label the early universe
    # Callout(
    #     a_start=0.01,
    #     a_end=0.05,
    #     text="Early Universe\nDensity fluctuations grow",
    #     position=(0.98, 0.85),
    #     style='text',
    #     fontsize=14,
    #     color='cyan',
    #     bbox=True,
    #     fade_in=0.01,
    #     fade_out=0.01,
    # ),

    # Example: Arrow pointing to first structure
    # Callout(
    #     a_start=0.1,
    #     a_end=0.3,
    #     text="First filaments form",
    #     position=(0.7, 0.8),
    #     target=(0.5, 0.6),  # Arrow points here
    #     style='arrow',
    #     fontsize=12,
    #     color='yellow',
    # ),

    # Example: Mark present day (a=1)
    # Callout(
    #     a_start=0.9,
    #     a_end=1.1,
    #     text="Present Day (z=0)",
    #     position=(0.98, 0.12),
    #     style='text',
    #     fontsize=16,
    #     fontweight='bold',
    #     color='gold',
    #     bbox=True,
    #     fade_in=0.05,
    #     fade_out=0.05,
    # ),

    # Example: Circle a region of interest
    # Callout(
    #     a_start=1.0,
    #     a_end=10.0,
    #     text="Massive cluster",
    #     position=(0.35, 0.45),  # Circle center
    #     target=(0.08,),        # Circle radius (single value tuple)
    #     style='circle',
    #     color='red',
    #     fontsize=10,
    # ),

    # Example: Far future label
    # Callout(
    #     a_start=50.0,
    #     a_end=100.0,
    #     text="Far Future\nStructure frozen",
    #     position=(0.98, 0.85),
    #     style='text',
    #     fontsize=14,
    #     color='orange',
    #     bbox=True,
    # ),
]


def compute_callout_alpha(callout: Callout, a: float) -> float:
    """Compute the alpha value for a callout at a given scale factor."""
    if a < callout.a_start or a > callout.a_end:
        return 0.0

    alpha = callout.alpha

    # Fade in
    if callout.fade_in > 0 and a < callout.a_start + callout.fade_in:
        fade_progress = (a - callout.a_start) / callout.fade_in
        alpha *= fade_progress

    # Fade out
    if callout.fade_out > 0 and a > callout.a_end - callout.fade_out:
        fade_progress = (callout.a_end - a) / callout.fade_out
        alpha *= fade_progress

    return max(0.0, min(1.0, alpha))


def create_callout_artists(
    ax: plt.Axes,
    callouts: List[Callout],
    L_pix: int
) -> List[Dict[str, Any]]:
    """Create matplotlib artists for all callouts."""
    callout_artists = []

    for callout in callouts:
        artists = {'callout': callout, 'elements': []}

        # Determine transform
        if callout.position_type == 'axes':
            transform = ax.transAxes
            pos = callout.position
        else:
            transform = ax.transData
            pos = callout.position

        # Create artists based on style
        if callout.style == 'text':
            bbox_props = None
            if callout.bbox:
                bbox_props = dict(
                    boxstyle='round,pad=0.3',
                    facecolor=callout.bbox_color,
                    alpha=callout.bbox_alpha,
                    edgecolor='none'
                )

            text = ax.text(
                pos[0], pos[1], callout.text,
                transform=transform,
                color=callout.color,
                fontsize=callout.fontsize,
                alpha=0,  # Start invisible
                verticalalignment='top',
                horizontalalignment='right' if pos[0] > 0.5 else 'left',
                bbox=bbox_props
            )
            artists['elements'].append(('text', text))

        elif callout.style == 'arrow':
            if callout.target is None:
                continue

            # Text at position
            bbox_props = None
            if callout.bbox:
                bbox_props = dict(
                    boxstyle='round,pad=0.3',
                    facecolor=callout.bbox_color,
                    alpha=callout.bbox_alpha,
                    edgecolor='none'
                )

            text = ax.annotate(
                callout.text,
                xy=callout.target,
                xytext=pos,
                xycoords=transform,
                textcoords=transform,
                color=callout.color,
                fontsize=callout.fontsize,
                alpha=0,
                arrowprops=dict(
                    arrowstyle='->',
                    color=callout.color,
                    lw=1.5,
                    alpha=0
                ),
                bbox=bbox_props,
                verticalalignment='center',
                horizontalalignment='center'
            )
            artists['elements'].append(('annotation', text))

        elif callout.style == 'box':
            if callout.target is None:
                continue

            # target is (width, height)
            width, height = callout.target
            rect = mpatches.Rectangle(
                (pos[0] - width/2, pos[1] - height/2),
                width, height,
                transform=transform,
                fill=False,
                edgecolor=callout.color,
                linewidth=2,
                alpha=0
            )
            ax.add_patch(rect)
            artists['elements'].append(('patch', rect))

            # Add label
            text = ax.text(
                pos[0], pos[1] + height/2 + 0.02, callout.text,
                transform=transform,
                color=callout.color,
                fontsize=callout.fontsize,
                alpha=0,
                verticalalignment='bottom',
                horizontalalignment='center'
            )
            artists['elements'].append(('text', text))

        elif callout.style == 'circle':
            if callout.target is None:
                continue

            radius = callout.target[0]
            circle = mpatches.Circle(
                pos,
                radius,
                transform=transform,
                fill=False,
                edgecolor=callout.color,
                linewidth=2,
                alpha=0
            )
            ax.add_patch(circle)
            artists['elements'].append(('patch', circle))

            # Add label above circle
            text = ax.text(
                pos[0], pos[1] + radius + 0.02, callout.text,
                transform=transform,
                color=callout.color,
                fontsize=callout.fontsize,
                alpha=0,
                verticalalignment='bottom',
                horizontalalignment='center'
            )
            artists['elements'].append(('text', text))

        callout_artists.append(artists)

    return callout_artists


def update_callout_artists(callout_artists: List[Dict], a: float) -> List:
    """Update callout artists for the current scale factor."""
    updated = []

    for artists in callout_artists:
        callout = artists['callout']
        alpha = compute_callout_alpha(callout, a)

        for elem_type, elem in artists['elements']:
            if elem_type == 'text':
                elem.set_alpha(alpha)
                updated.append(elem)
            elif elem_type == 'annotation':
                elem.set_alpha(alpha)
                if hasattr(elem, 'arrow_patch') and elem.arrow_patch:
                    elem.arrow_patch.set_alpha(alpha)
                updated.append(elem)
            elif elem_type == 'patch':
                elem.set_alpha(alpha)
                updated.append(elem)

    return updated


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
    show_time_indicator: bool = True,
    callouts: Optional[List[Callout]] = None
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
    callouts : List[Callout], optional
        List of callout annotations to display during animation.
        If None, uses CALLOUT_TEMPLATE from this module.
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

    # Initialize callouts
    if callouts is None:
        callouts = CALLOUT_TEMPLATE
    callout_artists = create_callout_artists(ax, callouts, L_pix)
    if callout_artists:
        print(f"Added {len(callout_artists)} callout annotations")

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

        # Update callouts
        if callout_artists:
            callout_elements = update_callout_artists(callout_artists, a)
            artists.extend(callout_elements)

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
    parser.add_argument(
        "--callouts", action="store_true",
        help="Enable callout annotations (edit CALLOUT_TEMPLATE in this file)"
    )
    parser.add_argument(
        "--no-callouts", action="store_true",
        help="Disable callout annotations even if CALLOUT_TEMPLATE is defined"
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

    # Determine callouts to use
    if args.no_callouts:
        callouts = []
    elif args.callouts or CALLOUT_TEMPLATE:
        callouts = CALLOUT_TEMPLATE
    else:
        callouts = []

    create_animation(
        config,
        output_path=args.output,
        show_time_indicator=not args.no_time,
        callouts=callouts
    )


if __name__ == "__main__":
    main()
