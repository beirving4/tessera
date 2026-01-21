"""
Halo Evolution Visualization Functions.

Generate animations and multi-panel figures from density fields,
showing the evolution of dark matter halos across cosmic time.

All functions use type hints compatible with Python 3.10+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EvolutionVisualizationConfig:
    """
    Configuration for evolution visualization output.

    Default values are chosen to match the reference figure style
    (halo_density_map_evo.pdf).

    Attributes
    ----------
    colormap : str
        Colormap name (default matches reference figure)
    log_scale : bool
        Use logarithmic color scale
    percentile_clip : tuple
        Percentile range for color normalization
    global_normalization : bool
        If False, each panel gets own colorbar range (like reference)
    show_overdensity : bool
        Plot 1+delta instead of raw density
    show_scale_factor : bool
        Show scale factor label in corner
    scale_factor_position : str
        Position of scale factor label
    show_mass : bool
        Show M200 label
    show_r200_circle : bool
        Show R200 circle overlay
    show_scale_bar : bool
        Show physical scale bar
    center_on_halo : bool
        Center axes at (0,0)
    axis_label_unit : str
        Unit label for axes
    fps : int
        Frames per second for animation
    dpi : int
        DPI for output images
    panel_size : float
        Size of each panel in inches
    colorbar_per_panel : bool
        Each panel has own colorbar (like reference)
    """

    colormap: str = 'viridis'
    log_scale: bool = True
    percentile_clip: Tuple[float, float] = (0.5, 99.9)
    global_normalization: bool = False
    show_overdensity: bool = True
    show_scale_factor: bool = True
    scale_factor_position: str = 'upper-left'
    show_mass: bool = False
    show_r200_circle: bool = False
    show_scale_bar: bool = False
    center_on_halo: bool = True
    axis_label_unit: str = 'cMpc/h'
    fps: int = 15
    dpi: int = 150
    panel_size: float = 3.5
    colorbar_per_panel: bool = True


def _find_nearest_epoch(
    available_epochs: List[float],
    target: float,
) -> float:
    """Find the available epoch nearest to target."""
    return min(available_epochs, key=lambda x: abs(x - target))


def auto_select_epochs(
    available_epochs: List[float],
    n_panels: int,
    log_spacing: bool = True,
) -> List[float]:
    """
    Automatically select epochs for multi-panel figure.

    Parameters
    ----------
    available_epochs : list
        Available scale factors
    n_panels : int
        Number of panels to select
    log_spacing : bool
        Use logarithmic spacing (default True)

    Returns
    -------
    selected : list
        Selected scale factors
    """
    epochs = sorted(available_epochs)

    if len(epochs) <= n_panels:
        return epochs

    a_min, a_max = epochs[0], epochs[-1]

    if log_spacing and a_min > 0:
        # Log-spaced targets
        targets = np.logspace(np.log10(a_min), np.log10(a_max), n_panels)
    else:
        # Linear spacing
        targets = np.linspace(a_min, a_max, n_panels)

    # Find nearest available epoch for each target
    selected = []
    for t in targets:
        nearest = _find_nearest_epoch(epochs, t)
        if nearest not in selected:
            selected.append(nearest)

    return sorted(selected)


def create_evolution_multipanel(
    densities: Dict[float, np.ndarray],
    branch: List,
    output_path: str | Path,
    select_epochs: List[float] | None = None,
    config: EvolutionVisualizationConfig | None = None,
    box_size: float = 1.5,
    mean_density: float = 1.0,
) -> None:
    """
    Create multi-panel figure showing key evolutionary stages.

    Default epochs [0.1, 1.0, 10.0, 100.0] match the reference figure style.

    Parameters
    ----------
    densities : dict
        Mapping of scale_factor -> 2D density array
    branch : List[HaloInfo]
        Halo info for annotations
    output_path : Path
        Output image file path
    select_epochs : List[float], optional
        Specific scale factors to show.
        Default: [0.1, 1.0, 10.0, 100.0] to match reference figure.
    config : EvolutionVisualizationConfig
        Visualization settings
    box_size : float
        Physical size of the density field
    mean_density : float
        Mean density for overdensity calculation
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return

    cfg = config or EvolutionVisualizationConfig()

    # Default epochs to match reference figure
    if select_epochs is None:
        select_epochs = [0.1, 1.0, 10.0, 100.0]

    # Find nearest available epochs
    available = sorted(densities.keys())
    epochs_to_plot = []
    for target in select_epochs:
        nearest = _find_nearest_epoch(available, target)
        if nearest not in epochs_to_plot:
            epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)
    if n_panels == 0:
        print("No epochs available to plot")
        return

    # Create figure
    fig_width = cfg.panel_size * n_panels + 1  # Extra space for colorbars
    fig_height = cfg.panel_size + 0.8  # Extra for labels

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_width, fig_height),
        dpi=cfg.dpi,
    )

    if n_panels == 1:
        axes = [axes]

    # Compute extent (centered on halo)
    half_size = box_size / 2
    extent = (-half_size, half_size, -half_size, half_size)

    # Create halo lookup
    halo_lookup = {}
    for h in branch:
        scale_factor = getattr(h, 'scale_factor', None)
        if scale_factor is not None:
            halo_lookup[scale_factor] = h

    for ax, epoch in zip(axes, epochs_to_plot):
        density = densities[epoch]

        # Convert to overdensity if requested
        if cfg.show_overdensity:
            plot_data = density / mean_density
        else:
            plot_data = density

        # Handle zeros for log scale
        if cfg.log_scale:
            positive_mask = plot_data > 0
            if positive_mask.any():
                min_positive = plot_data[positive_mask].min()
                plot_data = np.maximum(plot_data, min_positive * 0.1)
            else:
                plot_data = np.maximum(plot_data, 1e-10)

        # Color normalization
        if cfg.log_scale:
            vmin = np.percentile(plot_data[plot_data > 0], cfg.percentile_clip[0])
            vmax = np.percentile(plot_data, cfg.percentile_clip[1])
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = np.percentile(plot_data, cfg.percentile_clip[0])
            vmax = np.percentile(plot_data, cfg.percentile_clip[1])
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot image
        im = ax.imshow(
            plot_data,
            extent=extent,
            origin='lower',
            cmap=cfg.colormap,
            norm=norm,
            aspect='equal',
            interpolation='nearest',
        )

        # Colorbar
        if cfg.colorbar_per_panel:
            cbar = fig.colorbar(im, ax=ax, location='top', pad=0.02, shrink=0.9)
            if cfg.show_overdensity:
                cbar.set_label(r'$1 + \delta(a)$', fontsize=10)
            else:
                cbar.set_label(r'$\rho$', fontsize=10)

        # Scale factor annotation (matching reference style with underline)
        if cfg.show_scale_factor:
            label = f'$a = {epoch:.2f}$'
            ax.text(
                0.05, 0.95, label,
                transform=ax.transAxes,
                fontsize=11,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
            )

        # Axis labels
        ax.set_xlabel(f'$x$ [{cfg.axis_label_unit}]', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel(f'$y$ [{cfg.axis_label_unit}]', fontsize=10)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        # R200 circle if requested
        if cfg.show_r200_circle:
            nearest_epoch = _find_nearest_epoch(
                list(halo_lookup.keys()), epoch
            ) if halo_lookup else None
            halo = halo_lookup.get(nearest_epoch)
            if halo:
                r200c = getattr(halo, 'r200c', 0)
                if r200c > 0:
                    from matplotlib.patches import Circle
                    circle = Circle(
                        (0, 0), r200c,
                        fill=False, color='white',
                        linestyle='--', linewidth=1,
                    )
                    ax.add_patch(circle)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
    plt.close()

    print(f"Saved multi-panel figure to {output_path}")


def create_evolution_animation(
    densities: Dict[float, np.ndarray],
    branch: List,
    output_path: str | Path,
    config: EvolutionVisualizationConfig | None = None,
    box_size: float = 1.5,
    mean_density: float = 1.0,
) -> None:
    """
    Create MP4 animation of halo evolution.

    Parameters
    ----------
    densities : dict
        Mapping of scale_factor -> 2D density array
    branch : List[HaloInfo]
        Halo info for annotations
    output_path : Path
        Output MP4 file path
    config : EvolutionVisualizationConfig
        Visualization settings
    box_size : float
        Physical size of the density field
    mean_density : float
        Mean density for overdensity calculation
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("matplotlib not available - skipping animation")
        return

    cfg = config or EvolutionVisualizationConfig()

    # Sort epochs
    epochs = sorted(densities.keys())
    if len(epochs) == 0:
        print("No epochs available for animation")
        return

    # Compute extent
    half_size = box_size / 2
    extent = (-half_size, half_size, -half_size, half_size)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=cfg.dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Initial frame
    first_density = densities[epochs[0]]
    if cfg.show_overdensity:
        first_data = first_density / mean_density
    else:
        first_data = first_density

    if cfg.log_scale:
        positive_mask = first_data > 0
        if positive_mask.any():
            min_pos = first_data[positive_mask].min()
            first_data = np.maximum(first_data, min_pos * 0.1)
            vmin = np.percentile(first_data[positive_mask], cfg.percentile_clip[0])
            vmax = np.percentile(first_data, cfg.percentile_clip[1])
        else:
            vmin, vmax = 1, 100
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = np.percentile(first_data, cfg.percentile_clip[0])
        vmax = np.percentile(first_data, cfg.percentile_clip[1])
        norm = None

    im = ax.imshow(
        first_data,
        extent=extent,
        origin='lower',
        cmap=cfg.colormap,
        norm=norm,
        aspect='equal',
        interpolation='nearest',
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if cfg.show_overdensity:
        cbar.set_label(r'$1 + \delta$', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Scale factor text
    text = ax.text(
        0.05, 0.95, '',
        transform=ax.transAxes,
        fontsize=14,
        color='white',
        verticalalignment='top',
        fontweight='bold',
    )

    ax.set_xlabel(f'$x$ [{cfg.axis_label_unit}]', color='white', fontsize=12)
    ax.set_ylabel(f'$y$ [{cfg.axis_label_unit}]', color='white', fontsize=12)
    ax.tick_params(colors='white')

    def update(frame_idx):
        epoch = epochs[frame_idx]
        density = densities[epoch]

        if cfg.show_overdensity:
            plot_data = density / mean_density
        else:
            plot_data = density

        if cfg.log_scale:
            positive_mask = plot_data > 0
            if positive_mask.any():
                min_pos = plot_data[positive_mask].min()
                plot_data = np.maximum(plot_data, min_pos * 0.1)

        # Update color normalization per frame (like reference)
        if cfg.log_scale and not cfg.global_normalization:
            vmin = np.percentile(plot_data[plot_data > 0], cfg.percentile_clip[0])
            vmax = np.percentile(plot_data, cfg.percentile_clip[1])
            im.set_norm(LogNorm(vmin=vmin, vmax=vmax))

        im.set_data(plot_data)
        text.set_text(f'$a = {epoch:.2f}$')

        return [im, text]

    anim = FuncAnimation(
        fig, update,
        frames=len(epochs),
        interval=1000 // cfg.fps,
        blit=True,
    )

    # Save animation
    output_path = Path(output_path)
    if output_path.suffix.lower() == '.mp4':
        writer = FFMpegWriter(fps=cfg.fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    elif output_path.suffix.lower() == '.gif':
        anim.save(output_path, writer='pillow', fps=cfg.fps)
    else:
        # Default to mp4
        writer = FFMpegWriter(fps=cfg.fps, bitrate=2000)
        anim.save(output_path, writer=writer)

    plt.close()
    print(f"Saved animation to {output_path}")


# Aliases for backwards compatibility
VisualizationConfig = EvolutionVisualizationConfig
create_multipanel_figure = create_evolution_multipanel


__all__ = [
    'EvolutionVisualizationConfig',
    'VisualizationConfig',  # Alias
    'auto_select_epochs',
    'create_evolution_multipanel',
    'create_multipanel_figure',  # Alias
    'create_evolution_animation',
]
