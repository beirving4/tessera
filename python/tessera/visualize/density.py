"""
Density field visualization utilities.

Provides functions for visualizing 2D density slices and projections,
including single-panel plots and multi-panel comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .core import (
    add_colorbar,
    percentile_clim,
    prepare_log_data,
    save_figure,
    setup_matplotlib_backend,
    setup_publication_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_density_slice(
    density: NDArray[np.floating],
    output_path: str | Path | None = None,
    extent: tuple[float, float, float, float] | None = None,
    cmap: str = "magma",
    log_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.9,
    colorbar_label: str = r"$\log_{10}(1+\delta)$",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Create a 2D density slice visualization with colorbar.

    Parameters
    ----------
    density : NDArray
        2D density array to visualize.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    extent : tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax).
    cmap : str
        Matplotlib colormap name.
    log_scale : bool
        If True, use logarithmic color scale.
    vmin, vmax : float, optional
        Explicit color limits. If None, computed from percentiles.
    vmin_pct, vmax_pct : float
        Percentiles for automatic vmin/vmax computation.
    colorbar_label : str
        Label for the colorbar.
    title : str, optional
        Figure title.
    xlabel, ylabel : str, optional
        Axis labels.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple[float, float]
        Figure size in inches (width, height).
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_publication_style()

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Prepare data for log scale
    plot_data = density.copy()
    if log_scale:
        plot_data = prepare_log_data(plot_data)

    # Compute color limits
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = percentile_clim(
            plot_data, vmin_pct, vmax_pct, positive_only=log_scale
        )
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    # Create normalization
    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
        # For linear scale, apply vmin/vmax directly

    # Plot image
    im = ax.imshow(
        plot_data.T,
        origin='lower',
        cmap=cmap,
        norm=norm if log_scale else None,
        vmin=None if log_scale else vmin,
        vmax=None if log_scale else vmax,
        extent=extent,
    )

    # Add colorbar
    add_colorbar(ax, im, label=colorbar_label)

    # Set labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax


def plot_density_comparison(
    density1: NDArray[np.floating],
    density2: NDArray[np.floating],
    labels: tuple[str, str] = ("tessera", "Reference"),
    output_path: str | Path | None = None,
    extent: tuple[float, float, float, float] | None = None,
    show_difference: bool = True,
    cmap: str = "magma",
    diff_cmap: str = "RdBu_r",
    title: str | None = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.9,
    figsize: tuple[float, float] = (15, 5),
    dpi: int = 150,
) -> tuple[Figure, NDArray[np.floating]]:
    """
    Create side-by-side density comparison with optional difference panel.

    Parameters
    ----------
    density1, density2 : NDArray
        2D density arrays to compare.
    labels : tuple[str, str]
        Labels for the two density fields.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    extent : tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax).
    show_difference : bool
        If True, include a third panel showing relative difference.
    cmap : str
        Colormap for density plots.
    diff_cmap : str
        Colormap for difference plot (diverging recommended).
    title : str, optional
        Figure title.
    vmin_pct, vmax_pct : float
        Percentiles for automatic vmin/vmax computation.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, NDArray]
        The matplotlib Figure and the relative difference array.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    setup_publication_style()

    # Normalize both fields
    norm1 = density1 / np.mean(density1)
    norm2 = density2 / np.mean(density2)

    # Compute relative difference
    mask = norm2 > 0
    rel_diff = np.zeros_like(norm1)
    rel_diff[mask] = (norm1[mask] - norm2[mask]) / norm2[mask]

    # Determine number of panels
    n_panels = 3 if show_difference else 2

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # Compute shared color limits
    plot_data1 = prepare_log_data(norm1)
    plot_data2 = prepare_log_data(norm2)
    vmin = min(
        percentile_clim(plot_data1, vmin_pct, vmax_pct, positive_only=True)[0],
        percentile_clim(plot_data2, vmin_pct, vmax_pct, positive_only=True)[0],
    )
    vmax = max(
        percentile_clim(plot_data1, vmin_pct, vmax_pct, positive_only=True)[1],
        percentile_clim(plot_data2, vmin_pct, vmax_pct, positive_only=True)[1],
    )

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Plot first density
    im1 = axes[0].imshow(
        np.log10(plot_data1.T),
        origin='lower',
        cmap=cmap,
        vmin=np.log10(vmin),
        vmax=np.log10(vmax),
        extent=extent,
    )
    axes[0].set_title(rf'${{\rm {labels[0]}}}$')
    add_colorbar(axes[0], im1, label=r'$\log_{10}(1+\delta)$')

    # Plot second density
    im2 = axes[1].imshow(
        np.log10(plot_data2.T),
        origin='lower',
        cmap=cmap,
        vmin=np.log10(vmin),
        vmax=np.log10(vmax),
        extent=extent,
    )
    axes[1].set_title(rf'${{\rm {labels[1]}}}$')
    add_colorbar(axes[1], im2, label=r'$\log_{10}(1+\delta)$')

    # Plot difference if requested
    if show_difference:
        # Use percentile for symmetric limits
        finite_diff = rel_diff[np.isfinite(rel_diff)]
        diff_abs_max = np.percentile(np.abs(finite_diff), 99) if len(finite_diff) > 0 else 1.0

        im3 = axes[2].imshow(
            rel_diff.T,
            origin='lower',
            cmap=diff_cmap,
            vmin=-diff_abs_max,
            vmax=diff_abs_max,
            extent=extent,
        )
        axes[2].set_title(r'${\rm Relative\ Difference}$')
        add_colorbar(
            axes[2], im3,
            label=rf'$({labels[0]} - {labels[1]}) / {labels[1]}$'
        )

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, rel_diff
