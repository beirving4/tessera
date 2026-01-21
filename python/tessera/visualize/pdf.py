"""
PDF and histogram visualization utilities.

Provides functions for plotting probability density functions
and histograms of density field values.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .core import (
    save_figure,
    setup_matplotlib_backend,
    setup_publication_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_pdf_histogram(
    data: NDArray[np.floating],
    output_path: str | Path | None = None,
    bins: NDArray[np.floating] | None = None,
    n_bins: int = 100,
    log_x: bool = True,
    log_y: bool = True,
    density: bool = True,
    label: str | None = None,
    color: str | None = None,
    alpha: float = 0.7,
    ax: Axes | None = None,
    xlabel: str = r"$1+\delta$",
    ylabel: str = r"${\rm PDF}$",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Plot histogram/PDF of data values.

    Parameters
    ----------
    data : NDArray
        Input data array (will be flattened).
    output_path : str or Path, optional
        If provided, save the figure to this path.
    bins : NDArray, optional
        Custom bin edges. If None, auto-generated based on log_x.
    n_bins : int
        Number of bins if bins is None.
    log_x : bool
        If True, use logarithmic x-axis and log-spaced bins.
    log_y : bool
        If True, use logarithmic y-axis.
    density : bool
        If True, normalize histogram to probability density.
    label : str, optional
        Label for the histogram (for legend).
    color : str, optional
        Histogram color.
    alpha : float
        Histogram transparency.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    xlabel, ylabel : str
        Axis labels.
    title : str, optional
        Figure title.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt

    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Flatten data
    flat = data.ravel()

    # Filter positive values for log scale
    if log_x:
        flat = flat[flat > 0]

    # Generate bins if not provided
    if bins is None:
        if log_x:
            data_min = np.percentile(flat, 0.1) if len(flat) > 0 else 0.01
            data_max = np.percentile(flat, 99.9) if len(flat) > 0 else 100
            bins = np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)
        else:
            bins = n_bins

    # Plot histogram
    ax.hist(flat, bins=bins, density=density, alpha=alpha, label=label, color=color)

    # Set scales
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if label is not None:
        ax.legend()

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax


def plot_pdf_comparison(
    data1: NDArray[np.floating],
    data2: NDArray[np.floating],
    labels: tuple[str, str] = ("tessera", "Reference"),
    colors: tuple[str, str] = ("#e74c3c", "#3498db"),
    output_path: str | Path | None = None,
    bins: NDArray[np.floating] | None = None,
    n_bins: int = 100,
    log_scale: bool = True,
    title: str | None = None,
    xlabel: str = r"$1+\delta$",
    ylabel: str = r"${\rm PDF}$",
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Plot overlaid PDFs for comparison.

    Parameters
    ----------
    data1, data2 : NDArray
        Input data arrays to compare.
    labels : tuple[str, str]
        Labels for the two datasets.
    colors : tuple[str, str]
        Colors for the two histograms.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    bins : NDArray, optional
        Custom bin edges. If None, auto-generated.
    n_bins : int
        Number of bins if bins is None.
    log_scale : bool
        If True, use log-log scale.
    title : str, optional
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Flatten data
    flat1 = data1.ravel()
    flat2 = data2.ravel()

    # Filter positive values for log scale
    if log_scale:
        flat1 = flat1[flat1 > 0]
        flat2 = flat2[flat2 > 0]

    # Generate shared bins if not provided
    if bins is None:
        all_data = np.concatenate([flat1, flat2])
        if log_scale:
            data_min = np.percentile(all_data, 0.1)
            data_max = np.percentile(all_data, 99.9)
            bins = np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)
        else:
            bins = n_bins

    # Plot histograms
    ax.hist(
        flat1, bins=bins, density=True, alpha=0.7,
        label=rf'${{\rm {labels[0]}}}$', color=colors[0]
    )
    ax.hist(
        flat2, bins=bins, density=True, alpha=0.7,
        label=rf'${{\rm {labels[1]}}}$', color=colors[1]
    )

    # Set scales
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax


def plot_pdf_with_fit(
    data: NDArray[np.floating],
    output_path: str | Path | None = None,
    bins: NDArray[np.floating] | None = None,
    n_bins: int = 100,
    fit_range: tuple[float, float] | None = None,
    label: str | None = None,
    color: str = "#3498db",
    fit_color: str = "#e74c3c",
    xlabel: str = r"$1+\delta$",
    ylabel: str = r"${\rm PDF}$",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes, dict[str, float]]:
    """
    Plot PDF with power-law fit in high-density tail.

    Parameters
    ----------
    data : NDArray
        Input data array.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    bins : NDArray, optional
        Custom bin edges.
    n_bins : int
        Number of bins if bins is None.
    fit_range : tuple[float, float], optional
        Range for power-law fit (min, max).
    label : str, optional
        Label for the data histogram.
    color : str
        Histogram color.
    fit_color : str
        Fit line color.
    xlabel, ylabel : str
        Axis labels.
    title : str, optional
        Figure title.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, Axes, dict]
        Figure, Axes, and fit parameters dict with 'slope' and 'intercept'.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Flatten and filter
    flat = data.ravel()
    flat = flat[flat > 0]

    # Generate bins
    if bins is None:
        data_min = np.percentile(flat, 0.1)
        data_max = np.percentile(flat, 99.9)
        bins = np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)

    # Compute histogram
    hist, bin_edges = np.histogram(flat, bins=bins, density=True)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Plot histogram
    ax.loglog(bin_centers, hist, 'o-', alpha=0.7, label=label, color=color, markersize=3)

    # Fit power law if requested
    fit_params = {}
    if fit_range is not None:
        fit_mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1]) & (hist > 0)
        if np.sum(fit_mask) > 2:
            log_x = np.log10(bin_centers[fit_mask])
            log_y = np.log10(hist[fit_mask])

            # Linear fit in log-log space
            coeffs = np.polyfit(log_x, log_y, 1)
            slope, intercept = coeffs

            fit_params['slope'] = float(slope)
            fit_params['intercept'] = float(intercept)

            # Plot fit
            fit_x = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 100)
            fit_y = 10**intercept * fit_x**slope

            ax.loglog(
                fit_x, fit_y, '--',
                color=fit_color, linewidth=2,
                label=rf'Fit: $\propto \rho^{{{slope:.2f}}}$'
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax, fit_params
