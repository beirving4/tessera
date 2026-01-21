"""
ORIGAMI morphology visualization utilities.

Provides functions for visualizing morphology classification grids,
pie charts of mass/volume fractions, and comparison bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .core import (
    add_colorbar,
    save_figure,
    setup_matplotlib_backend,
    setup_publication_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure


# Standard colors for ORIGAMI morphology classes
MORPHOLOGY_COLORS: list[str] = [
    '#3498db',  # Void: blue
    '#2ecc71',  # Wall: green
    '#e74c3c',  # Filament: red
    '#9b59b6',  # Halo: purple
]

# Alternative viridis-style colors (matches some literature)
MORPHOLOGY_COLORS_VIRIDIS: list[str] = [
    '#440154',  # Void: dark purple
    '#31688e',  # Wall: blue
    '#35b779',  # Filament: green
    '#fde725',  # Halo: yellow
]

# Class names
MORPHOLOGY_NAMES: list[str] = ['Void', 'Wall', 'Filament', 'Halo']

# LaTeX-formatted class names
MORPHOLOGY_NAMES_LATEX: list[str] = [
    r'${\rm Void}$',
    r'${\rm Wall}$',
    r'${\rm Filament}$',
    r'${\rm Halo}$',
]


def create_morphology_cmap(
    colors: list[str] | None = None,
) -> tuple[ListedColormap, BoundaryNorm]:
    """
    Create discrete colormap for ORIGAMI morphology classes (0-3).

    Parameters
    ----------
    colors : list[str], optional
        Custom colors for [Void, Wall, Filament, Halo].
        If None, uses MORPHOLOGY_COLORS.

    Returns
    -------
    tuple[ListedColormap, BoundaryNorm]
        Colormap and norm for discrete morphology visualization.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if colors is None:
        colors = MORPHOLOGY_COLORS

    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def plot_morphology_slice(
    morphology_grid: NDArray[np.integer],
    output_path: str | Path | None = None,
    slice_axis: int = 0,
    slice_index: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    show_colorbar: bool = True,
    colors: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Plot 2D slice of ORIGAMI morphology grid.

    Parameters
    ----------
    morphology_grid : NDArray
        3D array of morphology classifications (0=Void, 1=Wall, 2=Filament, 3=Halo).
    output_path : str or Path, optional
        If provided, save the figure to this path.
    slice_axis : int
        Axis along which to take the slice (0=x, 1=y, 2=z).
    slice_index : int, optional
        Index of the slice. If None, uses middle of the axis.
    extent : tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax).
    title : str, optional
        Figure title.
    show_colorbar : bool
        If True, display a colorbar with class labels.
    colors : list[str], optional
        Custom colors for morphology classes.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
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

    # Get slice
    if slice_index is None:
        slice_index = morphology_grid.shape[slice_axis] // 2

    if slice_axis == 0:
        slice_data = morphology_grid[slice_index, :, :]
    elif slice_axis == 1:
        slice_data = morphology_grid[:, slice_index, :]
    else:  # slice_axis == 2
        slice_data = morphology_grid[:, :, slice_index]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create colormap
    cmap, norm = create_morphology_cmap(colors)

    # Plot
    im = ax.imshow(
        slice_data.T,
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=extent,
    )

    # Add colorbar
    if show_colorbar:
        # Position ticks at center of each color band
        ticks = [0, 1, 2, 3]
        add_colorbar(
            ax, im,
            ticks=ticks,
            ticklabels=MORPHOLOGY_NAMES_LATEX,
        )

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax


def plot_morphology_slices_3panel(
    morphology_grid: NDArray[np.integer],
    output_path: str | Path | None = None,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (15, 5),
    dpi: int = 150,
) -> tuple[Figure, NDArray]:
    """
    Plot three orthogonal slices of morphology grid (XY, XZ, YZ).

    Parameters
    ----------
    morphology_grid : NDArray
        3D array of morphology classifications.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    extent : tuple[float, float, float, float], optional
        Box extent (assumes cubic, used for labeling).
    title : str, optional
        Figure title.
    colors : list[str], optional
        Custom colors for morphology classes.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    tuple[Figure, NDArray]
        The matplotlib Figure and array of Axes objects.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt

    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    cmap, norm = create_morphology_cmap(colors)
    n_cells = morphology_grid.shape[0]

    # X-Y slice (z=middle)
    z_mid = n_cells // 2
    im0 = axes[0].imshow(
        morphology_grid[z_mid, :, :].T,
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    axes[0].set_title(rf'$\rm X\text{{-}}Y\ Slice\ (z={z_mid})$')
    axes[0].set_xlabel(r'$X$')
    axes[0].set_ylabel(r'$Y$')
    add_colorbar(axes[0], im0, ticks=[0, 1, 2, 3], ticklabels=MORPHOLOGY_NAMES_LATEX)

    # X-Z slice (y=middle)
    y_mid = n_cells // 2
    im1 = axes[1].imshow(
        morphology_grid[:, y_mid, :].T,
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    axes[1].set_title(rf'$\rm X\text{{-}}Z\ Slice\ (y={y_mid})$')
    axes[1].set_xlabel(r'$X$')
    axes[1].set_ylabel(r'$Z$')
    add_colorbar(axes[1], im1, ticks=[0, 1, 2, 3], ticklabels=MORPHOLOGY_NAMES_LATEX)

    # Y-Z slice (x=middle)
    x_mid = n_cells // 2
    im2 = axes[2].imshow(
        morphology_grid[:, :, x_mid].T,
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    axes[2].set_title(rf'$\rm Y\text{{-}}Z\ Slice\ (x={x_mid})$')
    axes[2].set_xlabel(r'$Y$')
    axes[2].set_ylabel(r'$Z$')
    add_colorbar(axes[2], im2, ticks=[0, 1, 2, 3], ticklabels=MORPHOLOGY_NAMES_LATEX)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, axes


def plot_morphology_fractions_pie(
    fractions: dict[str, float],
    output_path: str | Path | None = None,
    title: str = r"${\rm Mass\ Fractions}$",
    colors: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Create pie chart of morphology fractions.

    Parameters
    ----------
    fractions : dict[str, float]
        Dictionary with keys 'void', 'wall', 'filament', 'halo' and
        fraction values summing to 1.0.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    title : str
        Chart title.
    colors : list[str], optional
        Custom colors for morphology classes.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
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

    if colors is None:
        colors = MORPHOLOGY_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Extract fractions in standard order
    values = [
        fractions.get('void', 0),
        fractions.get('wall', 0),
        fractions.get('filament', 0),
        fractions.get('halo', 0),
    ]

    ax.pie(
        values,
        labels=MORPHOLOGY_NAMES_LATEX,
        colors=colors,
        autopct='%.1f%%',
        startangle=90,
    )
    ax.set_title(title)

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax


def plot_morphology_comparison_bar(
    fractions1: dict[str, float],
    fractions2: dict[str, float],
    labels: tuple[str, str] = ("Original", "tessera"),
    output_path: str | Path | None = None,
    title: str = r"${\rm Classification\ Comparison}$",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Create grouped bar chart comparing morphology fractions.

    Parameters
    ----------
    fractions1, fractions2 : dict[str, float]
        Dictionaries with keys 'void', 'wall', 'filament', 'halo'.
    labels : tuple[str, str]
        Labels for the two datasets.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    title : str
        Chart title.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
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

    x = np.arange(4)
    width = 0.35

    # Extract values (convert to percentages)
    values1 = [
        fractions1.get('void', 0) * 100,
        fractions1.get('wall', 0) * 100,
        fractions1.get('filament', 0) * 100,
        fractions1.get('halo', 0) * 100,
    ]
    values2 = [
        fractions2.get('void', 0) * 100,
        fractions2.get('wall', 0) * 100,
        fractions2.get('filament', 0) * 100,
        fractions2.get('halo', 0) * 100,
    ]

    ax.bar(
        x - width/2, values1, width,
        label=rf'${{\rm {labels[0]}}}$',
        color='#2c3e50',
    )
    ax.bar(
        x + width/2, values2, width,
        label=rf'${{\rm {labels[1]}}}$',
        color='#e67e22',
    )

    ax.set_ylabel(r'${\rm Fraction\ (\%)}$')
    ax.set_xticks(x)
    ax.set_xticklabels(MORPHOLOGY_NAMES_LATEX)
    ax.legend()
    ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi, close=False)

    return fig, ax
