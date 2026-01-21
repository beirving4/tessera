"""
Core visualization utilities.

Provides foundation functionality for matplotlib-based visualizations:
- Backend configuration
- Publication-quality style settings
- Colorbar utilities
- Percentile-based color limit computation
- Figure saving
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage


def setup_matplotlib_backend(backend: str = "Agg") -> None:
    """
    Configure matplotlib for non-interactive rendering.

    Parameters
    ----------
    backend : str
        Matplotlib backend to use. Default is "Agg" for non-interactive
        rendering suitable for saving figures to files.

    Notes
    -----
    Must be called before importing pyplot.
    """
    import matplotlib
    matplotlib.use(backend)


def setup_publication_style() -> None:
    """
    Set matplotlib rcParams for publication-quality figures.

    Configures:
    - STIX math font for LaTeX-style equations
    - STIXGeneral font family for consistency
    """
    import matplotlib.pyplot as plt

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'


def add_colorbar(
    ax: Axes,
    im: AxesImage,
    label: str | None = None,
    size: str = "5%",
    pad: float = 0.05,
    ticks: Sequence[float] | None = None,
    ticklabels: Sequence[str] | None = None,
) -> Colorbar:
    """
    Add a colorbar to an axes using make_axes_locatable.

    This creates a colorbar axis that is properly aligned with the main axes
    and doesn't affect the layout of other subplots.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes containing the image.
    im : AxesImage
        The image object returned by imshow() or similar.
    label : str, optional
        Label for the colorbar.
    size : str
        Width of the colorbar as a percentage of the axes width.
    pad : float
        Padding between the axes and colorbar.
    ticks : Sequence[float], optional
        Specific tick locations for the colorbar.
    ticklabels : Sequence[str], optional
        Labels for the ticks (must match length of ticks).

    Returns
    -------
    Colorbar
        The created colorbar object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = plt.colorbar(im, cax=cax)

    if label is not None:
        cbar.set_label(label)

    if ticks is not None:
        cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.ax.set_yticklabels(ticklabels)

    return cbar


def percentile_clim(
    data: NDArray[np.floating],
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.9,
    positive_only: bool = False,
) -> tuple[float, float]:
    """
    Compute percentile-based color limits for visualization.

    Parameters
    ----------
    data : NDArray
        Input data array.
    vmin_pct : float
        Percentile for minimum value (0-100).
    vmax_pct : float
        Percentile for maximum value (0-100).
    positive_only : bool
        If True, only consider positive values when computing percentiles.
        Useful for log-scale visualizations.

    Returns
    -------
    tuple[float, float]
        (vmin, vmax) color limits.
    """
    flat = data.ravel()

    if positive_only:
        flat = flat[flat > 0]

    # Handle edge cases
    if len(flat) == 0:
        return (0.0, 1.0)

    # Filter out non-finite values
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return (0.0, 1.0)

    vmin = float(np.percentile(flat, vmin_pct))
    vmax = float(np.percentile(flat, vmax_pct))

    # Ensure vmin < vmax
    if vmin >= vmax:
        vmax = vmin + 1.0

    return (vmin, vmax)


def save_figure(
    fig: Figure,
    output_path: str | Path,
    dpi: int = 150,
    close: bool = True,
    bbox_inches: str = "tight",
) -> None:
    """
    Save a matplotlib figure with standard settings.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to save.
    output_path : str or Path
        Output file path. Format is inferred from extension.
    dpi : int
        Resolution in dots per inch.
    close : bool
        If True, close the figure after saving to free memory.
    bbox_inches : str
        Bounding box setting. "tight" removes excess whitespace.
    """
    import matplotlib.pyplot as plt

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    if close:
        plt.close(fig)


def prepare_log_data(
    data: NDArray[np.floating],
    floor_factor: float = 0.1,
) -> NDArray[np.floating]:
    """
    Prepare data for log-scale visualization by handling zeros/negatives.

    Parameters
    ----------
    data : NDArray
        Input data array.
    floor_factor : float
        Factor to multiply minimum positive value for floor.

    Returns
    -------
    NDArray
        Data with zeros/negatives replaced by a small positive floor.
    """
    # Find minimum positive value
    positive_mask = data > 0
    if not np.any(positive_mask):
        # No positive values - return ones
        return np.ones_like(data)

    min_positive = np.min(data[positive_mask])
    floor_value = min_positive * floor_factor

    # Replace non-positive values
    result = np.maximum(data, floor_value)
    return result
