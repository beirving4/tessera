"""
Validation comparison figure utilities.

Provides functions for creating multi-panel comparison figures
used in validation tests against reference implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from .morphology import (
    MORPHOLOGY_COLORS,
    MORPHOLOGY_NAMES_LATEX,
    plot_morphology_comparison_bar,
    plot_morphology_fractions_pie,
)


def create_density_validation_figure(
    tessera_density: NDArray[np.floating],
    reference_density: NDArray[np.floating],
    output_path: str | Path,
    title: str = "Density Validation",
    correlation: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (15, 10),
    dpi: int = 150,
) -> None:
    """
    Create 2x3 validation figure comparing density fields.

    Top row: tessera density, reference density, relative difference
    Bottom row: overdensity PDF, scatter plot, difference histogram

    Parameters
    ----------
    tessera_density : NDArray
        2D density field from tessera.
    reference_density : NDArray
        2D density field from reference implementation.
    output_path : str or Path
        Path to save the figure.
    title : str
        Figure title.
    correlation : float, optional
        Pre-computed correlation coefficient. If None, computed internally.
    extent : tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax).
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    setup_publication_style()

    # Normalize both fields
    tess_norm = tessera_density / np.mean(tessera_density)
    ref_norm = reference_density / np.mean(reference_density)

    # Compute relative difference
    mask = ref_norm > 0
    rel_diff = np.zeros_like(tess_norm)
    rel_diff[mask] = (tess_norm[mask] - ref_norm[mask]) / ref_norm[mask]

    # Compute correlation if not provided
    if correlation is None:
        correlation = float(np.corrcoef(tess_norm.ravel(), ref_norm.ravel())[0, 1])

    # Prepare log data
    tess_log = prepare_log_data(tess_norm)
    ref_log = prepare_log_data(ref_norm)

    # Compute shared color limits
    vmin = np.log10(min(
        percentile_clim(tess_log, 1, 99.9, positive_only=True)[0],
        percentile_clim(ref_log, 1, 99.9, positive_only=True)[0],
    ))
    vmax = np.log10(max(
        percentile_clim(tess_log, 1, 99.9, positive_only=True)[1],
        percentile_clim(ref_log, 1, 99.9, positive_only=True)[1],
    ))

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Top row: Density comparison
    im1 = axes[0, 0].imshow(
        np.log10(tess_log.T),
        origin='lower',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    axes[0, 0].set_title(r'${\rm tessera}$')
    add_colorbar(axes[0, 0], im1, label=r'$\log_{10}(1+\delta)$')

    im2 = axes[0, 1].imshow(
        np.log10(ref_log.T),
        origin='lower',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    axes[0, 1].set_title(r'${\rm Reference}$')
    add_colorbar(axes[0, 1], im2, label=r'$\log_{10}(1+\delta)$')

    # Relative difference
    finite_diff = rel_diff[np.isfinite(rel_diff)]
    diff_abs_max = np.percentile(np.abs(finite_diff), 99) if len(finite_diff) > 0 else 1.0

    im3 = axes[0, 2].imshow(
        rel_diff.T,
        origin='lower',
        cmap='RdBu_r',
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
        extent=extent,
    )
    axes[0, 2].set_title(r'${\rm Relative\ Difference}$')
    add_colorbar(axes[0, 2], im3, label=r'${\rm (tessera - Ref) / Ref}$')

    # Bottom row: Statistical comparisons
    tess_flat = tess_norm.ravel()
    ref_flat = ref_norm.ravel()

    # Overdensity PDF
    bins = np.logspace(-2, 4, 100)
    axes[1, 0].hist(
        tess_flat[tess_flat > 0], bins=bins, alpha=0.7,
        label=r'${\rm tessera}$', color='#e74c3c', density=True
    )
    axes[1, 0].hist(
        ref_flat[ref_flat > 0], bins=bins, alpha=0.7,
        label=r'${\rm Reference}$', color='#3498db', density=True
    )
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel(r'$1+\delta$')
    axes[1, 0].set_ylabel(r'${\rm PDF}$')
    axes[1, 0].set_title(r'${\rm Overdensity\ PDF}$')
    axes[1, 0].legend()

    # Scatter plot
    n_sample = min(10000, len(tess_flat))
    idx = np.random.choice(len(tess_flat), n_sample, replace=False)
    axes[1, 1].scatter(ref_flat[idx], tess_flat[idx], s=1, alpha=0.5, c='#2c3e50')

    lims = [
        min(ref_flat[idx].min(), tess_flat[idx].min()),
        max(ref_flat[idx].max(), tess_flat[idx].max()),
    ]
    axes[1, 1].plot(lims, lims, 'r-', linewidth=2, label=r'$1:1$')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel(r'${\rm Reference}\ (1+\delta)$')
    axes[1, 1].set_ylabel(r'${\rm tessera}\ (1+\delta)$')
    axes[1, 1].set_title(rf'$r={correlation:.6f}$')
    axes[1, 1].legend()

    # Difference histogram
    rel_diff_flat = rel_diff.ravel()
    valid_diff = rel_diff_flat[np.isfinite(rel_diff_flat) & (ref_flat > 0)]
    hist_min = np.percentile(valid_diff, 1) if len(valid_diff) > 0 else -0.5
    hist_max = np.percentile(valid_diff, 99) if len(valid_diff) > 0 else 0.5
    hist_range = (min(hist_min, -0.5), max(hist_max, 0.5))

    axes[1, 2].hist(
        valid_diff, bins=100, range=hist_range,
        alpha=0.7, color='#27ae60', density=True
    )
    axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    if len(valid_diff) > 0:
        mean_diff = np.mean(valid_diff)
        std_diff = np.std(valid_diff)
        axes[1, 2].axvline(
            x=mean_diff, color='blue', linestyle='-', linewidth=2,
            label=f'Mean: {mean_diff:.4f}'
        )
        axes[1, 2].set_title(f'Diff Distribution (σ={std_diff:.4f})')
    axes[1, 2].set_xlabel(r'${\rm Relative\ Difference}$')
    axes[1, 2].set_ylabel(r'${\rm PDF}$')
    axes[1, 2].legend()

    # Overall title
    passed = correlation > 0.999
    status = "PASS" if passed else "FAIL"
    plt.suptitle(f'{title}\nCorrelation: {correlation:.6f} | {status}', fontsize=14)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=dpi)


def create_origami_validation_figure(
    stats: dict[str, Any],
    output_path: str | Path,
    title: str = "ORIGAMI Validation",
    figsize: tuple[float, float] = (15, 5),
    dpi: int = 150,
) -> None:
    """
    Create 1x3 ORIGAMI validation figure.

    Panels: mass fractions pie, volume fractions pie, classification bar chart.

    Parameters
    ----------
    stats : dict[str, Any]
        Validation statistics dictionary containing:
        - 'mass_fractions': dict with 'void', 'wall', 'filament', 'halo'
        - 'volume_fractions': dict with 'void', 'wall', 'filament', 'halo'
        - 'per_class': dict with per-class statistics
        - 'match_rate': float
    output_path : str or Path
        Path to save the figure.
    title : str
        Figure title prefix.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Resolution for saved figure.
    """
    setup_matplotlib_backend()
    import matplotlib.pyplot as plt

    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Mass fractions pie chart
    mass_fracs = stats['mass_fractions']
    values = [
        mass_fracs.get('void', 0),
        mass_fracs.get('wall', 0),
        mass_fracs.get('filament', 0),
        mass_fracs.get('halo', 0),
    ]
    axes[0].pie(
        values,
        labels=MORPHOLOGY_NAMES_LATEX,
        colors=MORPHOLOGY_COLORS,
        autopct='%.1f%%',
        startangle=90,
    )
    axes[0].set_title(r'${\rm Mass\ Fractions}$')

    # Volume fractions pie chart
    vol_fracs = stats['volume_fractions']
    values = [
        vol_fracs.get('void', 0),
        vol_fracs.get('wall', 0),
        vol_fracs.get('filament', 0),
        vol_fracs.get('halo', 0),
    ]
    axes[1].pie(
        values,
        labels=MORPHOLOGY_NAMES_LATEX,
        colors=MORPHOLOGY_COLORS,
        autopct='%.1f%%',
        startangle=90,
    )
    axes[1].set_title(r'${\rm Volume\ Fractions}$')

    # Classification comparison bar chart
    x = np.arange(4)
    width = 0.35
    class_names = ['void', 'wall', 'filament', 'halo']

    orig_counts = [
        stats['per_class'][n]['original_fraction'] * 100
        for n in class_names
    ]
    tessera_counts = [
        stats['per_class'][n]['tessera_fraction'] * 100
        for n in class_names
    ]

    axes[2].bar(
        x - width/2, orig_counts, width,
        label=r'${\rm Original\ ORIGAMI}$', color='#2c3e50'
    )
    axes[2].bar(
        x + width/2, tessera_counts, width,
        label=r'${\rm tessera}$', color='#e67e22'
    )
    axes[2].set_ylabel(r'${\rm Fraction\ (\%)}$')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(MORPHOLOGY_NAMES_LATEX)
    axes[2].legend()
    axes[2].set_title(r'${\rm Classification\ Comparison}$')

    # Overall title with match rate
    match_rate = stats.get('match_rate', 0) * 100
    plt.suptitle(
        rf'{title}' + '\n' + rf'$\rm Match\ Rate:\ {match_rate:.2f}\%$'
    )

    plt.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
