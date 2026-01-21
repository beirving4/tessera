"""
Tessera visualization utilities.

This package provides reusable visualization functions for density fields,
ORIGAMI morphology classifications, validation comparisons, and PDF analysis.

All functions use type hints compatible with Python 3.10+.
"""

from .core import (
    setup_matplotlib_backend,
    setup_publication_style,
    add_colorbar,
    percentile_clim,
    save_figure,
)

from .density import (
    plot_density_slice,
    plot_density_comparison,
)

from .morphology import (
    MORPHOLOGY_COLORS,
    MORPHOLOGY_NAMES,
    MORPHOLOGY_NAMES_LATEX,
    create_morphology_cmap,
    plot_morphology_slice,
    plot_morphology_slices_3panel,
    plot_morphology_fractions_pie,
    plot_morphology_comparison_bar,
)

from .comparison import (
    create_density_validation_figure,
    create_origami_validation_figure,
)

from .pdf import (
    plot_pdf_histogram,
    plot_pdf_comparison,
)

from .evolution import (
    EvolutionVisualizationConfig,
    VisualizationConfig,
    auto_select_epochs,
    create_evolution_multipanel,
    create_multipanel_figure,
    create_evolution_animation,
)

__all__ = [
    # Core utilities
    'setup_matplotlib_backend',
    'setup_publication_style',
    'add_colorbar',
    'percentile_clim',
    'save_figure',
    # Density visualization
    'plot_density_slice',
    'plot_density_comparison',
    # Morphology visualization
    'MORPHOLOGY_COLORS',
    'MORPHOLOGY_NAMES',
    'MORPHOLOGY_NAMES_LATEX',
    'create_morphology_cmap',
    'plot_morphology_slice',
    'plot_morphology_slices_3panel',
    'plot_morphology_fractions_pie',
    'plot_morphology_comparison_bar',
    # Validation comparison figures
    'create_density_validation_figure',
    'create_origami_validation_figure',
    # PDF/histogram plotting
    'plot_pdf_histogram',
    'plot_pdf_comparison',
    # Evolution visualization
    'EvolutionVisualizationConfig',
    'VisualizationConfig',
    'auto_select_epochs',
    'create_evolution_multipanel',
    'create_multipanel_figure',
    'create_evolution_animation',
]
