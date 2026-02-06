"""
Halo Evolution Visualization Pipeline

This package provides tools for visualizing the evolution of individual
dark matter halos across cosmic time using tessera's density field
capabilities and GADGET-4 merger trees.

Consolidated Modules
--------------------
The functionality from this package has been consolidated into the
main tessera utility modules:

- **utils.MergerTree**: C++ merger tree reader (high performance)
- **utils.HaloTracker**: C++ branch tracing with prefetching
- **utils.HaloInfo**: Halo information struct
- **utils.DensityRenderer**: Tessera density computation wrapper
- **visualize.create_evolution_multipanel**: Multi-panel figure generation
- **visualize.create_evolution_animation**: Animation generation

For new code, import from utils and visualize directly:

    from utils import MergerTree, HaloTracker, DensityRenderer
    from visualize import create_evolution_multipanel, auto_select_epochs

Legacy Imports
--------------
For backwards compatibility, this package re-exports from the consolidated
modules when available, falling back to local implementations if needed.

See Also
--------
halo_evolution_pipeline.py : Main consolidated pipeline script
"""

import sys
from pathlib import Path

# Try to import from consolidated modules first (C++ implementations)
try:
    from tessera.utils import (
        MergerTree,
        HaloTracker,
        HaloInfo,
        DensityRenderer,
        DensityRenderConfig,
    )
    from tessera.visualize import (
        create_evolution_animation,
        create_evolution_multipanel as create_multipanel_figure,
        VisualizationConfig,
        auto_select_epochs,
    )
    _USING_CONSOLIDATED = True
except ImportError:
    # Fall back to local implementations
    _USING_CONSOLIDATED = False
    from .merger_tree import MergerTree
    from .halo_tracker import HaloTracker, HaloInfo
    try:
        from .density_renderer import DensityRenderer, DensityRenderConfig
    except ImportError:
        DensityRenderer = None
        DensityRenderConfig = None
    try:
        from .visualize_evolution import (
            create_evolution_animation,
            create_multipanel_figure,
            VisualizationConfig,
            auto_select_epochs,
        )
    except ImportError:
        create_evolution_animation = None
        create_multipanel_figure = None
        VisualizationConfig = None
        auto_select_epochs = None

# Branch catalog for efficient array job processing
from .branch_catalog import BranchCatalog, Branch, unwrap_branch_coordinates

# Position-based tracking (alternative to merger trees)
from .position_tracker import (
    PositionMatchingTracker,
    HaloCatalogLoader,
    TrackedHalo,
    validate_branch_catalog,
)

__all__ = [
    'MergerTree',
    'HaloTracker',
    'HaloInfo',
    'BranchCatalog',
    'Branch',
    'unwrap_branch_coordinates',
    # Position tracking
    'PositionMatchingTracker',
    'HaloCatalogLoader',
    'TrackedHalo',
    'validate_branch_catalog',
]

if DensityRenderer is not None:
    __all__.extend(['DensityRenderer', 'DensityRenderConfig'])

if create_evolution_animation is not None:
    __all__.extend([
        'create_evolution_animation',
        'create_multipanel_figure',
        'VisualizationConfig',
        'auto_select_epochs',
    ])


def using_consolidated_modules() -> bool:
    """Check if using consolidated C++ modules."""
    return _USING_CONSOLIDATED
