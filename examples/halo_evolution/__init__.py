"""
Halo Evolution Visualization Pipeline

This package provides tools for visualizing the evolution of individual
dark matter halos across cosmic time using tessera's density field
capabilities and GADGET-4 merger trees.

Modules
-------
merger_tree : GADGET-4 merger tree reader
halo_tracker : Branch tracing and snapshot mapping
density_renderer : tessera density computation wrapper
visualize_evolution : Animation and figure generation
"""

from .merger_tree import MergerTree
from .halo_tracker import HaloTracker, HaloInfo

__all__ = [
    'MergerTree',
    'HaloTracker',
    'HaloInfo',
]

# Optional imports (modules may not exist yet)
try:
    from .density_renderer import DensityRenderer, DensityRenderConfig
    __all__.extend(['DensityRenderer', 'DensityRenderConfig'])
except ImportError:
    pass

try:
    from .visualize_evolution import (
        create_evolution_animation,
        create_multipanel_figure,
        VisualizationConfig,
    )
    __all__.extend([
        'create_evolution_animation',
        'create_multipanel_figure',
        'VisualizationConfig',
    ])
except ImportError:
    pass
