"""
Tessera utility functions for example scripts.

This package provides shared functionality used across multiple example scripts,
consolidating common patterns for snapshot loading, halo catalog handling,
density computation workflows, and halo evolution analysis.

All functions use type hints compatible with Python 3.10+.
"""

from .tessera_helpers import (
    # Snapshot loading
    load_snapshot,
    load_snapshot_h5py,
    infer_snapshot_number,

    # Halo catalog
    find_halo_catalog,
    load_halo_catalog,

    # Lagrangian sorting
    sort_positions_lagrangian,
    infer_grid_size,

    # Density utilities
    extract_2d_slice,
    save_density_hdf5,
    compute_mean_density,
    compute_mean_surface_density,

    # Visualization helpers
    get_axis_labels,

    # Availability checks
    check_tessera_available,
    check_h5py_available,
)

from .halo_evolution import (
    # C++ MergerTree re-exports
    MergerTree,
    MergerTreeHeader,
    HaloTracker,
    HaloInfo,
    TreeTableEntry,
    read_merger_tree_header,

    # Density rendering
    DensityRenderConfig,
    DensityRenderer,

    # Availability check
    check_merger_tree_available,
)

from .smoothing import (
    # Gaussian smoothing
    gaussian_smooth_density,
    adaptive_smooth_density,
    bilateral_smooth_density,
    compute_smoothing_quality_metrics,
)

from .particle_density import (
    # Particle-based density estimation
    compute_particle_density_cic,
    compute_particle_density_sph,
)

from .hybrid_density import (
    # Hybrid density computation
    HybridDensityConfig,
    recommend_method,
    compute_density_auto,
    compare_methods,
)

__all__ = [
    # Snapshot loading
    'load_snapshot',
    'load_snapshot_h5py',
    'infer_snapshot_number',
    # Halo catalog
    'find_halo_catalog',
    'load_halo_catalog',
    # Lagrangian sorting
    'sort_positions_lagrangian',
    'infer_grid_size',
    # Density utilities
    'extract_2d_slice',
    'save_density_hdf5',
    'compute_mean_density',
    'compute_mean_surface_density',
    # Visualization helpers
    'get_axis_labels',
    # Availability checks
    'check_tessera_available',
    'check_h5py_available',
    'check_merger_tree_available',
    # Halo evolution (C++ MergerTree)
    'MergerTree',
    'MergerTreeHeader',
    'HaloTracker',
    'HaloInfo',
    'TreeTableEntry',
    'read_merger_tree_header',
    # Density rendering
    'DensityRenderConfig',
    'DensityRenderer',
    # Artifact mitigation - Smoothing
    'gaussian_smooth_density',
    'adaptive_smooth_density',
    'bilateral_smooth_density',
    'compute_smoothing_quality_metrics',
    # Artifact mitigation - Particle-based density
    'compute_particle_density_cic',
    'compute_particle_density_sph',
    # Artifact mitigation - Hybrid density
    'HybridDensityConfig',
    'recommend_method',
    'compute_density_auto',
    'compare_methods',
]
