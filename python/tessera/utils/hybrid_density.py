"""
Hybrid density computation with automatic method selection.

This module provides a unified API for density field computation that
automatically selects the best method based on scale factor and other
simulation parameters.

Method selection logic:
- a < 5: Pure tessellation (fastest, best quality for early times)
- 5 <= a < 10: Tessellation with geometry filtering + optional smoothing
- a >= 10: SPH-based particle density (avoids tessellation artifacts)

Example usage:
    >>> from tessera.utils.hybrid_density import compute_density_auto, HybridDensityConfig
    >>>
    >>> config = HybridDensityConfig(
    ...     tessellation_max_scale_factor=10.0,
    ...     smoothing_sigma=1.5
    ... )
    >>> result = compute_density_auto(
    ...     positions, particle_ids, box_size=256.0,
    ...     scale_factor=100.0, config=config, projection_axis=2
    ... )
    >>> print(f"Method used: {result['method']}")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

# Import tessera
try:
    import tessera as ts
except ImportError:
    try:
        import _tessera as ts
    except ImportError:
        ts = None

from .smoothing import gaussian_smooth_density, adaptive_smooth_density
from .particle_density import compute_particle_density_sph, compute_particle_density_cic


@dataclass
class HybridDensityConfig:
    """
    Configuration for hybrid density computation.

    Attributes
    ----------
    tessellation_max_scale_factor : float
        Use pure tessellation below this scale factor.
        Above this, switch to smoothed tessellation or SPH.
    smoothing_scale_factor_range : tuple
        Scale factor range for transitional smoothing.
        (start, end) - smoothing strength increases across this range.
    sph_min_scale_factor : float
        Use SPH above this scale factor.

    n_samples : int
        Monte Carlo samples per tetrahedron for tessellation.
    max_aspect_ratio : float
        Skip tetrahedra with aspect ratio above this (0 = disabled).
    elongation_downweight : float
        Weight factor for elongated tetrahedra (1.0 = no downweight).

    smoothing_sigma : float
        Gaussian smoothing sigma in pixels.
    adaptive_smoothing : bool
        If True, use adaptive smoothing based on particle counts.
    smoothing_sigma_min : float
        Minimum sigma for adaptive smoothing.
    smoothing_sigma_max : float
        Maximum sigma for adaptive smoothing.

    sph_kernel : str
        Kernel type for SPH: 'cubic_spline', 'gaussian', 'tophat'.
    sph_n_neighbors : int
        Number of neighbors for adaptive smoothing length.

    output_cells : int
        Output grid resolution per dimension.
    n_threads : int
        Number of threads for tessellation (0 = auto).
    periodic : bool
        Use periodic boundary conditions.
    """

    # Method selection thresholds
    tessellation_max_scale_factor: float = 5.0
    smoothing_scale_factor_range: Tuple[float, float] = (3.0, 10.0)
    sph_min_scale_factor: float = 10.0

    # Tessellation options
    n_samples: int = 100
    max_aspect_ratio: float = 0.0  # Disabled by default
    elongation_downweight: float = 1.0
    aspect_ratio_soft_threshold: float = 10.0

    # Smoothing options
    smoothing_sigma: float = 1.5
    adaptive_smoothing: bool = True
    smoothing_sigma_min: float = 0.5
    smoothing_sigma_max: float = 3.0

    # SPH options
    sph_kernel: str = 'cubic_spline'
    sph_n_neighbors: int = 32

    # General options
    output_cells: int = 256
    n_threads: int = 0
    periodic: bool = True
    particle_mass: float = 1.0


def recommend_method(
    scale_factor: float,
    config: HybridDensityConfig | None = None,
) -> str:
    """
    Recommend density computation method based on scale factor.

    Parameters
    ----------
    scale_factor : float
        Cosmological scale factor (a = 1 at z = 0).
    config : HybridDensityConfig, optional
        Configuration with threshold values.

    Returns
    -------
    method : str
        Recommended method: 'tessellation', 'tessellation_smoothed', or 'sph'.

    Examples
    --------
    >>> recommend_method(1.0)
    'tessellation'
    >>> recommend_method(7.0)
    'tessellation_smoothed'
    >>> recommend_method(100.0)
    'sph'
    """
    if config is None:
        config = HybridDensityConfig()

    if scale_factor < config.tessellation_max_scale_factor:
        return 'tessellation'
    elif scale_factor < config.sph_min_scale_factor:
        return 'tessellation_smoothed'
    else:
        return 'sph'


def compute_density_auto(
    positions: NDArray[np.floating],
    particle_ids: NDArray[np.integer] | None,
    box_size: float,
    scale_factor: float,
    config: HybridDensityConfig | None = None,
    grid_size: int | None = None,
    projection_axis: int | None = None,
    slice_bounds: Tuple[float, float] | None = None,
    subbox_origin: Tuple[float, float, float] | None = None,
    subbox_width: float | None = None,
    force_method: str | None = None,
) -> Dict[str, Any]:
    """
    Compute density field using automatically selected method.

    The method is selected based on scale factor:
    - a < 5: Pure tessellation
    - 5 <= a < 10: Tessellation with smoothing
    - a >= 10: SPH-based particle density

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions.
    particle_ids : ndarray, shape (N,), optional
        Particle IDs (required for tessellation methods).
    box_size : float
        Simulation box size.
    scale_factor : float
        Cosmological scale factor.
    config : HybridDensityConfig, optional
        Configuration options. Uses defaults if None.
    grid_size : int, optional
        Lagrangian grid size (inferred from len(positions) if None).
    projection_axis : int, optional
        Axis for 2D projection (0=x, 1=y, 2=z). None for 3D.
    slice_bounds : tuple, optional
        (min, max) bounds for slice along projection axis.
    subbox_origin : tuple, optional
        Origin for sub-box rendering.
    subbox_width : float, optional
        Width of sub-box.
    force_method : str, optional
        Force a specific method: 'tessellation', 'tessellation_smoothed',
        'tessellation_filtered', 'sph', 'cic'. Overrides auto-selection.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'density': density array (2D or 3D)
        - 'method': method used
        - 'cells': grid resolution
        - 'cell_width': physical cell width
        - 'particle_counts': sample counts (for tessellation methods)
        - 'quality_metrics': dict with method-specific diagnostics
        - 'scale_factor': input scale factor
        - 'recommended_method': what auto-selection would have chosen

    Examples
    --------
    >>> # Automatic method selection
    >>> result = compute_density_auto(
    ...     positions, particle_ids, box_size=256.0,
    ...     scale_factor=1.0, projection_axis=2
    ... )
    >>> print(f"Method: {result['method']}")
    Method: tessellation

    >>> # Force SPH at early times for comparison
    >>> result = compute_density_auto(
    ...     positions, None, box_size=256.0,
    ...     scale_factor=1.0, projection_axis=2,
    ...     force_method='sph'
    ... )
    """
    if config is None:
        config = HybridDensityConfig()

    if grid_size is None:
        n_particles = len(positions)
        grid_size = int(round(n_particles ** (1/3)))

    # Determine method
    recommended = recommend_method(scale_factor, config)
    method = force_method if force_method else recommended

    result = {
        'scale_factor': scale_factor,
        'recommended_method': recommended,
        'box_size': box_size,
    }

    # =========================================================================
    # Tessellation-based methods
    # =========================================================================
    if method in ['tessellation', 'tessellation_smoothed', 'tessellation_filtered']:
        if ts is None:
            raise ImportError("tessera is required for tessellation methods")

        if particle_ids is None:
            raise ValueError("particle_ids required for tessellation methods")

        # Sort to Lagrangian order
        sorted_positions = ts.density.sort_by_lagrangian_id(
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(particle_ids, dtype=np.int64),
            grid_size,
            1  # GADGET uses 1-indexed IDs
        )

        # Configure tessera
        ts_config = ts.density.TetraDensityConfig()
        ts_config.lagrangian_grid_size = grid_size
        ts_config.box_size = box_size
        ts_config.output_cells = config.output_cells
        ts_config.n_samples = config.n_samples
        ts_config.n_threads = config.n_threads
        ts_config.periodic = config.periodic
        ts_config.particle_mass = config.particle_mass

        # Geometry filtering (if supported and enabled)
        if method == 'tessellation_filtered' or config.max_aspect_ratio > 0:
            if hasattr(ts_config, 'max_aspect_ratio'):
                ts_config.max_aspect_ratio = config.max_aspect_ratio
                ts_config.elongation_downweight = config.elongation_downweight
                if hasattr(ts_config, 'aspect_ratio_soft_threshold'):
                    ts_config.aspect_ratio_soft_threshold = config.aspect_ratio_soft_threshold

        # Sub-box mode
        if subbox_origin is not None and subbox_width is not None:
            ts_config.subbox_enabled = True
            ts_config.subbox_origin = subbox_origin
            ts_config.subbox_width = (subbox_width, subbox_width, subbox_width)

        # Compute density
        if projection_axis is None:
            ts_result = ts.density.compute_tetra_density_3d(sorted_positions, ts_config)
            density = np.array(ts_result.density).reshape(
                config.output_cells, config.output_cells, config.output_cells
            )
        else:
            ts_result = ts.density.compute_tetra_density_2d_projection(
                sorted_positions, ts_config, projection_axis
            )
            density = np.array(ts_result.density).reshape(
                config.output_cells, config.output_cells
            )

        # Get particle counts
        particle_counts = np.array(ts_result.particle_counts).reshape(density.shape)

        # Apply smoothing if requested
        if method == 'tessellation_smoothed':
            if config.adaptive_smoothing:
                density = adaptive_smooth_density(
                    density, particle_counts,
                    sigma_min=config.smoothing_sigma_min,
                    sigma_max=config.smoothing_sigma_max
                )
            else:
                density = gaussian_smooth_density(
                    density, sigma=config.smoothing_sigma,
                    particle_counts=particle_counts
                )

        result.update({
            'density': density,
            'method': method,
            'cells': config.output_cells,
            'cell_width': ts_result.cell_width,
            'particle_counts': particle_counts,
            'n_tetrahedra': ts_result.n_tetrahedra,
            'quality_metrics': {
                'zero_count_pixels': int((particle_counts == 0).sum()),
                'low_count_pixels': int((particle_counts < 2).sum()),
                'mean_counts': float(particle_counts.mean()),
            }
        })

        # Add geometry filtering metrics if available
        if hasattr(ts_result, 'skipped_tetrahedra'):
            result['quality_metrics']['skipped_tetrahedra'] = ts_result.skipped_tetrahedra
        if hasattr(ts_result, 'mean_aspect_ratio'):
            result['quality_metrics']['mean_aspect_ratio'] = ts_result.mean_aspect_ratio

    # =========================================================================
    # SPH-based method
    # =========================================================================
    elif method == 'sph':
        sph_result = compute_particle_density_sph(
            positions, box_size, config.output_cells,
            particle_mass=config.particle_mass,
            kernel=config.sph_kernel,
            n_neighbors=config.sph_n_neighbors,
            projection_axis=projection_axis,
            slice_bounds=slice_bounds,
            periodic=config.periodic
        )

        result.update({
            'density': sph_result['density'],
            'method': 'sph',
            'cells': sph_result['cells'],
            'cell_width': sph_result['cell_width'],
            'particle_counts': None,  # SPH doesn't have sample counts
            'quality_metrics': {
                'kernel': sph_result['kernel'],
                'n_neighbors': sph_result['n_neighbors'],
                'mean_smoothing_length': sph_result['mean_smoothing_length'],
                'n_particles_used': sph_result['n_particles_used'],
            }
        })

    # =========================================================================
    # CIC method
    # =========================================================================
    elif method == 'cic':
        cic_result = compute_particle_density_cic(
            positions, box_size, config.output_cells,
            particle_mass=config.particle_mass,
            projection_axis=projection_axis,
            slice_bounds=slice_bounds,
            periodic=config.periodic
        )

        result.update({
            'density': cic_result['density'],
            'method': 'cic',
            'cells': cic_result['cells'],
            'cell_width': cic_result['cell_width'],
            'particle_counts': None,
            'quality_metrics': {
                'n_particles_used': cic_result['n_particles_used'],
            }
        })

    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def compare_methods(
    positions: NDArray[np.floating],
    particle_ids: NDArray[np.integer] | None,
    box_size: float,
    scale_factor: float,
    methods: list[str] | None = None,
    config: HybridDensityConfig | None = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple density computation methods on the same data.

    Parameters
    ----------
    positions : ndarray
        Particle positions.
    particle_ids : ndarray, optional
        Particle IDs.
    box_size : float
        Simulation box size.
    scale_factor : float
        Scale factor.
    methods : list of str, optional
        Methods to compare. Default: ['tessellation', 'tessellation_smoothed', 'sph'].
    config : HybridDensityConfig, optional
        Configuration.
    **kwargs
        Additional arguments passed to compute_density_auto.

    Returns
    -------
    results : dict
        Dictionary mapping method name to result dict.

    Examples
    --------
    >>> results = compare_methods(
    ...     positions, particle_ids, box_size=256.0, scale_factor=100.0,
    ...     projection_axis=2
    ... )
    >>> for method, result in results.items():
    ...     print(f"{method}: mean={result['density'].mean():.2e}")
    """
    if methods is None:
        methods = ['tessellation', 'tessellation_smoothed', 'sph']

    results = {}
    for method in methods:
        try:
            results[method] = compute_density_auto(
                positions, particle_ids, box_size, scale_factor,
                config=config, force_method=method, **kwargs
            )
        except Exception as e:
            results[method] = {'error': str(e)}

    return results
