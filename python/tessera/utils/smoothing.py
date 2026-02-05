"""
Smoothing utilities for artifact mitigation in density fields.

This module provides Gaussian and adaptive smoothing functions to reduce
tessellation artifacts (radial streaks) that appear at late cosmological
times when tetrahedra become highly elongated.

Example usage:
    >>> from tessera.utils.smoothing import gaussian_smooth_density, adaptive_smooth_density
    >>>
    >>> # Simple Gaussian smoothing
    >>> smoothed = gaussian_smooth_density(density, sigma=1.5)
    >>>
    >>> # Adaptive smoothing using particle counts
    >>> smoothed = adaptive_smooth_density(density, particle_counts,
    ...                                     sigma_min=0.5, sigma_max=3.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def gaussian_smooth_density(
    density: NDArray[np.floating],
    sigma: float = 1.0,
    particle_counts: NDArray[np.integer] | None = None,
    count_threshold: int = 2,
    preserve_mass: bool = True,
    mode: str = 'reflect',
) -> NDArray[np.floating]:
    """
    Apply Gaussian smoothing to a density field.

    Parameters
    ----------
    density : ndarray, shape (N, N) or (N, N, N)
        Input density field (2D or 3D).
    sigma : float, default=1.0
        Standard deviation of Gaussian kernel in pixel units.
        Larger values = more smoothing.
    particle_counts : ndarray, optional
        Sample counts per cell from tessellation. If provided, cells with
        counts below count_threshold are replaced with local average before
        smoothing to reduce artifact propagation.
    count_threshold : int, default=2
        Cells with counts below this are considered unreliable.
    preserve_mass : bool, default=True
        If True, rescale output to preserve total mass.
    mode : str, default='reflect'
        Boundary handling mode for scipy.ndimage.gaussian_filter.
        Options: 'reflect', 'constant', 'nearest', 'wrap'.

    Returns
    -------
    smoothed : ndarray
        Smoothed density field with same shape as input.

    Notes
    -----
    This is a simple post-processing approach that works well for mild
    artifacts. For severe artifacts at a > 10, consider using adaptive
    smoothing or the hybrid density approach.

    Examples
    --------
    >>> result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)
    >>> density = np.array(result.density).reshape(256, 256)
    >>> smoothed = gaussian_smooth_density(density, sigma=1.5)
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for smoothing functions. "
            "Install with: pip install scipy"
        )

    density = np.asarray(density, dtype=np.float64)
    original_sum = density.sum() if preserve_mass else None

    # Pre-process unreliable cells if particle_counts provided
    if particle_counts is not None:
        particle_counts = np.asarray(particle_counts)
        if particle_counts.shape != density.shape:
            raise ValueError(
                f"particle_counts shape {particle_counts.shape} must match "
                f"density shape {density.shape}"
            )

        # Replace low-count cells with local average
        unreliable = particle_counts < count_threshold
        if np.any(unreliable):
            # Use uniform filter to get local average
            local_avg = uniform_filter(density, size=3, mode=mode)
            density = density.copy()
            density[unreliable] = local_avg[unreliable]

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(density, sigma=sigma, mode=mode)

    # Preserve total mass if requested
    if preserve_mass and original_sum is not None and original_sum > 0:
        smoothed *= original_sum / smoothed.sum()

    return smoothed


def adaptive_smooth_density(
    density: NDArray[np.floating],
    particle_counts: NDArray[np.integer],
    sigma_min: float = 0.5,
    sigma_max: float = 3.0,
    count_threshold: int = 2,
    count_saturation: int = 100,
    preserve_mass: bool = True,
    mode: str = 'reflect',
) -> NDArray[np.floating]:
    """
    Apply adaptive smoothing where kernel size varies with local sample count.

    Low-count regions (sparse, unreliable) receive stronger smoothing,
    while high-count regions (well-sampled) receive minimal smoothing.
    This preserves structure in dense regions while removing artifacts
    in sparse regions.

    Parameters
    ----------
    density : ndarray, shape (N, N) or (N, N, N)
        Input density field.
    particle_counts : ndarray
        Sample counts per cell from tessellation.
    sigma_min : float, default=0.5
        Minimum smoothing sigma (applied to well-sampled regions).
    sigma_max : float, default=3.0
        Maximum smoothing sigma (applied to poorly-sampled regions).
    count_threshold : int, default=2
        Counts below this get maximum smoothing.
    count_saturation : int, default=100
        Counts above this get minimum smoothing.
    preserve_mass : bool, default=True
        If True, rescale output to preserve total mass.
    mode : str, default='reflect'
        Boundary handling mode.

    Returns
    -------
    smoothed : ndarray
        Adaptively smoothed density field.

    Notes
    -----
    The algorithm works by:
    1. Computing a per-cell sigma based on particle_counts
    2. Applying multiple Gaussian filters at different scales
    3. Blending results based on local sigma values

    This is more computationally expensive than simple Gaussian smoothing
    but produces better results for fields with varying sample quality.

    Examples
    --------
    >>> result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)
    >>> density = np.array(result.density).reshape(256, 256)
    >>> counts = np.array(result.particle_counts).reshape(256, 256)
    >>> smoothed = adaptive_smooth_density(density, counts, sigma_min=0.5, sigma_max=3.0)
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for smoothing functions. "
            "Install with: pip install scipy"
        )

    density = np.asarray(density, dtype=np.float64)
    particle_counts = np.asarray(particle_counts)

    if particle_counts.shape != density.shape:
        raise ValueError(
            f"particle_counts shape {particle_counts.shape} must match "
            f"density shape {density.shape}"
        )

    original_sum = density.sum() if preserve_mass else None

    # Compute per-cell sigma based on counts
    # Low counts -> high sigma, high counts -> low sigma
    counts_clipped = np.clip(particle_counts, count_threshold, count_saturation)

    # Linear interpolation in log space for counts
    log_counts = np.log(counts_clipped + 1)
    log_min = np.log(count_threshold + 1)
    log_max = np.log(count_saturation + 1)

    # Normalized position: 0 = low counts (max sigma), 1 = high counts (min sigma)
    t = (log_counts - log_min) / (log_max - log_min)
    t = np.clip(t, 0, 1)

    # Per-cell sigma
    sigma_map = sigma_max + t * (sigma_min - sigma_max)

    # Multi-scale blending approach
    # Generate smoothed versions at multiple scales
    n_scales = 5
    sigmas = np.linspace(sigma_min, sigma_max, n_scales)
    smoothed_stack = np.stack([
        gaussian_filter(density, sigma=s, mode=mode) for s in sigmas
    ])

    # Interpolate between scales based on sigma_map
    # Find which two scales each pixel falls between
    sigma_indices = (sigma_map - sigma_min) / (sigma_max - sigma_min) * (n_scales - 1)
    sigma_indices = np.clip(sigma_indices, 0, n_scales - 1)

    lower_idx = np.floor(sigma_indices).astype(int)
    upper_idx = np.minimum(lower_idx + 1, n_scales - 1)
    blend_weight = sigma_indices - lower_idx

    # Blend between adjacent scales
    if density.ndim == 2:
        smoothed = np.zeros_like(density)
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                li, ui = lower_idx[i, j], upper_idx[i, j]
                w = blend_weight[i, j]
                smoothed[i, j] = (1 - w) * smoothed_stack[li, i, j] + w * smoothed_stack[ui, i, j]
    else:
        # 3D case - vectorized approach
        shape = density.shape
        flat_lower = lower_idx.ravel()
        flat_upper = upper_idx.ravel()
        flat_weight = blend_weight.ravel()

        smoothed_flat = np.zeros(density.size)
        for idx in range(density.size):
            li, ui = flat_lower[idx], flat_upper[idx]
            w = flat_weight[idx]
            i, j, k = np.unravel_index(idx, shape)
            smoothed_flat[idx] = (1 - w) * smoothed_stack[li, i, j, k] + w * smoothed_stack[ui, i, j, k]
        smoothed = smoothed_flat.reshape(shape)

    # Preserve total mass if requested
    if preserve_mass and original_sum is not None and original_sum > 0:
        smoothed *= original_sum / smoothed.sum()

    return smoothed


def bilateral_smooth_density(
    density: NDArray[np.floating],
    sigma_spatial: float = 1.5,
    sigma_intensity: float | None = None,
    preserve_mass: bool = True,
) -> NDArray[np.floating]:
    """
    Apply bilateral filtering to density field.

    Bilateral filtering smooths while preserving edges, which can help
    reduce artifacts while maintaining sharp density contrasts at
    halo boundaries.

    Parameters
    ----------
    density : ndarray, shape (N, N) or (N, N, N)
        Input density field.
    sigma_spatial : float, default=1.5
        Spatial smoothing scale in pixels.
    sigma_intensity : float, optional
        Intensity smoothing scale. If None, automatically determined
        from density statistics.
    preserve_mass : bool, default=True
        If True, rescale output to preserve total mass.

    Returns
    -------
    smoothed : ndarray
        Bilaterally smoothed density field.

    Notes
    -----
    This uses a simplified bilateral filter implementation. For best
    results on large grids, consider using OpenCV's bilateral filter
    if available.
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for smoothing functions. "
            "Install with: pip install scipy"
        )

    density = np.asarray(density, dtype=np.float64)
    original_sum = density.sum() if preserve_mass else None

    if sigma_intensity is None:
        # Auto-determine from density range
        sigma_intensity = np.std(density[density > 0]) * 0.5

    # Simple bilateral filter implementation
    # For each pixel, compute weighted average using spatial and intensity weights

    # Create spatial kernel
    kernel_size = int(np.ceil(sigma_spatial * 3)) * 2 + 1
    half_k = kernel_size // 2

    y, x = np.ogrid[-half_k:half_k+1, -half_k:half_k+1]
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    if density.ndim == 3:
        z = np.ogrid[-half_k:half_k+1]
        spatial_kernel = np.exp(-(x**2 + y**2 + z[0]**2) / (2 * sigma_spatial**2))

    # Pad density for boundary handling
    padded = np.pad(density, half_k, mode='reflect')
    smoothed = np.zeros_like(density)

    # Apply bilateral filter (simplified, works for 2D)
    if density.ndim == 2:
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                # Extract neighborhood
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                center_val = density[i, j]

                # Intensity weights
                intensity_weights = np.exp(
                    -(neighborhood - center_val)**2 / (2 * sigma_intensity**2)
                )

                # Combined weights
                weights = spatial_kernel * intensity_weights
                weights /= weights.sum()

                smoothed[i, j] = np.sum(neighborhood * weights)
    else:
        # For 3D, fall back to simple Gaussian (bilateral is expensive)
        smoothed = gaussian_filter(density, sigma=sigma_spatial, mode='reflect')

    # Preserve total mass if requested
    if preserve_mass and original_sum is not None and original_sum > 0:
        smoothed *= original_sum / smoothed.sum()

    return smoothed


def compute_smoothing_quality_metrics(
    original: NDArray[np.floating],
    smoothed: NDArray[np.floating],
    particle_counts: NDArray[np.integer] | None = None,
) -> dict:
    """
    Compute quality metrics comparing original and smoothed density fields.

    Parameters
    ----------
    original : ndarray
        Original density field.
    smoothed : ndarray
        Smoothed density field.
    particle_counts : ndarray, optional
        Sample counts for weighting metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'correlation': Pearson correlation coefficient
        - 'mass_ratio': smoothed.sum() / original.sum()
        - 'rmse': Root mean square error
        - 'relative_change_mean': Mean |smoothed - original| / original
        - 'high_density_correlation': Correlation in high-density regions
    """
    original = np.asarray(original).ravel()
    smoothed = np.asarray(smoothed).ravel()

    metrics = {}

    # Correlation
    metrics['correlation'] = float(np.corrcoef(original, smoothed)[0, 1])

    # Mass conservation
    metrics['mass_ratio'] = float(smoothed.sum() / original.sum()) if original.sum() > 0 else 0.0

    # RMSE
    metrics['rmse'] = float(np.sqrt(np.mean((smoothed - original)**2)))

    # Relative change (avoid division by zero)
    mask = original > 0
    if np.any(mask):
        rel_change = np.abs(smoothed[mask] - original[mask]) / original[mask]
        metrics['relative_change_mean'] = float(np.mean(rel_change))
    else:
        metrics['relative_change_mean'] = 0.0

    # High-density region correlation (top 10%)
    threshold = np.percentile(original, 90)
    high_mask = original > threshold
    if np.sum(high_mask) > 10:
        metrics['high_density_correlation'] = float(
            np.corrcoef(original[high_mask], smoothed[high_mask])[0, 1]
        )
    else:
        metrics['high_density_correlation'] = metrics['correlation']

    return metrics
