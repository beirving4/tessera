"""
Particle-based density estimation for artifact-free visualization.

This module provides SPH-style and Cloud-in-Cell (CIC) density estimation
that bypasses tessellation entirely. Useful for extreme scale factors (a > 10)
where tetrahedra become highly elongated and tessellation produces artifacts.

Example usage:
    >>> from tessera.utils.particle_density import compute_particle_density_sph
    >>>
    >>> # SPH-style density
    >>> result = compute_particle_density_sph(
    ...     positions, box_size=256.0, output_cells=256,
    ...     n_neighbors=32, projection_axis=2
    ... )
    >>> density = result['density']
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_particle_density_cic(
    positions: NDArray[np.floating],
    box_size: float,
    output_cells: int,
    particle_mass: float = 1.0,
    projection_axis: int | None = None,
    slice_bounds: Tuple[float, float] | None = None,
    periodic: bool = True,
) -> dict:
    """
    Compute density using Cloud-in-Cell (CIC) assignment.

    This is a simple and fast method that distributes each particle's mass
    to the 8 (or 4 in 2D) nearest grid cells with trilinear (bilinear) weights.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions.
    box_size : float
        Simulation box size.
    output_cells : int
        Output grid resolution per dimension.
    particle_mass : float, default=1.0
        Mass per particle.
    projection_axis : int, optional
        If provided, compute 2D projection along this axis (0=x, 1=y, 2=z).
        If None, compute full 3D density field.
    slice_bounds : tuple, optional
        If provided with projection_axis, restrict to particles within
        (slice_min, slice_max) along the projection axis.
    periodic : bool, default=True
        Use periodic boundary conditions.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'density': 2D or 3D density array
        - 'cells': grid size
        - 'cell_width': physical cell width
        - 'method': 'cic'
        - 'n_particles_used': number of particles deposited

    Examples
    --------
    >>> result = compute_particle_density_cic(positions, 256.0, 256, projection_axis=2)
    >>> density = result['density']  # shape (256, 256)
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_particles = len(positions)

    cell_width = box_size / output_cells
    inv_cell_width = 1.0 / cell_width

    # Filter particles if slice specified
    if projection_axis is not None and slice_bounds is not None:
        slice_min, slice_max = slice_bounds
        mask = (positions[:, projection_axis] >= slice_min) & \
               (positions[:, projection_axis] < slice_max)
        positions = positions[mask]

    n_used = len(positions)

    # Compute grid indices and weights
    # Position in cell units
    pos_cell = positions * inv_cell_width

    if projection_axis is None:
        # 3D CIC
        density = np.zeros((output_cells, output_cells, output_cells), dtype=np.float64)

        # Cell indices (lower corner)
        ix = np.floor(pos_cell[:, 0]).astype(int)
        iy = np.floor(pos_cell[:, 1]).astype(int)
        iz = np.floor(pos_cell[:, 2]).astype(int)

        # Fractional position within cell
        dx = pos_cell[:, 0] - ix
        dy = pos_cell[:, 1] - iy
        dz = pos_cell[:, 2] - iz

        # Deposit to 8 neighboring cells
        for di in [0, 1]:
            wx = (1 - dx) if di == 0 else dx
            for dj in [0, 1]:
                wy = (1 - dy) if dj == 0 else dy
                for dk in [0, 1]:
                    wz = (1 - dz) if dk == 0 else dz

                    weight = wx * wy * wz * particle_mass

                    # Handle periodic boundaries
                    iix = (ix + di) % output_cells if periodic else np.clip(ix + di, 0, output_cells - 1)
                    iiy = (iy + dj) % output_cells if periodic else np.clip(iy + dj, 0, output_cells - 1)
                    iiz = (iz + dk) % output_cells if periodic else np.clip(iz + dk, 0, output_cells - 1)

                    np.add.at(density, (iix, iiy, iiz), weight)

        # Convert to density (mass per volume)
        density /= cell_width**3

    else:
        # 2D projection
        density = np.zeros((output_cells, output_cells), dtype=np.float64)

        # Get the two axes for projection
        axes = [i for i in range(3) if i != projection_axis]
        ax1, ax2 = axes

        ix = np.floor(pos_cell[:, ax1]).astype(int)
        iy = np.floor(pos_cell[:, ax2]).astype(int)

        dx = pos_cell[:, ax1] - ix
        dy = pos_cell[:, ax2] - iy

        # Deposit to 4 neighboring cells
        for di in [0, 1]:
            wx = (1 - dx) if di == 0 else dx
            for dj in [0, 1]:
                wy = (1 - dy) if dj == 0 else dy

                weight = wx * wy * particle_mass

                iix = (ix + di) % output_cells if periodic else np.clip(ix + di, 0, output_cells - 1)
                iiy = (iy + dj) % output_cells if periodic else np.clip(iy + dj, 0, output_cells - 1)

                np.add.at(density, (iix, iiy), weight)

        # Convert to surface density (mass per area)
        density /= cell_width**2

    return {
        'density': density,
        'cells': output_cells,
        'cell_width': cell_width,
        'method': 'cic',
        'n_particles_used': n_used,
        'n_particles_total': n_particles,
    }


def compute_particle_density_sph(
    positions: NDArray[np.floating],
    box_size: float,
    output_cells: int,
    particle_mass: float = 1.0,
    kernel: str = 'cubic_spline',
    n_neighbors: int = 32,
    projection_axis: int | None = None,
    slice_bounds: Tuple[float, float] | None = None,
    periodic: bool = True,
    smoothing_length: float | None = None,
) -> dict:
    """
    Compute density field using SPH-style kernel density estimation.

    This bypasses tessellation entirely and is suitable for extreme
    scale factors (a > 10) where tetrahedra become highly elongated.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions (no Lagrangian ordering required).
    box_size : float
        Simulation box size.
    output_cells : int
        Output grid resolution per dimension.
    particle_mass : float, default=1.0
        Mass per particle.
    kernel : str, default='cubic_spline'
        Kernel type: 'cubic_spline', 'gaussian', or 'tophat'.
    n_neighbors : int, default=32
        Number of neighbors for adaptive smoothing length estimation.
    projection_axis : int, optional
        If provided, compute 2D projection along this axis (0=x, 1=y, 2=z).
    slice_bounds : tuple, optional
        If provided with projection_axis, restrict to this slice.
    periodic : bool, default=True
        Use periodic boundary conditions.
    smoothing_length : float, optional
        Fixed smoothing length. If None, computed adaptively from n_neighbors.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'density': 2D or 3D density array
        - 'cells': grid size
        - 'cell_width': physical cell width
        - 'method': 'sph'
        - 'kernel': kernel type used
        - 'mean_smoothing_length': average smoothing length

    Notes
    -----
    The SPH approach computes density by convolving particle positions with
    a smoothing kernel. This naturally handles sparse regions without the
    artifacts that tessellation produces.

    For large particle counts, this is slower than tessellation but produces
    cleaner results at extreme scale factors.

    Examples
    --------
    >>> # At a=100 where tessellation produces artifacts
    >>> result = compute_particle_density_sph(
    ...     positions, box_size=256.0, output_cells=256,
    ...     n_neighbors=32, projection_axis=2
    ... )
    >>> density = result['density']
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for SPH density estimation. "
            "Install with: pip install scipy"
        )

    positions = np.asarray(positions, dtype=np.float64)
    n_particles = len(positions)

    cell_width = box_size / output_cells
    inv_cell_width = 1.0 / cell_width

    # Filter particles if slice specified
    if projection_axis is not None and slice_bounds is not None:
        slice_min, slice_max = slice_bounds
        mask = (positions[:, projection_axis] >= slice_min) & \
               (positions[:, projection_axis] < slice_max)
        positions = positions[mask]

    n_used = len(positions)

    # Estimate smoothing lengths using KD-tree
    if smoothing_length is None:
        # Build KD-tree for neighbor finding
        if periodic:
            # For periodic, we use the tree without explicit periodicity
            # and rely on the smoothing being smaller than box_size/2
            tree = cKDTree(positions)
        else:
            tree = cKDTree(positions)

        # Find n_neighbors nearest neighbors for each particle
        # Smoothing length = distance to nth neighbor
        distances, _ = tree.query(positions, k=n_neighbors + 1)
        smoothing_lengths = distances[:, -1]  # Distance to nth neighbor
        mean_h = float(np.mean(smoothing_lengths))
    else:
        smoothing_lengths = np.full(n_used, smoothing_length)
        mean_h = smoothing_length

    # Create density grid
    if projection_axis is None:
        density = np.zeros((output_cells, output_cells, output_cells), dtype=np.float64)
    else:
        density = np.zeros((output_cells, output_cells), dtype=np.float64)

    # Deposit particles with kernel smoothing
    for i in range(n_used):
        pos = positions[i]
        h = smoothing_lengths[i]

        # Kernel support radius (depends on kernel type)
        if kernel == 'cubic_spline':
            support_radius = 2.0 * h
        elif kernel == 'gaussian':
            support_radius = 3.0 * h
        else:  # tophat
            support_radius = h

        # Grid cells affected
        if projection_axis is None:
            # 3D
            ix_min = int(np.floor((pos[0] - support_radius) * inv_cell_width))
            ix_max = int(np.ceil((pos[0] + support_radius) * inv_cell_width))
            iy_min = int(np.floor((pos[1] - support_radius) * inv_cell_width))
            iy_max = int(np.ceil((pos[1] + support_radius) * inv_cell_width))
            iz_min = int(np.floor((pos[2] - support_radius) * inv_cell_width))
            iz_max = int(np.ceil((pos[2] + support_radius) * inv_cell_width))

            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        # Cell center position
                        cx = (ix + 0.5) * cell_width
                        cy = (iy + 0.5) * cell_width
                        cz = (iz + 0.5) * cell_width

                        # Distance to particle
                        dx = cx - pos[0]
                        dy = cy - pos[1]
                        dz = cz - pos[2]

                        # Handle periodic boundaries
                        if periodic:
                            if dx > box_size / 2: dx -= box_size
                            if dx < -box_size / 2: dx += box_size
                            if dy > box_size / 2: dy -= box_size
                            if dy < -box_size / 2: dy += box_size
                            if dz > box_size / 2: dz -= box_size
                            if dz < -box_size / 2: dz += box_size

                        r = np.sqrt(dx*dx + dy*dy + dz*dz)
                        q = r / h

                        # Kernel value
                        w = _kernel_3d(q, kernel)

                        if w > 0:
                            # Map to grid index
                            gix = ix % output_cells if periodic else max(0, min(ix, output_cells - 1))
                            giy = iy % output_cells if periodic else max(0, min(iy, output_cells - 1))
                            giz = iz % output_cells if periodic else max(0, min(iz, output_cells - 1))

                            density[gix, giy, giz] += particle_mass * w / (h**3)
        else:
            # 2D projection
            axes = [j for j in range(3) if j != projection_axis]
            ax1, ax2 = axes

            ix_min = int(np.floor((pos[ax1] - support_radius) * inv_cell_width))
            ix_max = int(np.ceil((pos[ax1] + support_radius) * inv_cell_width))
            iy_min = int(np.floor((pos[ax2] - support_radius) * inv_cell_width))
            iy_max = int(np.ceil((pos[ax2] + support_radius) * inv_cell_width))

            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    cx = (ix + 0.5) * cell_width
                    cy = (iy + 0.5) * cell_width

                    dx = cx - pos[ax1]
                    dy = cy - pos[ax2]

                    if periodic:
                        if dx > box_size / 2: dx -= box_size
                        if dx < -box_size / 2: dx += box_size
                        if dy > box_size / 2: dy -= box_size
                        if dy < -box_size / 2: dy += box_size

                    r = np.sqrt(dx*dx + dy*dy)
                    q = r / h

                    w = _kernel_2d(q, kernel)

                    if w > 0:
                        gix = ix % output_cells if periodic else max(0, min(ix, output_cells - 1))
                        giy = iy % output_cells if periodic else max(0, min(iy, output_cells - 1))

                        density[gix, giy] += particle_mass * w / (h**2)

    return {
        'density': density,
        'cells': output_cells,
        'cell_width': cell_width,
        'method': 'sph',
        'kernel': kernel,
        'n_neighbors': n_neighbors,
        'mean_smoothing_length': mean_h,
        'n_particles_used': n_used,
        'n_particles_total': n_particles,
    }


def _kernel_3d(q: float, kernel_type: str) -> float:
    """Evaluate 3D SPH kernel at normalized distance q = r/h."""
    if kernel_type == 'cubic_spline':
        # M4 cubic spline kernel
        if q < 1:
            return (1 - 1.5 * q**2 + 0.75 * q**3) * (1.0 / np.pi)
        elif q < 2:
            return 0.25 * (2 - q)**3 * (1.0 / np.pi)
        else:
            return 0.0
    elif kernel_type == 'gaussian':
        if q < 3:
            return np.exp(-q**2) / (np.pi**1.5)
        else:
            return 0.0
    elif kernel_type == 'tophat':
        if q < 1:
            return 3.0 / (4.0 * np.pi)
        else:
            return 0.0
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def _kernel_2d(q: float, kernel_type: str) -> float:
    """Evaluate 2D SPH kernel at normalized distance q = r/h."""
    if kernel_type == 'cubic_spline':
        if q < 1:
            return (1 - 1.5 * q**2 + 0.75 * q**3) * (10.0 / (7.0 * np.pi))
        elif q < 2:
            return 0.25 * (2 - q)**3 * (10.0 / (7.0 * np.pi))
        else:
            return 0.0
    elif kernel_type == 'gaussian':
        if q < 3:
            return np.exp(-q**2) / np.pi
        else:
            return 0.0
    elif kernel_type == 'tophat':
        if q < 1:
            return 1.0 / np.pi
        else:
            return 0.0
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
