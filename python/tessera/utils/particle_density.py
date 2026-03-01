"""
Particle-based density estimation for artifact-free visualization.

This module provides SPH-style and Cloud-in-Cell (CIC) density estimation
that bypasses tessellation entirely. Useful for extreme scale factors (a > 10)
where tetrahedra become highly elongated and tessellation produces artifacts.

Also provides ``compute_cic_density_pdf`` for computing mass-weighted and
volume-weighted overdensity PDFs from CIC output in a single call.

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


# Import tessera C++ module (needed for compute_cic_density_pdf)
try:
    import tessera as _ts
except ImportError:
    try:
        import _tessera as _ts
    except ImportError:
        _ts = None


def compute_cic_density_pdf(
    positions: NDArray[np.floating],
    box_size: float,
    output_cells: int = 256,
    n_bins: int = 100,
    range_min: float = 0.0,
    range_max: float = 0.0,
    particle_mass: float = 1.0,
    n_threads: int = 0,
    periodic: bool = True,
    morphology: NDArray[np.uint8] | None = None,
    jackknife: bool = False,
    jackknife_subboxes: int = 2,
) -> dict:
    """
    Compute mass-weighted and volume-weighted overdensity PDFs from CIC density.

    This is a convenience function that calls the C++ CIC density estimator,
    computes overdensity (1+delta) using the appropriate mean for each weighting,
    and returns both PDFs along with the raw overdensity arrays.

    **Volume-weighted PDF**: Each grid cell contributes equally (uniform volume).
    Overdensity is normalised by the grid mean: ``rho_V = mean(grid_density)``.

    **Mass-weighted PDF**: Each particle contributes equally (uniform mass).
    Overdensity is normalised by the particle mean: ``rho_M = mean(particle_density)``.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions (no Lagrangian ordering required).
    box_size : float
        Physical size of the periodic simulation box.
    output_cells : int, default=256
        Grid resolution (cells per dimension).
    n_bins : int, default=100
        Number of logarithmic bins for the PDFs.
    range_min : float, default=0.0
        Lower edge of the bin range in (1+delta). If 0, auto-detected from
        the 0.005th percentile of positive overdensity values.
    range_max : float, default=0.0
        Upper edge of the bin range in (1+delta). If 0, auto-detected from
        the 99.995th percentile of positive overdensity values.
    particle_mass : float, default=1.0
        Mass per particle.
    n_threads : int, default=0
        Number of OpenMP threads. 0 = auto-detect (capped at 8).
    periodic : bool, default=True
        Use periodic boundary conditions.
    morphology : ndarray of uint8 or None, default=None
        ORIGAMI morphology classes (0=void, 1=wall, 2=filament, 3=halo).
        If provided, per-class mass-weighted PDFs are computed.
    jackknife : bool, default=False
        Enable jackknife resampling for uncertainty estimation on mass-weighted PDFs.
    jackknife_subboxes : int, default=2
        Sub-boxes per dimension for jackknife (2->8, 3->27).

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``pdf_mass_weighted``: mass-weighted PDF values, shape (n_bins,)
        - ``pdf_volume_weighted``: volume-weighted PDF values, shape (n_bins,)
        - ``bin_edges``: log-spaced bin edges in (1+delta), shape (n_bins+1,)
        - ``bin_centers``: geometric bin centers, shape (n_bins,)
        - ``overdensity_grid``: (1+delta) on the grid, shape (output_cells,)*3
        - ``overdensity_particles``: (1+delta) per particle, shape (N,)
        - ``rho_V``: volume-weighted mean density = total_mass / box_volume
        - ``rho_M``: mass-weighted mean density = mean(particle_density)
        - ``n_particles``: number of particles
        - ``output_cells``: grid resolution used

        When ``morphology`` is provided:

        - ``pdf_void_mass_weighted``, ``pdf_wall_mass_weighted``,
          ``pdf_filament_mass_weighted``, ``pdf_halo_mass_weighted``:
          per-class mass-weighted PDFs, shape (n_bins,)

        When ``jackknife=True``:

        - ``pdf_mass_weighted_error``: jackknife error for global mass-weighted PDF
        - ``pdf_void_mass_weighted_error``, ``pdf_wall_mass_weighted_error``,
          ``pdf_filament_mass_weighted_error``, ``pdf_halo_mass_weighted_error``:
          per-class jackknife errors (only when ``morphology`` is also provided)

    Raises
    ------
    ImportError
        If the tessera C++ module is not available.

    Examples
    --------
    >>> result = compute_cic_density_pdf(positions, box_size=256.0, output_cells=128)
    >>> plt.loglog(result['bin_centers'], result['pdf_mass_weighted'], label='mass')
    >>> plt.loglog(result['bin_centers'], result['pdf_volume_weighted'], label='volume')
    """
    if _ts is None:
        raise ImportError(
            "tessera C++ module is required for compute_cic_density_pdf. "
            "Install with: pip install ."
        )

    positions = np.ascontiguousarray(positions, dtype=np.float64)

    # Run C++ CIC density with particle sampling enabled
    cic_result = _ts.density.compute_cic_density(
        positions,
        box_size=box_size,
        output_cells=output_cells,
        particle_mass=particle_mass,
        n_threads=n_threads,
        periodic=periodic,
        sample_at_particles=True,
    )

    grid_density = np.array(cic_result.grid_density)
    particle_density = np.array(cic_result.particle_density)

    # Compute mean densities
    rho_V = float(np.mean(grid_density))          # volume-weighted mean
    rho_M = float(np.mean(particle_density))       # mass-weighted mean

    # Compute overdensity (1+delta) with appropriate normalisation
    overdensity_grid = grid_density / rho_V
    overdensity_particles = particle_density / rho_M

    # Auto-detect bin range from the union of both arrays (positive values only)
    od_grid_pos = overdensity_grid[overdensity_grid > 0].ravel()
    od_part_pos = overdensity_particles[overdensity_particles > 0]

    if range_min <= 0.0 or range_max <= 0.0:
        all_positive = np.concatenate([od_grid_pos, od_part_pos])
        if range_min <= 0.0:
            range_min = float(np.percentile(all_positive, 0.005))
        if range_max <= 0.0:
            range_max = float(np.percentile(all_positive, 99.995))

    bin_edges = np.logspace(np.log10(range_min), np.log10(range_max), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # Volume-weighted PDF: histogram grid cells with uniform weights
    hist_vol, _ = np.histogram(od_grid_pos, bins=bin_edges)
    total_vol = float(np.sum(hist_vol))
    pdf_volume_weighted = hist_vol / (total_vol * bin_widths) if total_vol > 0 else hist_vol.astype(float)

    # Build result dict
    result = {
        # PDFs
        'pdf_volume_weighted': pdf_volume_weighted,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        # Raw data
        'overdensity_grid': overdensity_grid,
        'overdensity_particles': overdensity_particles,
        # Normalisation info
        'rho_V': rho_V,
        'rho_M': rho_M,
        # Metadata
        'n_particles': int(cic_result.n_particles),
        'output_cells': int(cic_result.output_cells),
    }

    # Validate morphology if provided
    if morphology is not None:
        morphology = np.ascontiguousarray(morphology, dtype=np.uint8)
        if len(morphology) != len(overdensity_particles):
            raise ValueError(
                f"morphology length ({len(morphology)}) must match "
                f"number of particles ({len(overdensity_particles)})"
            )

    class_names = ['void', 'wall', 'filament', 'halo']

    if jackknife and morphology is not None:
        # Jackknife with morphology conditioning — use C++ for performance
        jk_result = _ts.stats.compute_jackknife_conditional_histogram(
            positions,
            np.ascontiguousarray(overdensity_particles, dtype=np.float64),
            morphology,
            box_size,
            n_bins,
            range_min,
            range_max,
            True,  # log_bins
            jackknife_subboxes,
            n_threads,
        )

        # Global mass-weighted PDF and error
        result['pdf_mass_weighted'] = np.array(jk_result.all.global_pdf)
        result['pdf_mass_weighted_error'] = np.array(jk_result.all.pdf_error)

        # Per-class PDFs and errors
        for cls_name, cls_result in zip(
            class_names,
            [jk_result.void_class, jk_result.wall_class,
             jk_result.filament_class, jk_result.halo_class],
        ):
            result[f'pdf_{cls_name}_mass_weighted'] = np.array(cls_result.global_pdf)
            result[f'pdf_{cls_name}_mass_weighted_error'] = np.array(cls_result.pdf_error)

    elif jackknife and morphology is None:
        # Jackknife without morphology — unconditional
        jk_result = _ts.stats.compute_jackknife_histogram(
            positions,
            np.ascontiguousarray(overdensity_particles, dtype=np.float64),
            box_size,
            n_bins,
            range_min,
            range_max,
            True,  # log_bins
            jackknife_subboxes,
            n_threads,
        )

        result['pdf_mass_weighted'] = np.array(jk_result.global_pdf)
        result['pdf_mass_weighted_error'] = np.array(jk_result.pdf_error)

    elif morphology is not None:
        # Per-class PDFs without jackknife — simple histogram per class
        # Global mass-weighted PDF
        hist_mass, _ = np.histogram(od_part_pos, bins=bin_edges)
        total_mass = float(np.sum(hist_mass))
        result['pdf_mass_weighted'] = (
            hist_mass / (total_mass * bin_widths) if total_mass > 0
            else hist_mass.astype(float)
        )

        for cls_idx, cls_name in enumerate(class_names):
            mask = morphology == cls_idx
            od_cls = overdensity_particles[mask]
            od_cls_pos = od_cls[od_cls > 0]
            hist_cls, _ = np.histogram(od_cls_pos, bins=bin_edges)
            total_cls = float(np.sum(hist_cls))
            result[f'pdf_{cls_name}_mass_weighted'] = (
                hist_cls / (total_cls * bin_widths) if total_cls > 0
                else hist_cls.astype(float)
            )

    else:
        # No morphology, no jackknife — simple global mass-weighted PDF
        hist_mass, _ = np.histogram(od_part_pos, bins=bin_edges)
        total_mass = float(np.sum(hist_mass))
        result['pdf_mass_weighted'] = (
            hist_mass / (total_mass * bin_widths) if total_mass > 0
            else hist_mass.astype(float)
        )

    return result
