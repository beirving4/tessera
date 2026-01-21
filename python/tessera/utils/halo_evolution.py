"""
Halo Evolution Utilities.

This module provides utilities for tracing and visualizing halo evolution
through merger trees, including:

- MergerTree: High-performance C++ reader for GADGET-4 merger tree files
- HaloTracker: Branch tracing through merger trees
- HaloInfo: Halo information at a single snapshot
- DensityRenderer: Compute density fields around halos using tessera

All functions use type hints compatible with Python 3.10+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

# Lazy import to avoid circular imports when inside tessera package
ts = None
HAS_TESSERA = False
HAS_MERGER_TREE = False

# Placeholder values - will be set by _init_tessera()
MergerTree = None
MergerTreeHeader = None
HaloTracker = None
HaloInfo = None
TreeTableEntry = None
read_merger_tree_header = None

_initialized = False

def _init_tessera():
    """Initialize tessera imports lazily."""
    global ts, HAS_TESSERA, HAS_MERGER_TREE, _initialized
    global MergerTree, MergerTreeHeader, HaloTracker, HaloInfo, TreeTableEntry, read_merger_tree_header

    if _initialized:
        return
    _initialized = True

    try:
        # Try native module first
        import _tessera as _ts
        ts = _ts
    except ImportError:
        try:
            import tessera as _ts
            ts = _ts
        except ImportError:
            return

    HAS_TESSERA = hasattr(ts, 'density') and ts.density is not None

    if hasattr(ts.io, 'MergerTree'):
        MergerTree = ts.io.MergerTree
        MergerTreeHeader = ts.io.MergerTreeHeader
        HaloTracker = ts.io.HaloTracker
        HaloInfo = ts.io.HaloInfo
        TreeTableEntry = ts.io.TreeTableEntry
        read_merger_tree_header = ts.io.read_merger_tree_header
        HAS_MERGER_TREE = True


@dataclass
class DensityRenderConfig:
    """
    Configuration for density field rendering.

    Attributes
    ----------
    box_mode : str
        Box sizing mode:
        - 'fixed_comoving': Fixed comoving size (default)
        - 'r200_multiple': Multiple of halo R200
        - 'fixed_physical': Fixed physical size (scales with a)
    box_size : float
        Size in Mpc/h (for fixed modes) or R200 multiple
    min_box_size : float
        Minimum box size in Mpc/h (for r200_multiple mode)
    output_cells : int
        Grid resolution per dimension
    n_samples : int
        Monte Carlo samples per tetrahedron
    projection_axis : int
        Axis to project along: 0=x, 1=y, 2=z
    n_threads : int
        Number of threads (0=auto)
    particle_type : int
        Particle type to use (1=dark matter)
    id_offset : int
        Offset for particle IDs when sorting by Lagrangian ID
    """

    box_mode: str = 'fixed_comoving'
    box_size: float = 1.5
    min_box_size: float = 0.5
    output_cells: int = 256
    n_samples: int = 100
    projection_axis: int = 2
    n_threads: int = 0
    particle_type: int = 1
    id_offset: int = 1


class DensityRenderer:
    """
    Render density fields around halos using tessera.

    Uses the high-performance C++ tessera density computation to generate
    2D or 3D density fields centered on halo positions at each snapshot.

    Parameters
    ----------
    snapshot_dir : Path
        Directory containing snapshot files
    snapshot_pattern : str
        Format pattern for snapshot filenames (e.g., 'snapshot_{:03d}.hdf5')
    config : DensityRenderConfig, optional
        Rendering configuration

    Examples
    --------
    >>> renderer = DensityRenderer(
    ...     snapshot_dir='/path/to/snapshots',
    ...     snapshot_pattern='snapshot_{:03d}.hdf5'
    ... )
    >>>
    >>> # Render density for a traced branch
    >>> densities = renderer.render_branch(branch, mode='2d')
    >>>
    >>> # Result: dict mapping scale_factor -> 2D array
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        snapshot_pattern: str = 'snapshot_{:03d}.hdf5',
        config: DensityRenderConfig | None = None,
    ):
        _init_tessera()  # Initialize lazy imports
        if not HAS_TESSERA:
            raise ImportError(
                "tessera module not available. Build with: pip install ."
            )

        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_pattern = snapshot_pattern
        self.config = config or DensityRenderConfig()

        # Cache for loaded/sorted particles
        self._particle_cache: Dict[int, Tuple[np.ndarray, float, int]] = {}

    def _get_snapshot_path(self, snap_num: int) -> Path:
        """Get path to snapshot file."""
        filename = self.snapshot_pattern.format(snap_num)
        return self.snapshot_dir / filename

    def _load_and_sort_particles(
        self, snap_num: int
    ) -> Tuple[np.ndarray, float, int]:
        """
        Load snapshot and sort particles by Lagrangian ID.

        Returns (sorted_positions, box_size, grid_size)
        """
        if snap_num in self._particle_cache:
            return self._particle_cache[snap_num]

        snap_path = self._get_snapshot_path(snap_num)
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")

        # Load snapshot
        particle_types = 1 << self.config.particle_type
        snapshot = ts.io.read_gadget4_snapshot(
            str(snap_path),
            particle_types=particle_types,
            read_velocities=False,
            read_ids=True,
        )

        # Get particle data
        particles = snapshot.particles[self.config.particle_type]
        positions = np.array(particles.coordinates, dtype=np.float64)
        particle_ids = np.array(particles.particle_ids, dtype=np.int64)
        box_size = snapshot.header.box_size

        # Infer Lagrangian grid size
        n_particles = len(positions)
        grid_size = int(round(n_particles ** (1 / 3)))
        if grid_size ** 3 != n_particles:
            raise ValueError(
                f"Particle count {n_particles} is not a perfect cube."
            )

        # Sort by Lagrangian ID
        sorted_positions = ts.density.sort_by_lagrangian_id(
            positions, particle_ids, grid_size,
            id_offset=self.config.id_offset
        )

        # Cache result
        self._particle_cache[snap_num] = (sorted_positions, box_size, grid_size)

        return sorted_positions, box_size, grid_size

    def _compute_box_params(
        self,
        halo,
        box_size: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute subbox origin and width for a halo.

        Returns (origin, width) where origin is the corner position.
        """
        cfg = self.config

        # Get position as numpy array
        if hasattr(halo, 'position'):
            position = np.array(halo.position)
        else:
            position = np.array([halo.pos_x, halo.pos_y, halo.pos_z])

        # Get R200c
        r200c = getattr(halo, 'r200c', 0.0)

        # Determine box width based on mode
        if cfg.box_mode == 'fixed_comoving':
            width = cfg.box_size
        elif cfg.box_mode == 'r200_multiple':
            r200 = r200c if r200c > 0 else cfg.min_box_size
            width = max(r200 * cfg.box_size, cfg.min_box_size)
        elif cfg.box_mode == 'fixed_physical':
            scale_factor = getattr(halo, 'scale_factor', 1.0)
            width = cfg.box_size / scale_factor
        else:
            raise ValueError(f"Unknown box_mode: {cfg.box_mode}")

        # Compute origin (corner of box centered on halo)
        half_width = width / 2.0
        origin = position - half_width

        # Handle periodic wrapping
        for i in range(3):
            if origin[i] < 0:
                origin[i] += box_size
            elif origin[i] >= box_size:
                origin[i] -= box_size

        return origin, width

    def render_single(
        self,
        halo,
        mode: str = '2d',
    ) -> np.ndarray:
        """
        Render density field around a single halo.

        Parameters
        ----------
        halo : HaloInfo
            Halo information (position, snapshot, etc.)
        mode : str
            '2d' for projected density, '3d' for volume

        Returns
        -------
        density : np.ndarray
            2D array (mode='2d') or 3D array (mode='3d')
        """
        cfg = self.config

        # Load and sort particles
        snap_num = halo.snap_num
        sorted_positions, box_size, grid_size = self._load_and_sort_particles(
            snap_num
        )

        # Compute subbox parameters
        origin, width = self._compute_box_params(halo, box_size)

        # Configure tessera
        config = ts.density.TetraDensityConfig()
        config.box_size = box_size
        config.lagrangian_grid_size = grid_size
        config.output_cells = cfg.output_cells
        config.n_samples = cfg.n_samples
        config.n_threads = cfg.n_threads
        config.periodic = True
        config.particle_mass = 1.0

        # Enable subbox
        config.subbox_enabled = True
        config.subbox_origin = tuple(origin)
        config.subbox_width = (width, width, width)

        # Compute density
        if mode == '2d':
            result = ts.density.compute_tetra_density_2d_projection(
                sorted_positions, config, cfg.projection_axis
            )
            density = np.array(result.density).reshape(
                cfg.output_cells, cfg.output_cells
            )
        elif mode == '3d':
            result = ts.density.compute_tetra_density_3d(
                sorted_positions, config
            )
            density = np.array(result.density).reshape(
                cfg.output_cells, cfg.output_cells, cfg.output_cells
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return density

    def render_branch(
        self,
        branch: List,
        mode: str = '2d',
        progress_callback=None,
    ) -> Dict[float, np.ndarray]:
        """
        Render density fields for entire branch.

        Parameters
        ----------
        branch : List[HaloInfo]
            Halo info at each snapshot, sorted by scale factor
        mode : str
            '2d' or '3d'
        progress_callback : callable, optional
            Called with (current_index, total) for progress reporting

        Returns
        -------
        densities : dict
            Mapping scale_factor -> density array
        """
        densities = {}

        for i, halo in enumerate(branch):
            if progress_callback:
                progress_callback(i, len(branch))

            try:
                density = self.render_single(halo, mode=mode)
                scale_factor = getattr(halo, 'scale_factor', float(i))
                densities[scale_factor] = density
            except FileNotFoundError as e:
                scale_factor = getattr(halo, 'scale_factor', float(i))
                print(f"Warning: Skipping a={scale_factor:.3f}: {e}")
                continue

        return densities

    def compute_overdensity(
        self,
        density: np.ndarray,
        box_size: float,
        total_mass: float,
    ) -> np.ndarray:
        """
        Convert density to overdensity (1 + delta).

        Parameters
        ----------
        density : np.ndarray
            Density field
        box_size : float
            Simulation box size
        total_mass : float
            Total mass in the simulation

        Returns
        -------
        overdensity : np.ndarray
            1 + delta where delta = (rho - rho_mean) / rho_mean
        """
        mean_density = total_mass / (box_size ** 3)
        return density / mean_density

    def clear_cache(self) -> None:
        """Clear the particle cache to free memory."""
        self._particle_cache.clear()


def check_merger_tree_available() -> bool:
    """Check if C++ MergerTree is available."""
    _init_tessera()  # Trigger lazy import
    return HAS_MERGER_TREE


__all__ = [
    # C++ re-exports
    'MergerTree',
    'MergerTreeHeader',
    'HaloTracker',
    'HaloInfo',
    'TreeTableEntry',
    'read_merger_tree_header',
    # Density rendering
    'DensityRenderConfig',
    'DensityRenderer',
    # Availability check
    'check_merger_tree_available',
]
