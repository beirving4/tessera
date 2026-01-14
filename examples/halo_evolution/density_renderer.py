"""
Density Field Renderer for Halo Evolution

Uses tessera to compute density fields around halos at each epoch.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys
import numpy as np

# Import tessera (handles platform-specific configuration automatically)
HAS_TESSERA = False
ts = None

try:
    import tessera as ts
    HAS_TESSERA = hasattr(ts, 'density') and ts.density is not None
except ImportError:
    # Fallback for development: try to import native module directly
    _script_dir = Path(__file__).parent.resolve()
    _repo_root = _script_dir.parent.parent
    _build_dir = _repo_root / 'build'
    if _build_dir.exists():
        sys.path.insert(0, str(_build_dir))
    try:
        import _tessera as ts
        HAS_TESSERA = True
    except ImportError:
        pass

from .halo_tracker import HaloInfo


@dataclass
class DensityRenderConfig:
    """Configuration for density field rendering."""

    # Box sizing
    box_mode: str = 'fixed_comoving'
    """Box sizing mode:
    - 'fixed_comoving': Fixed comoving size (default)
    - 'r200_multiple': Multiple of halo R200
    - 'fixed_physical': Fixed physical size (scales with a)
    """

    box_size: float = 1.5
    """Size in Mpc/h (for fixed modes) or R200 multiple."""

    min_box_size: float = 0.5
    """Minimum box size in Mpc/h (for r200_multiple mode)."""

    # Resolution
    output_cells: int = 256
    """Grid resolution per dimension."""

    n_samples: int = 100
    """Monte Carlo samples per tetrahedron."""

    # Projection (for 2D)
    projection_axis: int = 2
    """Axis to project along: 0=x, 1=y, 2=z."""

    # tessera settings
    n_threads: int = 0
    """Number of threads (0=auto)."""

    particle_type: int = 1
    """Particle type to use (1=dark matter)."""

    id_offset: int = 1
    """Offset for particle IDs when sorting by Lagrangian ID."""


class DensityRenderer:
    """
    Render density fields around halos using tessera.

    Parameters
    ----------
    snapshot_dir : Path
        Directory containing snapshot files
    snapshot_pattern : str
        Format pattern for snapshot filenames
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
        snapshot_dir: Union[str, Path],
        snapshot_pattern: str = 'snapshot_{:03d}.hdf5',
        config: Optional[DensityRenderConfig] = None,
    ):
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
        halo: HaloInfo,
        box_size: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute subbox origin and width for a halo.

        Returns (origin, width) where origin is the corner position.
        """
        cfg = self.config

        # Determine box width based on mode
        if cfg.box_mode == 'fixed_comoving':
            width = cfg.box_size
        elif cfg.box_mode == 'r200_multiple':
            r200 = halo.r200c if halo.r200c > 0 else cfg.min_box_size
            width = max(r200 * cfg.box_size, cfg.min_box_size)
        elif cfg.box_mode == 'fixed_physical':
            # Physical size = comoving size * scale_factor
            width = cfg.box_size / halo.scale_factor
        else:
            raise ValueError(f"Unknown box_mode: {cfg.box_mode}")

        # Compute origin (corner of box centered on halo)
        half_width = width / 2.0
        origin = halo.position - half_width

        # Handle periodic wrapping
        for i in range(3):
            if origin[i] < 0:
                origin[i] += box_size
            elif origin[i] >= box_size:
                origin[i] -= box_size

        return origin, width

    def render_single(
        self,
        halo: HaloInfo,
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
        sorted_positions, box_size, grid_size = self._load_and_sort_particles(
            halo.snap_num
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
        branch: List[HaloInfo],
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
                densities[halo.scale_factor] = density
            except FileNotFoundError as e:
                print(f"Warning: Skipping a={halo.scale_factor:.3f}: {e}")
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
