"""
Configuration and utilities for time-series visualization pipeline.

This module provides the TimeSeriesConfig dataclass and utility functions
for managing the panoramic cosmic web visualization pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import h5py


@dataclass
class TimeSeriesConfig:
    """Configuration for time-series visualization pipeline."""

    # Input/Output paths
    snapshot_dir: Path
    output_dir: Path
    snapshot_pattern: str = "snapshot_{:03d}.hdf5"  # Format string for snapshot files

    # Simulation parameters
    box_size: float = 256.0                    # Comoving box size [Mpc/h]
    n_particles_per_dim: int = 256             # Particles per dimension (N^3 total)

    # Density field parameters
    output_cells: int = 512                    # Output grid resolution
    n_samples: int = 100                       # Monte Carlo samples per tetrahedron
    n_threads: int = 0                         # OpenMP threads (0 = auto)

    # Projection parameters
    projection_axis: int = 2                   # Axis to project along (0=x, 1=y, 2=z)
    slab_fraction: float = 1.0                # Fraction of box depth for slab (1.0 = full box, 0.08 = 8%)
    slab_center: float = 0.5                  # Slab center as fraction of box (0.5 = middle)

    # Time-series image parameters
    n_box_replications: int = 4               # Number of box tilings in x (time) direction
    a_min: float = 0.02                       # Starting scale factor
    a_max: float = 100.0                      # Ending scale factor
    log_time_mapping: bool = True             # Map x to log(a) if True, else linear
    use_physical_density: bool = False        # Scale by a^-3 for physical density

    # Visualization parameters
    log_density_scale: bool = True            # Apply log10 to density
    percentile_clip: Tuple[float, float] = (0.5, 99.9)  # Percentile clipping for colormap
    colormap: str = "cosmic_blue"             # Matplotlib colormap name

    # Animation parameters
    fps: int = 15                             # Frames per second
    interpolate_frames: int = 1               # Frames to interpolate between snapshots (1 = none)
    dpi: int = 150                            # Output DPI for animation

    # Snapshot selection (if None, use all available)
    snapshot_indices: Optional[List[int]] = None
    scale_factors: Optional[List[float]] = None  # If provided, maps snapshot index to scale factor

    # Internal: for overriding file paths when running standalone scripts
    _projections_file_override: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        self.snapshot_dir = Path(self.snapshot_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._projections_file_override is not None:
            self._projections_file_override = Path(self._projections_file_override)

    @property
    def projections_file(self) -> Path:
        if self._projections_file_override is not None:
            return self._projections_file_override
        return self.output_dir / "density_projections.h5"

    @property
    def static_image_file(self) -> Path:
        return self.output_dir / "time_series.png"

    @property
    def animation_file(self) -> Path:
        return self.output_dir / "evolution.mp4"


def get_snapshot_files(config: TimeSeriesConfig) -> List[Tuple[int, Path]]:
    """
    Find all snapshot files matching the pattern.

    Returns list of (index, path) tuples sorted by index.
    """
    snapshot_files = []

    if config.snapshot_indices is not None:
        # Use specified indices
        for idx in config.snapshot_indices:
            path = config.snapshot_dir / config.snapshot_pattern.format(idx)
            if path.exists():
                snapshot_files.append((idx, path))
            else:
                print(f"Warning: Snapshot {path} not found")
    else:
        # Auto-discover snapshots
        for path in sorted(config.snapshot_dir.glob("snapshot_*.hdf5")):
            # Extract index from filename
            try:
                idx = int(path.stem.split("_")[-1])
                snapshot_files.append((idx, path))
            except ValueError:
                continue

    return sorted(snapshot_files, key=lambda x: x[0])


def get_scale_factor(snapshot_path: Path, snapshot_idx: int, config: TimeSeriesConfig) -> float:
    """
    Get scale factor for a snapshot.

    Tries in order:
    1. config.scale_factors lookup
    2. HDF5 header attribute
    3. Raises error if neither available
    """
    if config.scale_factors is not None and snapshot_idx < len(config.scale_factors):
        return config.scale_factors[snapshot_idx]

    # Try reading from HDF5 header
    with h5py.File(snapshot_path, 'r') as f:
        header = f['Header'].attrs
        if 'Time' in header:
            return float(header['Time'])
        elif 'ExpansionFactor' in header:
            return float(header['ExpansionFactor'])
        elif 'Redshift' in header:
            z = float(header['Redshift'])
            return 1.0 / (1.0 + z)

    raise ValueError(f"Could not determine scale factor for {snapshot_path}")


def save_projection(
    projections_file: Path,
    scale_factor: float,
    density_2d: np.ndarray,
    snapshot_idx: int
):
    """Save a 2D projection to the HDF5 file."""
    with h5py.File(projections_file, 'a') as f:
        # Use scale factor as dataset name (formatted to avoid floating point issues)
        dset_name = f"a_{scale_factor:.6f}"

        if dset_name in f:
            del f[dset_name]

        dset = f.create_dataset(dset_name, data=density_2d, compression='gzip')
        dset.attrs['scale_factor'] = scale_factor
        dset.attrs['snapshot_idx'] = snapshot_idx


def load_projections(projections_file: Path) -> dict:
    """
    Load all projections from HDF5 file.

    Returns dict mapping scale_factor -> 2D density array.
    """
    projections = {}

    with h5py.File(projections_file, 'r') as f:
        for key in f.keys():
            if key.startswith('a_'):
                dset = f[key]
                a = float(dset.attrs['scale_factor'])
                projections[a] = dset[:]

    return projections


def cosmic_time_gyr(a: float, H0: float = 70.0, Om: float = 0.3, OL: float = 0.7) -> float:
    """
    Compute cosmic time in Gyr for a given scale factor.

    Uses flat LCDM cosmology with default Planck-like parameters.

    Parameters
    ----------
    a : float
        Scale factor
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float
        Matter density parameter
    OL : float
        Dark energy density parameter

    Returns
    -------
    t : float
        Cosmic time in Gyr
    """
    from scipy import integrate

    # Convert H0 to 1/Gyr
    H0_per_gyr = H0 * 1.0227e-3  # km/s/Mpc to 1/Gyr

    def integrand(a_prime):
        return 1.0 / (a_prime * np.sqrt(Om / a_prime**3 + OL))

    result, _ = integrate.quad(integrand, 0, a)
    return result / H0_per_gyr
