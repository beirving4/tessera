#!/usr/bin/env python3
"""
Shared utility functions for tessera example scripts.

This module consolidates common functionality used across multiple scripts:
- Snapshot loading and parsing
- Halo catalog discovery and loading
- ID ordering detection and Lagrangian sorting
- Common cosmology calculations

Usage:
    from utils.tessera_helpers import load_snapshot, find_halo_catalog
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

# Try to import tessera module
try:
    import tessera as ts
    HAS_TESSERA = True
except ImportError:
    try:
        import _tessera as ts
        HAS_TESSERA = True
    except ImportError:
        HAS_TESSERA = False
        ts = None

# Try h5py for direct HDF5 access
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def infer_snapshot_number(snapshot_path: Union[str, Path]) -> Optional[int]:
    """
    Infer snapshot number from filename.

    Handles patterns like:
      - snapshot_034.hdf5 -> 34
      - snapshot_ics_000.hdf5 -> -1 (ICs)
      - snapshot_074.0.hdf5 -> 74

    Parameters
    ----------
    snapshot_path : str or Path
        Path to snapshot file

    Returns
    -------
    int or None
        Snapshot number, -1 for ICs, or None if not detectable
    """
    path = Path(snapshot_path)
    name = path.stem

    # Check for ICs
    if 'ics' in name.lower():
        return -1

    # Try to extract number from snapshot_XXX pattern
    match = re.search(r'snapshot[_-]?(\d+)', name)
    if match:
        return int(match.group(1))

    return None


def find_halo_catalog(
    snapshot_path: Union[str, Path],
    snapshot_num: Optional[int] = None
) -> Optional[Path]:
    """
    Find the halo catalog file associated with a snapshot.

    Looks for fof_subhalo_tab_XXX.hdf5 in the same directory as the snapshot.

    Parameters
    ----------
    snapshot_path : str or Path
        Path to the snapshot file
    snapshot_num : int, optional
        Snapshot number (auto-detected if None)

    Returns
    -------
    Path or None
        Path to halo catalog file, or None if not found
    """
    path = Path(snapshot_path)
    directory = path.parent

    if snapshot_num is None:
        snapshot_num = infer_snapshot_number(snapshot_path)

    if snapshot_num is None or snapshot_num < 0:
        return None

    # Look for fof_subhalo_tab_XXX.hdf5
    catalog_name = f"fof_subhalo_tab_{snapshot_num:03d}.hdf5"
    catalog_path = directory / catalog_name

    if catalog_path.exists():
        return catalog_path

    # Also try without leading zeros
    catalog_name = f"fof_subhalo_tab_{snapshot_num}.hdf5"
    catalog_path = directory / catalog_name

    if catalog_path.exists():
        return catalog_path

    return None


def load_snapshot(
    path: Union[str, Path],
    snapshot_num: Optional[int] = None,
    particle_type: int = 1,
    read_ids: bool = True,
    read_velocities: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], "Gadget4Header"]:
    """
    Load particle positions from a GADGET-4 snapshot.

    Uses the tessera C++ reader for efficient HDF5 loading.

    Parameters
    ----------
    path : str or Path
        Path to snapshot file or directory containing distributed files
    snapshot_num : int, optional
        Snapshot number for distributed files (auto-detected if None)
    particle_type : int
        Particle type to load (default: 1 for dark matter)
    read_ids : bool
        Whether to read particle IDs (needed for tessellation)
    read_velocities : bool
        Whether to read velocities

    Returns
    -------
    positions : ndarray, shape (N, 3)
        Particle positions
    particle_ids : ndarray or None
        Particle IDs if read_ids=True, else None
    header : Gadget4Header
        Snapshot header with box size, redshift, etc.

    Examples
    --------
    >>> positions, ids, header = load_snapshot("snapshot_034.hdf5")
    >>> print(f"Box size: {header.box_size}")
    >>> print(f"Redshift: {header.redshift}")
    """
    if not HAS_TESSERA:
        raise ImportError(
            "Could not import tessera. Make sure the build directory "
            "is in your PYTHONPATH or the package is installed."
        )

    # Use unified interface - auto-detects single file vs distributed
    particle_types = 1 << particle_type  # Bitmask for particle type

    if snapshot_num is not None:
        snap = ts.io.read_gadget4_snapshot(
            str(path),
            snapshot_num=snapshot_num,
            particle_types=particle_types,
            read_velocities=read_velocities,
            read_ids=read_ids
        )
    else:
        snap = ts.io.read_gadget4_snapshot(
            str(path),
            particle_types=particle_types,
            read_velocities=read_velocities,
            read_ids=read_ids
        )

    # Get positions as numpy array
    coords = snap.particles[particle_type].coordinates
    positions = np.array(coords, dtype=np.float64)

    # Get particle IDs if requested
    particle_ids = None
    if read_ids:
        ids = snap.particles[particle_type].particle_ids
        particle_ids = np.array(ids, dtype=np.int64)

    return positions, particle_ids, snap.header


def load_snapshot_h5py(
    path: Union[str, Path],
    particle_type: int = 1,
    read_ids: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
    """
    Load particle data directly using h5py (fallback when tessera unavailable).

    Parameters
    ----------
    path : str or Path
        Path to snapshot HDF5 file
    particle_type : int
        Particle type to load
    read_ids : bool
        Whether to read particle IDs

    Returns
    -------
    positions : ndarray, shape (N, 3)
    particle_ids : ndarray or None
    box_size : float
    scale_factor : float
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for this function")

    with h5py.File(path, 'r') as f:
        part_key = f'PartType{particle_type}'
        positions = np.ascontiguousarray(f[f'{part_key}/Coordinates'][:], dtype=np.float64)

        particle_ids = None
        if read_ids:
            particle_ids = np.ascontiguousarray(f[f'{part_key}/ParticleIDs'][:], dtype=np.int64)

        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])

    return positions, particle_ids, box_size, scale_factor


def load_halo_catalog(catalog_path: Union[str, Path]):
    """
    Load GADGET-4 halo catalog using C++ reader.

    Parameters
    ----------
    catalog_path : str or Path
        Path to fof_subhalo_tab_XXX.hdf5 file

    Returns
    -------
    catalog : HaloCatalog
        Halo catalog object with groups and subhalos
    """
    if not HAS_TESSERA:
        raise ImportError("tessera is required for halo catalog loading")

    return ts.io.read_gadget4_halo_catalog(str(catalog_path))


def sort_positions_lagrangian(
    positions: np.ndarray,
    particle_ids: np.ndarray,
    grid_size: Optional[int] = None,
    id_offset: int = 1
) -> np.ndarray:
    """
    Sort particle positions to Lagrangian order.

    This is required before density tessellation, as tetrahedra are formed
    based on Lagrangian grid connectivity.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions in arbitrary order
    particle_ids : ndarray, shape (N,)
        Particle IDs encoding Lagrangian positions
    grid_size : int, optional
        Lagrangian grid size (auto-detected from N if None)
    id_offset : int
        Offset subtracted from IDs (default: 1 for GADGET 1-indexed)

    Returns
    -------
    sorted_positions : ndarray, shape (N, 3)
        Positions sorted into Lagrangian order
    """
    if not HAS_TESSERA:
        raise ImportError("tessera is required for Lagrangian sorting")

    n_particles = len(positions)

    if grid_size is None:
        grid_size = int(round(n_particles ** (1/3)))
        if grid_size ** 3 != n_particles:
            raise ValueError(
                f"Particle count {n_particles} is not a perfect cube. "
                f"Specify grid_size explicitly."
            )

    return ts.density.sort_by_lagrangian_id(positions, particle_ids, grid_size, id_offset)


def compute_mean_density(n_particles: int, mass_per_particle: float, box_size: float) -> float:
    """
    Compute mean 3D density of the simulation.

    Parameters
    ----------
    n_particles : int
        Total number of particles
    mass_per_particle : float
        Mass per particle in code units
    box_size : float
        Simulation box size

    Returns
    -------
    float
        Mean density (mass/volume)
    """
    total_mass = n_particles * mass_per_particle
    volume = box_size ** 3
    return total_mass / volume


def compute_mean_surface_density(
    n_particles: int,
    mass_per_particle: float,
    box_size: float,
    slice_thickness: float
) -> float:
    """
    Compute mean surface density for a slice.

    Parameters
    ----------
    n_particles : int
        Total number of particles
    mass_per_particle : float
        Mass per particle
    box_size : float
        Simulation box size
    slice_thickness : float
        Thickness of the slice

    Returns
    -------
    float
        Mean surface density (mass/area)
    """
    mean_3d_density = compute_mean_density(n_particles, mass_per_particle, box_size)
    return mean_3d_density * slice_thickness


def infer_grid_size(n_particles: int) -> int:
    """
    Infer Lagrangian grid size from particle count.

    Parameters
    ----------
    n_particles : int
        Total number of particles

    Returns
    -------
    int
        Grid size (N for N^3 particles)

    Raises
    ------
    ValueError
        If particle count is not a perfect cube
    """
    grid_size = int(round(n_particles ** (1/3)))
    if grid_size ** 3 != n_particles:
        raise ValueError(
            f"Particle count {n_particles} is not a perfect cube. "
            f"Tessellation requires N^3 particles."
        )
    return grid_size


def extract_2d_slice(
    density_3d: np.ndarray,
    slice_axis: int,
    slice_min_idx: int,
    slice_max_idx: int,
    cell_width: float
) -> np.ndarray:
    """
    Extract and project a 2D slice from a 3D density field.

    Parameters
    ----------
    density_3d : ndarray, shape (cells, cells, cells)
        3D density field (ordered as [z, y, x] from C++ layout)
    slice_axis : int
        Axis perpendicular to slice (0=z, 1=y, 2=x in [z,y,x] ordering)
    slice_min_idx : int
        Start index along slice axis
    slice_max_idx : int
        End index along slice axis
    cell_width : float
        Width of each cell (for surface density conversion)

    Returns
    -------
    density_2d : ndarray, shape (cells, cells)
        2D surface density field (mass/area)
    """
    if slice_axis == 0:  # z slice
        density_2d = density_3d[slice_min_idx:slice_max_idx, :, :].sum(axis=0)
    elif slice_axis == 1:  # y slice
        density_2d = density_3d[:, slice_min_idx:slice_max_idx, :].sum(axis=1)
    else:  # x slice
        density_2d = density_3d[:, :, slice_min_idx:slice_max_idx].sum(axis=2)

    # Convert to surface density (multiply by cell width)
    return density_2d * cell_width


def save_density_hdf5(
    filename: Union[str, Path],
    density: np.ndarray,
    header,
    metadata: dict,
    extent: Optional[Tuple[float, float, float, float]] = None
):
    """
    Save a density field to an HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Output filename
    density : ndarray
        2D or 3D density array
    header : Gadget4Header
        Snapshot header with cosmological info
    metadata : dict
        Additional metadata to store (slice info, method, etc.)
    extent : tuple, optional
        Spatial extent (x_min, x_max, y_min, y_max) for 2D fields
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for saving HDF5 files")

    with h5py.File(filename, 'w') as f:
        # Store the density field
        dset = f.create_dataset('density', data=density, compression='gzip')
        dset.attrs['units'] = 'mass / volume (code units)' if density.ndim == 3 else 'mass / area (code units)'

        # Store header/cosmology info
        hdr = f.create_group('Header')
        hdr.attrs['BoxSize'] = header.box_size
        hdr.attrs['Redshift'] = header.redshift
        hdr.attrs['Time'] = header.time
        hdr.attrs['NumPartTotal'] = list(header.num_part_total)
        hdr.attrs['MassTable'] = list(header.mass_table)

        # Store metadata
        meta = f.create_group('Metadata')
        for key, val in metadata.items():
            if isinstance(val, np.ndarray):
                meta.create_dataset(key, data=val)
            else:
                meta.attrs[key] = val

        # Store grid extent for 2D fields
        if extent is not None:
            grid = f.create_group('Grid')
            grid.attrs['extent'] = extent
            grid.attrs['x_min'] = extent[0]
            grid.attrs['x_max'] = extent[1]
            grid.attrs['y_min'] = extent[2]
            grid.attrs['y_max'] = extent[3]

    print(f"Saved density field to {filename}")


# Visualization helpers

def get_axis_labels(projection_axis: str) -> Tuple[str, str]:
    """
    Get axis labels for a given projection axis.

    Parameters
    ----------
    projection_axis : str
        Projection axis ('x', 'y', or 'z')

    Returns
    -------
    xlabel, ylabel : str
        Labels for the two remaining axes
    """
    axis_map = {
        'x': ('Y', 'Z'),
        'y': ('X', 'Z'),
        'z': ('X', 'Y'),
    }
    return axis_map[projection_axis.lower()]


def check_tessera_available() -> bool:
    """Check if tessera C++ module is available."""
    return HAS_TESSERA


def check_h5py_available() -> bool:
    """Check if h5py is available."""
    return HAS_H5PY
