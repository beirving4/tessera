#!/usr/bin/env python3
"""
Tetrahedron-based Density Field Computation

This module implements the gotetra algorithm for computing density fields from
N-body simulations using phase-space tessellation.

The key idea:
1. Particles start on a regular Lagrangian grid (N^3 particles)
2. Each Lagrangian cell is decomposed into 6 tetrahedra
3. Monte Carlo samples within each tetrahedron deposit mass onto the grid
4. This properly handles stream crossing and multi-valued velocity fields

This Python implementation serves as a reference. For production use, prefer
the C++ implementation via `asymptotic_tetra.density.compute_tetra_density_3d()`.
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

# Try to import C++ module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

try:
    import _asymptotic_tetra as at
    HAS_CPP = True
except ImportError:
    try:
        import asymptotic_tetra as _at
        at = _at
        HAS_CPP = True
    except ImportError:
        HAS_CPP = False
        at = None


# =============================================================================
# Tetrahedron direction table (from gotetra)
# =============================================================================

# Each Lagrangian cell is decomposed into 6 tetrahedra.
# Corner ordering for direction d:
#   corners[0] = origin + DIRS[d][0]
#   corners[1] = origin + DIRS[d][1]
#   corners[2] = origin + (1,1,1)
#   corners[3] = origin
DIRS = np.array([
    [[1, 0, 0], [1, 1, 0]],  # dir 0
    [[1, 0, 0], [1, 0, 1]],  # dir 1
    [[0, 1, 0], [1, 1, 0]],  # dir 2
    [[0, 0, 1], [1, 0, 1]],  # dir 3
    [[0, 1, 0], [0, 1, 1]],  # dir 4
    [[0, 0, 1], [0, 1, 1]],  # dir 5
], dtype=np.int64)


# =============================================================================
# Core algorithms
# =============================================================================

def rocchini_cignoni_transform(samples: np.ndarray) -> np.ndarray:
    """
    Apply Rocchini-Cignoni transformation to map unit cube to unit tetrahedron.
    
    This transforms uniformly distributed points in [0,1]^3 to uniformly
    distributed barycentric coordinates (s, t, u, v) where v = 1-s-t-u.
    
    Reference: C. Rocchini, P. Cignoni, "Generating Random Points in a Tetrahedron",
    Journal of Graphics Tools, 2001.
    
    Parameters
    ----------
    samples : ndarray, shape (N, 3)
        Points uniformly distributed in [0,1]^3
    
    Returns
    -------
    bary : ndarray, shape (N, 4)
        Barycentric coordinates (s, t, u, v) where v = 1-s-t-u
    """
    s = samples[:, 0].copy()
    t = samples[:, 1].copy()
    u = samples[:, 2].copy()
    
    # Fold to get uniform distribution in unit tetrahedron
    mask1 = (s + t) > 1
    s[mask1] = 1 - s[mask1]
    t[mask1] = 1 - t[mask1]
    
    mask2 = (t + u) > 1
    mask3 = (~mask2) & ((s + t + u) > 1)
    
    # Handle t + u > 1 case
    old_t = t.copy()
    t[mask2] = 1 - u[mask2]
    u[mask2] = 1 - s[mask2] - old_t[mask2]
    
    # Handle s + t + u > 1 (but not t + u > 1) case
    old_s = s.copy()
    s[mask3] = 1 - t[mask3] - u[mask3]
    u[mask3] = old_s[mask3] + t[mask3] + u[mask3] - 1
    
    v = 1 - s - t - u
    
    return np.column_stack([s, t, u, v])


def generate_tetra_indices(grid_size: int, periodic: bool = True) -> np.ndarray:
    """
    Generate tetrahedron corner indices for a Lagrangian grid.
    
    Each Lagrangian cell is decomposed into 6 tetrahedra following the
    gotetra convention.
    
    Parameters
    ----------
    grid_size : int
        Number of particles per dimension
    periodic : bool
        If True, include all grid_size^3 cells with periodic wrapping.
        If False, only include (grid_size-1)^3 interior cells.
    
    Returns
    -------
    indices : ndarray, shape (n_tetra, 4)
        Indices into the flattened particle array for each tetrahedron's corners.
        Corner order follows gotetra: [DIRS[d][0], DIRS[d][1], (1,1,1), origin]
    """
    n_cells_per_dim = grid_size if periodic else grid_size - 1
    n_cells = n_cells_per_dim ** 3
    n_tetra = n_cells * 6
    
    indices = np.empty((n_tetra, 4), dtype=np.int64)
    
    def idx_3d_to_1d(x, y, z):
        # Apply periodic wrapping
        x = x % grid_size
        y = y % grid_size
        z = z % grid_size
        return x + y * grid_size + z * grid_size * grid_size
    
    tetra_i = 0
    for iz in range(n_cells_per_dim):
        for iy in range(n_cells_per_dim):
            for ix in range(n_cells_per_dim):
                for dir_idx in range(6):
                    d0, d1 = DIRS[dir_idx]
                    # Corner 0: origin + DIRS[dir][0]
                    indices[tetra_i, 0] = idx_3d_to_1d(ix + d0[0], iy + d0[1], iz + d0[2])
                    # Corner 1: origin + DIRS[dir][1]
                    indices[tetra_i, 1] = idx_3d_to_1d(ix + d1[0], iy + d1[1], iz + d1[2])
                    # Corner 2: origin + (1,1,1)
                    indices[tetra_i, 2] = idx_3d_to_1d(ix + 1, iy + 1, iz + 1)
                    # Corner 3: origin
                    indices[tetra_i, 3] = idx_3d_to_1d(ix, iy, iz)
                    tetra_i += 1
    
    return indices


def sort_particles_by_lagrangian_id(
    positions: np.ndarray, 
    particle_ids: np.ndarray, 
    grid_size: int
) -> np.ndarray:
    """
    Sort particles back to their original Lagrangian grid ordering.
    
    In GADGET simulations, particle IDs typically encode the original
    Lagrangian grid position. This function reorders particles so that
    particle i corresponds to Lagrangian position (ix, iy, iz) where:
        i = ix + iy * grid_size + iz * grid_size^2
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Current particle positions (Eulerian)
    particle_ids : ndarray, shape (N,)
        Particle IDs (encode Lagrangian positions)
    grid_size : int
        Number of particles per dimension (N for N^3 particles)
    
    Returns
    -------
    sorted_positions : ndarray, shape (N, 3)
        Positions sorted by Lagrangian ID
    """
    lag_indices = particle_ids.astype(np.int64)
    
    # Check if 1-indexed (IDs start from 1)
    if lag_indices.min() >= 1:
        lag_indices = lag_indices - 1
    
    expected_n = grid_size ** 3
    if len(positions) != expected_n:
        raise ValueError(
            f"Expected {expected_n} particles for grid_size={grid_size}, "
            f"but got {len(positions)}"
        )
    
    # Create output array and place particles in Lagrangian order
    sorted_positions = np.empty_like(positions)
    sorted_positions[lag_indices] = positions
    
    return sorted_positions


def barycentric_to_physical(
    corners: np.ndarray,
    bary_coords: np.ndarray,
    box_size: float
) -> np.ndarray:
    """
    Transform barycentric coordinates to physical positions within tetrahedra.
    
    This follows gotetra's distribute_tetra algorithm:
    1. Compute barycenter of tetrahedron
    2. Compute displacement vectors from barycenter to each corner
    3. Position = barycenter + s*d0 + t*d1 + u*d2 + v*d3
    
    Parameters
    ----------
    corners : ndarray, shape (n_tetra, 4, 3)
        Corner positions for each tetrahedron (already unwrapped for periodicity)
    bary_coords : ndarray, shape (n_samples, 4)
        Barycentric coordinates (s, t, u, v)
    box_size : float
        Box size for periodic boundary conditions
    
    Returns
    -------
    positions : ndarray, shape (n_tetra, n_samples, 3)
        Physical positions of samples within each tetrahedron
    """
    n_tetra = corners.shape[0]
    n_samples = bary_coords.shape[0]
    
    # Compute barycenter: average of 4 corners
    # Shape: (n_tetra, 3)
    barycenter = corners.mean(axis=1)
    
    # Compute displacement vectors from barycenter to each corner
    # Shape: (n_tetra, 4, 3)
    displacements = corners - barycenter[:, np.newaxis, :]
    
    # Compute positions: barycenter + sum_k(bary[k] * displacement[k])
    # Using einsum: 'sk,tkd->tsd' means:
    #   s = sample index, k = corner index (0-3), t = tetra index, d = dimension
    #   Sum over k, output indexed by (tetra, sample, dim)
    positions = barycenter[:, np.newaxis, :] + np.einsum('sk,tkd->tsd', bary_coords, displacements)
    
    # Apply periodic boundary conditions
    positions = positions % box_size
    
    return positions


def unwrap_corners(
    positions: np.ndarray,
    tetra_indices: np.ndarray,
    box_size: float
) -> np.ndarray:
    """
    Get tetrahedron corners and unwrap for periodic boundaries.
    
    Tetrahedra near box boundaries may have corners on opposite sides of the box.
    We unwrap relative to corner 0 to ensure correct geometry.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions in Lagrangian order
    tetra_indices : ndarray, shape (n_tetra, 4)
        Corner indices for each tetrahedron
    box_size : float
        Simulation box size
    
    Returns
    -------
    corners : ndarray, shape (n_tetra, 4, 3)
        Unwrapped corner positions
    """
    # Get corner positions
    # Shape: (n_tetra, 4, 3)
    corners = positions[tetra_indices]
    
    # Reference position is corner 0
    ref = corners[:, 0:1, :]  # Shape: (n_tetra, 1, 3)
    
    # Compute displacements from reference
    diff = corners - ref
    
    # Wrap displacements that are > box_size/2
    half_box = box_size / 2
    diff = np.where(diff > half_box, diff - box_size, diff)
    diff = np.where(diff < -half_box, diff + box_size, diff)
    
    # Unwrapped corners
    return ref + diff


def compute_tetra_density_field_python(
    positions: np.ndarray,
    box_size: float,
    grid_size: int,
    output_cells: int,
    n_samples: int = 100,
    particle_mass: float = 1.0,
    verbose: bool = True,
    batch_size: int = 10000,
    seed: int = 42
) -> np.ndarray:
    """
    Compute a 3D density field using tetrahedron tessellation (Python reference).
    
    This is the core gotetra algorithm implemented in Python for reference.
    For production use, prefer the C++ implementation.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions, MUST be sorted in Lagrangian order
        (use sort_particles_by_lagrangian_id first)
    box_size : float
        Size of the simulation box
    grid_size : int
        Original Lagrangian grid size (N for N^3 particles)
    output_cells : int
        Number of cells per dimension in output density grid
    n_samples : int
        Number of Monte Carlo samples per tetrahedron (default: 100)
    particle_mass : float
        Mass per particle in code units
    verbose : bool
        Print progress information
    batch_size : int
        Number of tetrahedra to process per batch (for memory efficiency)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    density : ndarray, shape (output_cells, output_cells, output_cells)
        3D density field (mass per unit volume)
    """
    n_particles = len(positions)
    expected_n = grid_size ** 3
    
    if n_particles != expected_n:
        raise ValueError(
            f"Expected {expected_n} particles for grid_size={grid_size}, "
            f"got {n_particles}"
        )
    
    positions = np.asarray(positions, dtype=np.float64)
    
    # Initialize density grid
    density = np.zeros((output_cells, output_cells, output_cells), dtype=np.float64)
    cell_width = box_size / output_cells
    
    # Mass per sample point
    # Each tetrahedron gets 1/6 of a particle's mass, spread over n_samples
    mass_per_sample = particle_mass / n_samples / 6.0
    
    # Generate barycentric sample points (Rocchini-Cignoni)
    if verbose:
        print(f"Generating {n_samples} Monte Carlo sample points...")
    rng = np.random.default_rng(seed)
    cube_samples = rng.random((n_samples, 3))
    bary_samples = rocchini_cignoni_transform(cube_samples)
    
    # Generate tetrahedron indices
    if verbose:
        print(f"Building tetrahedron connectivity for {grid_size}^3 grid...")
    tetra_indices = generate_tetra_indices(grid_size, periodic=True)
    n_tetra = len(tetra_indices)
    
    if verbose:
        print(f"Processing {n_tetra:,} tetrahedra...")
    
    # Process in batches for memory efficiency
    n_batches = (n_tetra + batch_size - 1) // batch_size
    
    for batch_i in range(n_batches):
        start_idx = batch_i * batch_size
        end_idx = min((batch_i + 1) * batch_size, n_tetra)
        batch_indices = tetra_indices[start_idx:end_idx]
        
        # Get and unwrap corner positions
        corners = unwrap_corners(positions, batch_indices, box_size)
        
        # Compute sample positions using barycentric coordinates
        sample_pos = barycentric_to_physical(corners, bary_samples, box_size)
        
        # Convert to grid indices
        grid_idx = (sample_pos / cell_width).astype(np.int32) % output_cells
        
        # Flatten and accumulate
        flat_idx = (grid_idx[:, :, 0].ravel() +
                    grid_idx[:, :, 1].ravel() * output_cells +
                    grid_idx[:, :, 2].ravel() * output_cells * output_cells)
        
        np.add.at(density.ravel(), flat_idx, mass_per_sample)
        
        if verbose and (batch_i + 1) % max(1, n_batches // 10) == 0:
            print(f"  Processed {end_idx:,}/{n_tetra:,} tetrahedra "
                  f"({100 * end_idx / n_tetra:.0f}%)")
    
    # Convert mass to density (mass per unit volume)
    cell_volume = cell_width ** 3
    density /= cell_volume
    
    if verbose:
        print(f"Done! Density field shape: {density.shape}")
        print(f"  Min: {density.min():.6e}")
        print(f"  Max: {density.max():.6e}")
        print(f"  Mean: {density.mean():.6e}")
    
    return density


def compute_tetra_density_3d(
    positions: np.ndarray,
    box_size: float,
    grid_size: int,
    output_cells: int,
    n_samples: int = 100,
    particle_mass: float = 1.0,
    n_threads: int = 0,
    verbose: bool = True,
    seed: int = 42,
    use_cpp: bool = True
) -> np.ndarray:
    """
    Compute a 3D density field using tetrahedron tessellation.
    
    Automatically uses the C++ implementation if available, falls back to Python.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions in Lagrangian order
    box_size : float
        Simulation box size
    grid_size : int
        Lagrangian grid size (N for N^3 particles)
    output_cells : int
        Output grid cells per dimension
    n_samples : int
        Monte Carlo samples per tetrahedron (default: 100)
    particle_mass : float
        Mass per particle
    n_threads : int
        Number of threads for C++ (0 = auto)
    verbose : bool
        Print progress
    seed : int
        Random seed
    use_cpp : bool
        Use C++ implementation if available (default: True)
    
    Returns
    -------
    density : ndarray, shape (output_cells, output_cells, output_cells)
        3D density field
    """
    if use_cpp and HAS_CPP:
        if verbose:
            print("Using C++ implementation (OpenMP parallelized)...")
        
        config = at.density.TetraDensityConfig(
            lagrangian_grid_size=grid_size,
            output_cells=output_cells,
            box_size=box_size,
            particle_mass=particle_mass,
            n_samples=n_samples,
            n_threads=n_threads,
            periodic=True,
            seed=seed
        )
        
        result = at.density.compute_tetra_density_3d(
            np.ascontiguousarray(positions, dtype=np.float64),
            config
        )
        
        if verbose:
            print(f"Done! Shape: {result.density.shape}")
            print(f"  Min: {result.density.min():.6e}")
            print(f"  Max: {result.density.max():.6e}")
            print(f"  Mean: {result.mean_density:.6e}")
            print(f"  Tetrahedra processed: {result.n_tetrahedra:,}")
        
        return np.array(result.density)
    else:
        if verbose and use_cpp:
            print("C++ implementation not available, using Python fallback...")
        return compute_tetra_density_field_python(
            positions, box_size, grid_size, output_cells,
            n_samples, particle_mass, verbose, seed=seed
        )


def project_density_slice(
    density_3d: np.ndarray,
    box_size: float,
    slice_center: float,
    slice_thickness: float,
    projection_axis: str = 'z'
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Extract and project a thin slice from a 3D density field.
    
    Parameters
    ----------
    density_3d : ndarray, shape (Nx, Ny, Nz)
        3D density field
    box_size : float
        Physical size of the box
    slice_center : float
        Center of the slice along projection axis
    slice_thickness : float
        Thickness of the slice
    projection_axis : str
        Axis to project along ('x', 'y', or 'z')
    
    Returns
    -------
    density_2d : ndarray
        2D projected density (surface density)
    extent : tuple
        (min1, max1, min2, max2) for plotting
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis = axis_map[projection_axis.lower()]
    
    n_cells = density_3d.shape[proj_axis]
    cell_width = box_size / n_cells
    
    # Find cell indices for the slice
    slice_min = slice_center - slice_thickness / 2
    slice_max = slice_center + slice_thickness / 2
    
    idx_min = max(0, int(slice_min / cell_width))
    idx_max = min(n_cells, int(np.ceil(slice_max / cell_width)))
    
    # Extract and sum along projection axis
    if proj_axis == 0:
        density_2d = density_3d[idx_min:idx_max, :, :].sum(axis=0)
    elif proj_axis == 1:
        density_2d = density_3d[:, idx_min:idx_max, :].sum(axis=1)
    else:
        density_2d = density_3d[:, :, idx_min:idx_max].sum(axis=2)
    
    # Multiply by cell width to get surface density
    density_2d *= cell_width
    
    extent = (0.0, box_size, 0.0, box_size)
    
    return density_2d, extent


def save_density_field(
    filename: str,
    density: np.ndarray,
    header_info: dict,
    grid_info: dict,
    is_3d: bool = True
) -> None:
    """
    Save density field to HDF5.
    """
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('density', data=density, compression='gzip')
        dset.attrs['units'] = 'mass / volume' if is_3d else 'mass / area'
        dset.attrs['method'] = 'tetrahedron_tessellation'
        
        hdr = f.create_group('Header')
        for key, val in header_info.items():
            hdr.attrs[key] = val
        
        grid = f.create_group('Grid')
        for key, val in grid_info.items():
            grid.attrs[key] = val
    
    print(f"Saved density field to {filename}")


def load_gadget4_for_tessellation(
    snapshot_path: str,
    snapshot_num: Optional[int] = None,
    particle_type: int = 1
) -> Tuple[np.ndarray, np.ndarray, "at.io.Gadget4Header", int]:
    """
    Load a GADGET-4 snapshot with particle IDs for tessellation.
    
    Parameters
    ----------
    snapshot_path : str
        Path to snapshot file or directory
    snapshot_num : int, optional
        Snapshot number for distributed files
    particle_type : int
        Particle type (default: 1 for DM)
    
    Returns
    -------
    positions : ndarray
        Particle positions (NOT yet sorted)
    particle_ids : ndarray
        Particle IDs for Lagrangian ordering
    header : Gadget4Header
        Snapshot header
    grid_size : int
        Inferred Lagrangian grid size
    """
    if not HAS_CPP:
        raise ImportError("C++ module required for GADGET I/O")
    
    particle_types_mask = 1 << particle_type
    
    if snapshot_num is not None:
        snap = at.io.read_gadget4_snapshot(
            snapshot_path,
            snapshot_num=snapshot_num,
            particle_types=particle_types_mask,
            read_velocities=False,
            read_ids=True
        )
    else:
        snap = at.io.read_gadget4_snapshot(
            snapshot_path,
            particle_types=particle_types_mask,
            read_velocities=False,
            read_ids=True
        )
    
    coords = snap.particles[particle_type].coordinates
    ids = snap.particles[particle_type].particle_ids
    
    positions = np.array(coords, dtype=np.float64)
    particle_ids = np.array(ids, dtype=np.int64)
    
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))
    
    if grid_size ** 3 != n_particles:
        raise ValueError(
            f"Particle count {n_particles} is not a perfect cube. "
            f"Nearest cube root: {grid_size}"
        )
    
    return positions, particle_ids, snap.header, grid_size


# =============================================================================
# Main script for command-line usage
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute tetrahedron-based density field from GADGET-4 snapshot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses the proper phase-space tessellation method from gotetra:
- Particles are connected into tetrahedra based on their Lagrangian grid
- Monte Carlo sampling within tetrahedra deposits mass onto the density grid
- This correctly handles stream crossing (caustics) in the density field

Examples:
  # Compute 128^3 density field using C++ (fast)
  python tetra_density.py snapshot_034.hdf5 -o density_3d.h5 --cells 128

  # Compute and project a slice
  python tetra_density.py snapshot.hdf5 -o density.h5 --cells 128 \\
      --slice-center 128 --slice-thickness 25 --plot slice.png

  # Use Python reference implementation
  python tetra_density.py snapshot.hdf5 -o density.h5 --cells 64 --python-only
        """
    )
    
    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 snapshot file or directory')
    parser.add_argument('-n', '--snapshot-num', type=int, default=None,
                        help='Snapshot number (for distributed files)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 filename')
    parser.add_argument('--cells', type=int, default=128,
                        help='Output grid cells per dimension (default: 128)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (default: 100)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (default: auto)')
    parser.add_argument('--python-only', action='store_true',
                        help='Force Python implementation (slower)')
    
    # Slice options
    parser.add_argument('--slice-center', type=float, default=None,
                        help='If set, extract a 2D slice at this center position')
    parser.add_argument('--slice-thickness', type=float, default=None,
                        help='Slice thickness (default: 10%% of box)')
    parser.add_argument('--axis', type=str, default='z', choices=['x', 'y', 'z'],
                        help='Projection axis for slice (default: z)')
    
    # Visualization
    parser.add_argument('--plot', type=str, default=None,
                        help='Output image filename (requires slice)')
    parser.add_argument('--cmap', type=str, default='magma',
                        help='Colormap for visualization')
    parser.add_argument('--overdensity', action='store_true',
                        help='Plot 1+delta instead of raw density')
    
    args = parser.parse_args()
    
    # Load snapshot
    print(f"Loading snapshot from {args.snapshot}...")
    positions, particle_ids, header, grid_size = load_gadget4_for_tessellation(
        args.snapshot,
        snapshot_num=args.snapshot_num,
        particle_type=args.particle_type
    )
    
    box_size = header.box_size
    particle_mass = header.mass_table[args.particle_type]
    if particle_mass == 0:
        particle_mass = 1.0
    
    print(f"Box size: {box_size}")
    print(f"Redshift: {header.redshift:.4f}")
    print(f"Grid size: {grid_size}^3 = {grid_size**3:,} particles")
    print(f"Particle mass: {particle_mass:.6e}")
    
    # Sort particles by Lagrangian ID
    print("\nSorting particles by Lagrangian ID...")
    if HAS_CPP and not args.python_only:
        sorted_positions = at.density.sort_by_lagrangian_id(
            positions, particle_ids, grid_size, id_offset=1
        )
    else:
        sorted_positions = sort_particles_by_lagrangian_id(positions, particle_ids, grid_size)
    
    # Compute 3D density field
    print(f"\nComputing tetrahedron-based density field ({args.cells}^3 cells)...")
    density_3d = compute_tetra_density_3d(
        sorted_positions,
        box_size,
        grid_size,
        args.cells,
        n_samples=args.samples,
        particle_mass=particle_mass,
        n_threads=args.threads,
        verbose=True,
        use_cpp=not args.python_only
    )
    
    # Prepare metadata
    header_info = {
        'BoxSize': box_size,
        'Redshift': header.redshift,
        'Time': header.time,
        'ParticleMass': particle_mass,
        'LagrangianGridSize': grid_size,
        'NumParticles': grid_size ** 3
    }
    
    grid_info = {
        'cells': args.cells,
        'cell_width': box_size / args.cells,
        'n_samples_per_tetra': args.samples
    }
    
    # Mean density for overdensity calculation
    total_mass = particle_mass * (grid_size ** 3)
    mean_density = total_mass / (box_size ** 3)
    header_info['MeanDensity'] = mean_density
    
    # Handle slice if requested
    if args.slice_center is not None:
        slice_thickness = args.slice_thickness or box_size * 0.1
        
        print(f"\nExtracting slice: center={args.slice_center}, "
              f"thickness={slice_thickness}, axis={args.axis}")
        
        density_2d, extent = project_density_slice(
            density_3d, box_size, args.slice_center, slice_thickness, args.axis
        )
        
        grid_info['slice_center'] = args.slice_center
        grid_info['slice_thickness'] = slice_thickness
        grid_info['projection_axis'] = args.axis
        grid_info['extent'] = extent
        
        mean_surface_density = mean_density * slice_thickness
        header_info['MeanSurfaceDensity'] = mean_surface_density
        
        print(f"2D slice shape: {density_2d.shape}")
        print(f"  Min: {density_2d.min():.6e}")
        print(f"  Max: {density_2d.max():.6e}")
        print(f"  Mean: {density_2d.mean():.6e}")
        
        save_density_field(args.output, density_2d, header_info, grid_info, is_3d=False)
        
        if args.plot:
            plot_density(density_2d, extent, header_info, grid_info,
                        args.plot, args.cmap, args.overdensity)
    else:
        save_density_field(args.output, density_3d, header_info, grid_info, is_3d=True)


def plot_density(density, extent, header_info, grid_info, output_file, cmap, overdensity):
    """Plot a 2D density field."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    plot_data = density.T
    
    if overdensity:
        mean_density = header_info.get('MeanSurfaceDensity', density.mean())
        plot_data = plot_data / mean_density
        label = r'$1 + \delta$'
    else:
        label = 'Surface Density'
    
    plot_data = np.maximum(plot_data, plot_data[plot_data > 0].min() * 0.1)
    vmin = np.percentile(plot_data[plot_data > 0], 1)
    vmax = np.percentile(plot_data, 99.9)
    
    im = ax.imshow(plot_data, extent=extent, origin='lower', cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest')
    
    ax.set_xlabel('Position [code units]', fontsize=12)
    ax.set_ylabel('Position [code units]', fontsize=12)
    ax.set_title(f"Tessellation Density (z={header_info['Redshift']:.2f})", fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{label} (log scale)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()


if __name__ == '__main__':
    main()
