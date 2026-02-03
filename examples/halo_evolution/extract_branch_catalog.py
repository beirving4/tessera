#!/usr/bin/env python3
"""
Extract Main Progenitor Branch Catalog

Pre-computes main progenitor branches for all trees in a GADGET-4 merger tree
file and saves them in a compact HDF5 format for fast random access.

This enables efficient SLURM array jobs where each task processes a subset
of trees without needing to load the full merger tree.

Output HDF5 structure:
    /Header (attrs)
        n_trees, n_total_halos, source_file, box_size, hubble, ...

    /BranchIndex/
        tree_id[n_trees]        - Tree identifier (sorted by root M200c)
        start_offset[n_trees]   - Start index in BranchData arrays
        length[n_trees]         - Number of halos in this branch
        root_snap[n_trees]      - Root halo snapshot number
        root_subhalo[n_trees]   - Root halo subhalo index
        root_m200c[n_trees]     - Root halo M200c for filtering
        root_r200c[n_trees]     - Root halo R200c
        a_min[n_trees]          - Earliest scale factor in branch
        a_max[n_trees]          - Latest scale factor in branch

    /BranchData/  (flattened arrays, ~24 bytes/halo after compression)
        snap_num[n_total]       - Snapshot numbers
        subhalo_nr[n_total]     - Subhalo indices
        scale_factor[n_total]   - Scale factors
        position[n_total, 3]    - Comoving positions [Mpc/h]
        velocity[n_total, 3]    - Velocities [km/s]
        m200c[n_total]          - M200 critical [10^10 Msun/h]
        r200c[n_total]          - R200 critical [Mpc/h]
        mass[n_total]           - Subhalo mass [10^10 Msun/h]

Usage:
    python extract_branch_catalog.py /path/to/trees.hdf5 -o branch_catalog.h5
    python extract_branch_catalog.py /path/to/trees.hdf5 --mass-min 10.0  # M200c > 10^11 Msun/h

Performance:
    When tessera C++ extension is available, uses OpenMP parallel extraction
    which is ~100-200x faster than pure Python.
"""

import argparse
from pathlib import Path
from typing import Optional

# Try to import tessera C++ extension FIRST to avoid HDF5 library conflicts
# (tessera uses HighFive, which must be loaded before h5py in some environments)
try:
    import tessera as ts
    HAS_TESSERA = hasattr(ts.io, 'extract_all_branches_parallel')
except ImportError:
    HAS_TESSERA = False

import numpy as np
import h5py
from tqdm import tqdm

# Handle imports for both package and standalone usage
try:
    from .merger_tree import MergerTree
    from .halo_tracker import HaloTracker
except ImportError:
    from merger_tree import MergerTree
    from halo_tracker import HaloTracker


def extract_branch_catalog_cpp(
    tree_file: Path,
    output_file: Path,
    mass_min: Optional[float] = None,
    a_min: Optional[float] = None,
    n_threads: int = 0,
    chunk_size: int = 0,
    show_progress: bool = True,
) -> dict:
    """
    Extract branches using C++ OpenMP parallel implementation.

    This is ~100-200x faster than pure Python for large tree files.

    Parameters
    ----------
    chunk_size : int
        Maximum halos to load at once (0 = load all, >0 = low-memory streaming mode)
    """
    import time

    print(f"Loading merger tree from {tree_file}...")
    tree = ts.io.MergerTree(str(tree_file))
    print(f"  Found {tree.n_trees():,} trees, {tree.n_halos():,} total halos")

    mode_str = "streaming" if chunk_size > 0 else "parallel"
    if show_progress:
        print(f"Extracting branches (C++ {mode_str})...", end=" ", flush=True)
    start_time = time.time()

    # Extract in parallel (no progress callback - runs too fast to need it)
    data = ts.io.extract_all_branches_parallel(
        tree,
        mass_min=mass_min,
        a_min=a_min,
        n_threads=n_threads,
        chunk_size=chunk_size,
    )

    elapsed = time.time() - start_time
    if show_progress:
        print(f"done in {elapsed:.2f}s")

    n_trees = data.n_branches()
    n_total_halos = data.n_halos()
    print(f"\nCollected {n_total_halos:,} halos from {n_trees:,} branches")

    if n_trees == 0:
        print("No trees match the filter criteria.")
        return {'n_trees': 0, 'n_halos': 0}

    # Build position/velocity arrays for HDF5 (combine x,y,z)
    position = np.column_stack([data.pos_x, data.pos_y, data.pos_z]).astype(np.float32)
    velocity = np.column_stack([data.vel_x, data.vel_y, data.vel_z]).astype(np.float32)

    # Write output file
    print(f"Writing to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Header
        header = f.create_group('Header')
        header.attrs['n_trees'] = n_trees
        header.attrs['n_total_halos'] = n_total_halos
        header.attrs['source_file'] = str(tree_file)
        header.attrs['box_size'] = tree.box_size()
        header.attrs['hubble'] = tree.header().hubble
        header.attrs['omega_matter'] = tree.header().omega_matter
        header.attrs['omega_lambda'] = tree.header().omega_lambda
        header.attrs['n_snapshots'] = tree.n_snapshots()
        header.attrs['last_snap'] = tree.header().last_snap
        header.attrs['extraction_method'] = 'cpp_parallel'
        if mass_min is not None:
            header.attrs['mass_min_filter'] = mass_min
        if a_min is not None:
            header.attrs['a_min_filter'] = a_min

        # Store snapshot time mapping
        f.create_dataset('TreeTimes/Time', data=np.array(tree.header().snap_times))
        f.create_dataset('TreeTimes/Redshift', data=np.array(tree.header().snap_redshifts))

        # Branch index
        idx = f.create_group('BranchIndex')
        idx.create_dataset('tree_id', data=np.array(data.tree_id, dtype=np.int32))
        idx.create_dataset('start_offset', data=np.array(data.start_offset, dtype=np.int64))
        idx.create_dataset('length', data=np.array(data.length, dtype=np.int32))
        idx.create_dataset('root_snap', data=np.array(data.root_snap, dtype=np.int32))
        idx.create_dataset('root_subhalo', data=np.array(data.root_subhalo, dtype=np.int32))
        idx.create_dataset('root_m200c', data=np.array(data.root_m200c, dtype=np.float32))
        idx.create_dataset('root_r200c', data=np.array(data.root_r200c, dtype=np.float32))
        idx.create_dataset('a_min', data=np.array(data.a_min, dtype=np.float32))
        idx.create_dataset('a_max', data=np.array(data.a_max, dtype=np.float32))

        # Branch data (compressed)
        bd = f.create_group('BranchData')
        bd.create_dataset('snap_num', data=np.array(data.snap_num, dtype=np.int32), compression='gzip')
        bd.create_dataset('subhalo_nr', data=np.array(data.subhalo_nr, dtype=np.int32), compression='gzip')
        bd.create_dataset('scale_factor', data=np.array(data.scale_factor, dtype=np.float32), compression='gzip')
        bd.create_dataset('position', data=position, compression='gzip')
        bd.create_dataset('velocity', data=velocity, compression='gzip')
        bd.create_dataset('m200c', data=np.array(data.m200c, dtype=np.float32), compression='gzip')
        bd.create_dataset('r200c', data=np.array(data.r200c, dtype=np.float32), compression='gzip')
        bd.create_dataset('mass', data=np.array(data.mass, dtype=np.float32), compression='gzip')

    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Bytes per halo: {output_file.stat().st_size / max(1, n_total_halos):.1f}")

    return {
        'n_trees': n_trees,
        'n_halos': n_total_halos,
        'file_size_mb': file_size_mb,
        'source_file': str(tree_file),
    }


def extract_branch_catalog(
    tree_file: Path,
    output_file: Path,
    mass_min: Optional[float] = None,
    a_min: Optional[float] = None,
    n_threads: int = 0,
    chunk_size: int = 0,
    show_progress: bool = True,
) -> dict:
    """
    Extract main progenitor branches for all trees.

    When tessera C++ extension is available, uses OpenMP parallel extraction
    which is ~100-200x faster than pure Python.

    Parameters
    ----------
    tree_file : Path
        Path to GADGET-4 trees.hdf5 file (or directory for distributed files)
    output_file : Path
        Output HDF5 file path
    mass_min : float, optional
        Minimum root halo M200c in code units (10^10 Msun/h).
        E.g., mass_min=10.0 corresponds to M200c > 10^11 Msun/h
    a_min : float, optional
        Minimum scale factor to trace branches to (default: trace all)
    n_threads : int, optional
        Number of threads for C++ extraction (0 = auto, ignored for Python)
    chunk_size : int, optional
        Maximum halos to load at once for low-memory mode.
        0 = load all (default, fastest), >0 = streaming mode.
        Recommended: 10_000_000 to 50_000_000 for large files (60-100 GB).
    show_progress : bool
        Show progress bar

    Returns
    -------
    stats : dict
        Extraction statistics (n_trees, n_halos, etc.)
    """
    # Use C++ parallel extraction if available (much faster)
    if HAS_TESSERA:
        return extract_branch_catalog_cpp(
            tree_file, output_file,
            mass_min=mass_min, a_min=a_min,
            n_threads=n_threads, chunk_size=chunk_size,
            show_progress=show_progress
        )

    # Python fallback (slower, but works without C++ extension)
    print("Note: Using Python extraction (tessera C++ not available).")
    print("      Install tessera for ~100-200x faster extraction.")
    print()

    # Load merger tree
    print(f"Loading merger tree from {tree_file}...")
    tree = MergerTree(tree_file)
    tracker = HaloTracker(tree)

    print(f"  Found {tree.n_trees:,} trees, {tree.n_halos:,} total halos")
    print(f"  Scale factor range: [{tree.snap_times.min():.4f}, {tree.snap_times.max():.4f}]")

    # Get root halo M200c values for filtering and sorting
    root_indices = tree.tree_table['StartOffset']
    root_m200c = tree.tree_halos['Group_M_Crit200'][root_indices]
    root_r200c = tree.tree_halos['Group_R_Crit200'][root_indices]
    root_snap = tree.tree_halos['SnapNum'][root_indices]
    root_subhalo = tree.tree_halos['SubhaloNr'][root_indices]

    # Sort trees by root M200c (descending) for convenient mass-ordered access
    sort_order = np.argsort(-root_m200c)

    # Apply mass filter if specified
    if mass_min is not None:
        mass_mask = root_m200c[sort_order] >= mass_min
        sort_order = sort_order[mass_mask]
        print(f"  After mass filter (M200c >= {mass_min:.2g}): {len(sort_order):,} trees")

    n_trees_to_extract = len(sort_order)

    if n_trees_to_extract == 0:
        print("No trees match the filter criteria.")
        return {'n_trees': 0, 'n_halos': 0}

    # Pre-allocate index arrays
    idx_tree_id = np.zeros(n_trees_to_extract, dtype=np.int32)
    idx_start_offset = np.zeros(n_trees_to_extract, dtype=np.int64)
    idx_length = np.zeros(n_trees_to_extract, dtype=np.int32)
    idx_root_snap = np.zeros(n_trees_to_extract, dtype=np.int32)
    idx_root_subhalo = np.zeros(n_trees_to_extract, dtype=np.int32)
    idx_root_m200c = np.zeros(n_trees_to_extract, dtype=np.float32)
    idx_root_r200c = np.zeros(n_trees_to_extract, dtype=np.float32)
    idx_a_min = np.zeros(n_trees_to_extract, dtype=np.float32)
    idx_a_max = np.zeros(n_trees_to_extract, dtype=np.float32)

    # Collect branch data in lists (will concatenate at end)
    all_snap_num = []
    all_subhalo_nr = []
    all_scale_factor = []
    all_position = []
    all_velocity = []
    all_m200c = []
    all_r200c = []
    all_mass = []

    current_offset = 0

    # Extract branches
    iterator = tqdm(enumerate(sort_order), total=n_trees_to_extract,
                    desc="Extracting branches", disable=not show_progress)

    for i, tree_id in iterator:
        # Get root halo info
        r_snap = int(root_snap[tree_id])
        r_subhalo = int(root_subhalo[tree_id])

        # Trace main branch backward from root
        branch = tracker.trace_main_branch(
            snap_num=r_snap,
            subhalo_nr=r_subhalo,
            backward=True,
            forward=False,  # Root is already at final snapshot
            a_min=a_min,
        )

        if not branch:
            # Empty branch - still record in index
            idx_tree_id[i] = tree_id
            idx_start_offset[i] = current_offset
            idx_length[i] = 0
            idx_root_snap[i] = r_snap
            idx_root_subhalo[i] = r_subhalo
            idx_root_m200c[i] = root_m200c[tree_id]
            idx_root_r200c[i] = root_r200c[tree_id]
            idx_a_min[i] = 0.0
            idx_a_max[i] = 0.0
            continue

        # Store index info
        branch_len = len(branch)
        idx_tree_id[i] = tree_id
        idx_start_offset[i] = current_offset
        idx_length[i] = branch_len
        idx_root_snap[i] = r_snap
        idx_root_subhalo[i] = r_subhalo
        idx_root_m200c[i] = root_m200c[tree_id]
        idx_root_r200c[i] = root_r200c[tree_id]
        idx_a_min[i] = branch[0].scale_factor  # Branch is sorted early to late
        idx_a_max[i] = branch[-1].scale_factor

        # Store branch data
        for halo in branch:
            all_snap_num.append(halo.snap_num)
            all_subhalo_nr.append(halo.subhalo_nr)
            all_scale_factor.append(halo.scale_factor)
            all_position.append(halo.position)
            all_velocity.append(halo.velocity)
            all_m200c.append(halo.m200c)
            all_r200c.append(halo.r200c)
            all_mass.append(halo.mass)

        current_offset += branch_len

    # Convert lists to arrays
    n_total_halos = len(all_snap_num)
    print(f"\nCollected {n_total_halos:,} halos from {n_trees_to_extract:,} branches")

    data_snap_num = np.array(all_snap_num, dtype=np.int32)
    data_subhalo_nr = np.array(all_subhalo_nr, dtype=np.int32)
    data_scale_factor = np.array(all_scale_factor, dtype=np.float32)
    data_position = np.array(all_position, dtype=np.float32)
    data_velocity = np.array(all_velocity, dtype=np.float32)
    data_m200c = np.array(all_m200c, dtype=np.float32)
    data_r200c = np.array(all_r200c, dtype=np.float32)
    data_mass = np.array(all_mass, dtype=np.float32)

    # Write output file
    print(f"Writing to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Header
        header = f.create_group('Header')
        header.attrs['n_trees'] = n_trees_to_extract
        header.attrs['n_total_halos'] = n_total_halos
        header.attrs['source_file'] = str(tree_file)
        header.attrs['box_size'] = tree.box_size
        header.attrs['hubble'] = tree.hubble
        header.attrs['omega_matter'] = tree.omega_matter
        header.attrs['omega_lambda'] = tree.omega_lambda
        header.attrs['n_snapshots'] = tree.n_snapshots
        header.attrs['last_snap'] = tree.last_snap
        if mass_min is not None:
            header.attrs['mass_min_filter'] = mass_min
        if a_min is not None:
            header.attrs['a_min_filter'] = a_min

        # Store snapshot time mapping
        f.create_dataset('TreeTimes/Time', data=tree.snap_times)
        f.create_dataset('TreeTimes/Redshift', data=tree.snap_redshifts)

        # Branch index (small, always loaded)
        idx = f.create_group('BranchIndex')
        idx.create_dataset('tree_id', data=idx_tree_id)
        idx.create_dataset('start_offset', data=idx_start_offset)
        idx.create_dataset('length', data=idx_length)
        idx.create_dataset('root_snap', data=idx_root_snap)
        idx.create_dataset('root_subhalo', data=idx_root_subhalo)
        idx.create_dataset('root_m200c', data=idx_root_m200c)
        idx.create_dataset('root_r200c', data=idx_root_r200c)
        idx.create_dataset('a_min', data=idx_a_min)
        idx.create_dataset('a_max', data=idx_a_max)

        # Branch data (compressed, loaded on demand)
        data = f.create_group('BranchData')
        data.create_dataset('snap_num', data=data_snap_num, compression='gzip')
        data.create_dataset('subhalo_nr', data=data_subhalo_nr, compression='gzip')
        data.create_dataset('scale_factor', data=data_scale_factor, compression='gzip')
        data.create_dataset('position', data=data_position, compression='gzip')
        data.create_dataset('velocity', data=data_velocity, compression='gzip')
        data.create_dataset('m200c', data=data_m200c, compression='gzip')
        data.create_dataset('r200c', data=data_r200c, compression='gzip')
        data.create_dataset('mass', data=data_mass, compression='gzip')

    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Bytes per halo: {output_file.stat().st_size / max(1, n_total_halos):.1f}")

    return {
        'n_trees': n_trees_to_extract,
        'n_halos': n_total_halos,
        'file_size_mb': file_size_mb,
        'source_file': str(tree_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract main progenitor branches from GADGET-4 merger trees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'tree_file', type=Path,
        help="Input GADGET-4 trees.hdf5 file"
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help="Output HDF5 file (default: branch_catalog.h5 in same directory)"
    )
    parser.add_argument(
        '--mass-min', type=float, default=None,
        help="Minimum root M200c in code units [10^10 Msun/h]"
    )
    parser.add_argument(
        '--a-min', type=float, default=None,
        help="Minimum scale factor to trace to"
    )
    parser.add_argument(
        '--no-progress', action='store_true',
        help="Disable progress bar"
    )
    parser.add_argument(
        '-j', '--threads', type=int, default=0,
        help="Number of threads for C++ extraction (0 = auto)"
    )
    parser.add_argument(
        '--chunk-size', type=int, default=0,
        help="Low-memory mode: max halos to load at once (0 = load all, fastest). "
             "Use 10000000-50000000 for 60-100 GB tree files."
    )
    parser.add_argument(
        '--low-memory', action='store_true',
        help="Enable low-memory streaming mode (equivalent to --chunk-size 20000000)"
    )

    args = parser.parse_args()

    # Handle --low-memory flag
    chunk_size = args.chunk_size
    if args.low_memory and chunk_size == 0:
        chunk_size = 20_000_000  # Default chunk size for low-memory mode

    if not args.tree_file.exists():
        raise FileNotFoundError(f"Tree file not found: {args.tree_file}")

    output_file = args.output
    if output_file is None:
        output_file = args.tree_file.parent / 'branch_catalog.h5'

    stats = extract_branch_catalog(
        tree_file=args.tree_file,
        output_file=output_file,
        mass_min=args.mass_min,
        a_min=args.a_min,
        n_threads=args.threads,
        chunk_size=chunk_size,
        show_progress=not args.no_progress,
    )

    print(f"\nExtraction complete!")
    print(f"  Trees: {stats['n_trees']:,}")
    print(f"  Halos: {stats['n_halos']:,}")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
