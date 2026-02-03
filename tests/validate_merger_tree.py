#!/usr/bin/env python3
"""
Validate tessera C++ MergerTree implementation against raw HDF5 data.

This script validates that tessera correctly:
1. Reads tree and halo counts from headers
2. Loads TreeTable entries (StartOffset, Length)
3. Walks TreeMainProgenitor chains
4. Handles both single files and split files (trees.N.hdf5)

Usage
-----
# Run all validations with default test paths
python validate_merger_tree.py

# Run with custom paths
python validate_merger_tree.py \
    --single-file "/path/to/trees.hdf5" \
    --split-files "/path/to/treedata" \
    --large-file "/path/to/large/trees.hdf5"

# Run specific validation only
python validate_merger_tree.py --single-file "/path/to/trees.hdf5"
"""

import argparse
from pathlib import Path
import random
import sys

import h5py
import numpy as np

try:
    import tessera as ts
except ImportError:
    print("ERROR: tessera not installed. Install with: pip install -e /path/to/tessera")
    sys.exit(1)


def validate_single_file(tree_path: str, n_trees_to_test: int = 50) -> bool:
    """
    Validate a single HDF5 tree file.

    Parameters
    ----------
    tree_path : str
        Path to trees.hdf5
    n_trees_to_test : int
        Number of random trees to validate

    Returns
    -------
    bool
        True if all validations pass
    """
    print(f"Loading tessera: {tree_path}")
    tree = ts.io.MergerTree(tree_path)
    tracker = ts.io.HaloTracker(tree)
    print(f"  Loaded: {tree.n_trees()} trees, {tree.n_halos()} halos")

    # Load HDF5 data
    print("Loading HDF5 data...")
    with h5py.File(tree_path, 'r') as f:
        hdf5_n_trees = int(f['Header'].attrs.get('Ntrees_Total',
                          f['Header'].attrs.get('TotNtrees', 0)))
        hdf5_n_halos = int(f['Header'].attrs.get('Nhalos_Total',
                          f['Header'].attrs.get('TotNhalos', 0)))
        snap_num = f['TreeHalos/SnapNum'][:]
        subhalo_nr = f['TreeHalos/SubhaloNr'][:]
        group_nr = f['TreeHalos/GroupNr'][:]
        main_prog = f['TreeHalos/TreeMainProgenitor'][:]
        starts = f['TreeTable/StartOffset'][:]
        lengths = f['TreeTable/Length'][:]

    print(f"  HDF5: {hdf5_n_trees} trees, {hdf5_n_halos} halos")

    # Test 1: Counts match
    counts_ok = (tree.n_trees() == hdf5_n_trees and
                 tree.n_halos() == hdf5_n_halos)
    print(f"Counts match: {'PASSED' if counts_ok else 'FAILED'}")

    if not counts_ok:
        return False

    # Test 2: Branch validation
    def walk_hdf5_chain(tree_id):
        tree_start = int(starts[tree_id])
        tree_length = int(lengths[tree_id])
        tree_idx = 0
        chain = []
        while tree_idx >= 0 and tree_idx < tree_length:
            global_idx = tree_start + tree_idx
            chain.append((int(snap_num[global_idx]),
                         int(subhalo_nr[global_idx]),
                         int(group_nr[global_idx])))
            next_prog = int(main_prog[global_idx])
            if next_prog < 0:
                break
            tree_idx = next_prog
        return chain

    random.seed(42)
    test_ids = random.sample(range(tree.n_trees()),
                            min(n_trees_to_test, tree.n_trees()))

    n_passed = 0
    for tree_id in test_ids:
        hdf5_chain = walk_hdf5_chain(tree_id)
        tess_branch = tracker.trace_from_tree_id(tree_id, backward=True, forward=False)

        # Compare as sets (tessera sorts by scale factor, HDF5 is tree order)
        hdf5_set = set(hdf5_chain)
        tess_set = set((h.snap_num, h.subhalo_nr, h.group_nr) for h in tess_branch)

        if hdf5_set == tess_set:
            n_passed += 1

    print(f"Branch validation: {n_passed}/{len(test_ids)} passed")

    return counts_ok and n_passed == len(test_ids)


def validate_split_files(tree_dir: str, n_trees_to_test: int = 50) -> bool:
    """
    Validate split HDF5 tree files (trees.0.hdf5, trees.1.hdf5, ...).

    Parameters
    ----------
    tree_dir : str
        Path to directory containing trees.N.hdf5 files, or path to trees.0.hdf5
    n_trees_to_test : int
        Number of random trees to validate

    Returns
    -------
    bool
        True if all validations pass
    """
    from glob import glob

    # Find all split files
    tree_path = Path(tree_dir)
    if tree_path.is_file():
        directory = tree_path.parent
    else:
        directory = tree_path

    files = sorted(glob(str(directory / "trees.*.hdf5")),
                  key=lambda x: int(x.split('.')[-2]))

    if not files:
        print(f"ERROR: No tree files found in {directory}")
        return False

    print(f"Loading tessera: {files[0]}")
    tree = ts.io.MergerTree(files[0])
    tracker = ts.io.HaloTracker(tree)
    print(f"  Loaded: {tree.n_trees()} trees, {tree.n_halos()} halos")
    print(f"  Found {len(files)} split files")

    # Load header from first file
    with h5py.File(files[0], 'r') as f:
        hdf5_n_trees = int(f['Header'].attrs.get('Ntrees_Total',
                          f['Header'].attrs.get('TotNtrees', 0)))
        hdf5_n_halos = int(f['Header'].attrs.get('Nhalos_Total',
                          f['Header'].attrs.get('TotNhalos', 0)))

    print(f"  HDF5: {hdf5_n_trees} trees, {hdf5_n_halos} halos")

    # Test 1: Counts match
    counts_ok = (tree.n_trees() == hdf5_n_trees and
                 tree.n_halos() == hdf5_n_halos)
    print(f"Counts match: {'PASSED' if counts_ok else 'FAILED'}")

    if not counts_ok:
        return False

    # Build file offset map for reading split data
    file_offsets = []
    cumulative = 0
    for f in files:
        with h5py.File(f, 'r') as hf:
            n = len(hf['TreeHalos/SnapNum'])
            file_offsets.append((cumulative, cumulative + n, f))
            cumulative += n

    def get_halo_data(global_idx):
        for start, end, filepath in file_offsets:
            if start <= global_idx < end:
                local_idx = global_idx - start
                with h5py.File(filepath, 'r') as hf:
                    return (int(hf['TreeHalos/SnapNum'][local_idx]),
                            int(hf['TreeHalos/SubhaloNr'][local_idx]),
                            int(hf['TreeHalos/GroupNr'][local_idx]),
                            int(hf['TreeHalos/TreeMainProgenitor'][local_idx]))
        raise ValueError(f'Index {global_idx} not found in any file')

    def walk_hdf5_chain(tree_id):
        entry = tree.get_tree_entry(tree_id)
        tree_start = entry.start_offset
        tree_length = entry.length
        tree_idx = 0
        chain = []
        while tree_idx >= 0 and tree_idx < tree_length:
            global_idx = tree_start + tree_idx
            snap, sub, grp, next_prog = get_halo_data(global_idx)
            chain.append((snap, sub, grp))
            if next_prog < 0:
                break
            tree_idx = next_prog
        return chain

    # Test 2: Branch validation
    random.seed(42)
    test_ids = random.sample(range(tree.n_trees()),
                            min(n_trees_to_test, tree.n_trees()))

    n_passed = 0
    for tree_id in test_ids:
        hdf5_chain = walk_hdf5_chain(tree_id)
        tess_branch = tracker.trace_from_tree_id(tree_id, backward=True, forward=False)

        hdf5_set = set(hdf5_chain)
        tess_set = set((h.snap_num, h.subhalo_nr, h.group_nr) for h in tess_branch)

        if hdf5_set == tess_set:
            n_passed += 1

    print(f"Branch validation: {n_passed}/{len(test_ids)} passed")

    return counts_ok and n_passed == len(test_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Validate tessera MergerTree implementation against HDF5 data"
    )
    parser.add_argument('--single-file', type=str,
                       help='Path to single trees.hdf5 file')
    parser.add_argument('--split-files', type=str,
                       help='Path to directory with split tree files or trees.0.hdf5')
    parser.add_argument('--large-file', type=str,
                       help='Path to large trees.hdf5 file')
    parser.add_argument('--n-trees', type=int, default=50,
                       help='Number of trees to validate per file (default: 50)')

    args = parser.parse_args()

    # Default test paths (update these for your system)
    default_single = "/Users/bryen/Documents/Physics Research/Stanford/ytree_data/gadget4/trees/trees.hdf5"
    default_split = "/Users/bryen/Documents/Physics Research/Stanford/ytree_data/gadget4/treedata"
    default_large = "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/trees.hdf5"

    # Determine which tests to run
    run_single = args.single_file or (not args.split_files and not args.large_file)
    run_split = args.split_files or (not args.single_file and not args.large_file)
    run_large = args.large_file or (not args.single_file and not args.split_files)

    # If no args given, run all with defaults
    if not args.single_file and not args.split_files and not args.large_file:
        run_single = run_split = run_large = True

    results = {}

    # Validation 1: Single file
    if run_single:
        path = args.single_file or default_single
        if Path(path).exists():
            print("=" * 70)
            print("VALIDATION 1: Single Tree File")
            print("=" * 70)
            results['single'] = validate_single_file(path, args.n_trees)
            print(f"VALIDATION 1: {'PASSED' if results['single'] else 'FAILED'}")
            print()
        else:
            print(f"Skipping single file validation: {path} not found")

    # Validation 2: Split files
    if run_split:
        path = args.split_files or default_split
        check_path = Path(path)
        if check_path.exists() or (check_path.parent.exists() and
                                    list(check_path.parent.glob("trees.*.hdf5"))):
            print("=" * 70)
            print("VALIDATION 2: Split Tree Files")
            print("=" * 70)
            results['split'] = validate_split_files(path, args.n_trees)
            print(f"VALIDATION 2: {'PASSED' if results['split'] else 'FAILED'}")
            print()
        else:
            print(f"Skipping split file validation: {path} not found")

    # Validation 3: Large file
    if run_large:
        path = args.large_file or default_large
        if Path(path).exists():
            print("=" * 70)
            print("VALIDATION 3: Large Tree File")
            print("=" * 70)
            results['large'] = validate_single_file(path, args.n_trees)
            print(f"VALIDATION 3: {'PASSED' if results['large'] else 'FAILED'}")
            print()
        else:
            print(f"Skipping large file validation: {path} not found")

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed and results:
        print("\nAll validations PASSED")
        return 0
    elif not results:
        print("\nNo validations were run (files not found)")
        return 1
    else:
        print("\nSome validations FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
