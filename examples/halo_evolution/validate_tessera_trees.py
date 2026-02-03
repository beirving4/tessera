#!/usr/bin/env python3
"""
Validate tessera C++ MergerTree implementation against raw HDF5 data.

This script compares:
1. Number of trees and halos
2. Tree sizes (from TreeTable)
3. Main progenitor chain walking (using TreeMainProgenitor)

Properly handles split tree files by concatenating all data.

Usage
-----
python validate_tessera_trees.py --tree-path "/path/to/trees.hdf5"
python validate_tessera_trees.py --tree-path "/path/to/treedata"  # split files
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import h5py
from glob import glob


def load_hdf5_trees(tree_path: Path):
    """
    Load all tree data from HDF5 file(s), handling split files.

    Returns
    -------
    dict with keys:
        - n_trees: Total number of trees
        - n_halos: Total number of halos
        - tree_table: dict with 'StartOffset', 'Length', 'TreeID'
        - halos: dict with concatenated halo arrays
    """
    tree_path = Path(tree_path)

    # Find all tree files
    if tree_path.is_file():
        files = [tree_path]
    else:
        # Directory with split files
        pattern = str(tree_path / "trees.*.hdf5")
        files = sorted(glob(pattern), key=lambda x: int(x.split('.')[-2]))
        if not files:
            # Try without directory
            files = sorted(glob(str(tree_path) + "/trees.*.hdf5"),
                          key=lambda x: int(x.split('.')[-2]))

    if not files:
        raise ValueError(f"No tree files found at {tree_path}")

    print(f"Found {len(files)} tree file(s)")

    # Read header from first file
    with h5py.File(files[0], 'r') as f:
        header = dict(f['Header'].attrs)
        n_trees = int(header.get('Ntrees_Total', header.get('TotNtrees', 0)))
        n_halos = int(header.get('Nhalos_Total', header.get('TotNhalos', 0)))
        num_files = int(header.get('NumFiles', 1))

    print(f"Header: {n_trees} trees, {n_halos} halos, {num_files} files")

    # Concatenate TreeTable from all files
    tree_table = {
        'StartOffset': [],
        'Length': [],
        'TreeID': []
    }

    # Concatenate halos from all files
    halos = {
        'SnapNum': [],
        'SubhaloNr': [],
        'GroupNr': [],
        'TreeMainProgenitor': [],
        'TreeDescendant': [],
        'TreeFirstProgenitor': [],
        'TreeNextProgenitor': [],
    }

    total_halos_read = 0
    total_trees_read = 0

    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            # Read TreeTable if present
            if 'TreeTable' in f:
                tt = f['TreeTable']
                if 'StartOffset' in tt:
                    tree_table['StartOffset'].append(tt['StartOffset'][:])
                    tree_table['Length'].append(tt['Length'][:])
                    if 'TreeID' in tt:
                        tree_table['TreeID'].append(tt['TreeID'][:])
                    total_trees_read += len(tt['StartOffset'])

            # Read TreeHalos
            if 'TreeHalos' in f:
                th = f['TreeHalos']
                n_in_file = len(th['SnapNum'])
                total_halos_read += n_in_file

                for key in halos:
                    if key in th:
                        halos[key].append(th[key][:])

    print(f"Read {total_trees_read} tree table entries, {total_halos_read} halos")

    # Concatenate arrays
    for key in tree_table:
        if tree_table[key]:
            tree_table[key] = np.concatenate(tree_table[key])
        else:
            tree_table[key] = np.array([], dtype=np.int64)

    for key in halos:
        if halos[key]:
            halos[key] = np.concatenate(halos[key])
        else:
            halos[key] = np.array([], dtype=np.int32)

    return {
        'n_trees': n_trees,
        'n_halos': n_halos,
        'tree_table': tree_table,
        'halos': halos,
    }


def walk_main_progenitor_hdf5(halos: dict, start_offset: int, length: int):
    """
    Walk the main progenitor chain using raw HDF5 data.

    TreeMainProgenitor values are relative to tree start (0-indexed within tree).
    -1 means no progenitor.

    Returns list of (snap_num, subhalo_nr, group_nr) tuples.
    """
    chain = []

    # Start at root (index 0 in tree = start_offset in global array)
    tree_idx = 0
    global_idx = start_offset + tree_idx

    while tree_idx >= 0 and tree_idx < length:
        snap = int(halos['SnapNum'][global_idx])
        subhalo = int(halos['SubhaloNr'][global_idx])
        group = int(halos['GroupNr'][global_idx])
        chain.append((snap, subhalo, group))

        # Get next main progenitor (relative to tree start)
        next_tree_idx = int(halos['TreeMainProgenitor'][global_idx])

        if next_tree_idx < 0:
            break

        tree_idx = next_tree_idx
        global_idx = start_offset + tree_idx

    return chain


def validate_tessera_vs_hdf5(tree_path: Path, n_trees_to_test: int = 100, verbose: bool = False):
    """
    Validate tessera MergerTree implementation against raw HDF5 data.
    """
    print("=" * 70)
    print("TESSERA MERGERTREE VALIDATION")
    print("=" * 70)
    print()

    # Load tessera
    print("Loading tessera...")
    try:
        import tessera as ts
    except ImportError:
        print("ERROR: tessera not installed")
        return False

    # Load tree with tessera
    tree_path = Path(tree_path)
    if tree_path.is_dir():
        # Split files - find first file
        pattern = str(tree_path / "trees.0.hdf5")
        tessera_tree = ts.io.MergerTree(pattern)
    else:
        tessera_tree = ts.io.MergerTree(str(tree_path))

    print(f"  tessera: {tessera_tree.n_trees()} trees, {tessera_tree.n_halos()} halos")
    print()

    # Load HDF5 data
    print("Loading HDF5 data...")
    hdf5_data = load_hdf5_trees(tree_path)
    print(f"  HDF5: {hdf5_data['n_trees']} trees, {hdf5_data['n_halos']} halos")
    print()

    all_passed = True

    # Test 1: Tree counts
    print("=== Test 1: Tree and halo counts ===")
    if tessera_tree.n_trees() != hdf5_data['n_trees']:
        print(f"  FAILED: Tree count mismatch: {tessera_tree.n_trees()} vs {hdf5_data['n_trees']}")
        all_passed = False
    else:
        print(f"  PASSED: Tree counts match ({tessera_tree.n_trees()})")

    if tessera_tree.n_halos() != hdf5_data['n_halos']:
        print(f"  FAILED: Halo count mismatch: {tessera_tree.n_halos()} vs {hdf5_data['n_halos']}")
        all_passed = False
    else:
        print(f"  PASSED: Halo counts match ({tessera_tree.n_halos()})")
    print()

    # Test 2: Tree table values
    print("=== Test 2: Tree table validation ===")
    n_to_check = min(100, tessera_tree.n_trees())
    tree_table_ok = True

    for i in range(n_to_check):
        tess_entry = tessera_tree.get_tree_entry(i)
        tess_start, tess_len = tess_entry.start_offset, tess_entry.length
        hdf5_start = int(hdf5_data['tree_table']['StartOffset'][i])
        hdf5_len = int(hdf5_data['tree_table']['Length'][i])

        if tess_start != hdf5_start or tess_len != hdf5_len:
            print(f"  FAILED: Tree {i} mismatch: tessera=({tess_start}, {tess_len}), HDF5=({hdf5_start}, {hdf5_len})")
            tree_table_ok = False
            all_passed = False
            break

    if tree_table_ok:
        print(f"  PASSED: Tree table matches for first {n_to_check} trees")
    print()

    # Test 3: Main progenitor chain walking
    print(f"=== Test 3: Main progenitor chain validation ({n_trees_to_test} trees) ===")
    tracker = ts.io.HaloTracker(tessera_tree)

    n_passed = 0
    n_failed = 0

    for tree_id in range(min(n_trees_to_test, tessera_tree.n_trees())):
        # Get tessera branch (root-first order: late to early)
        tess_branch = tracker.trace_from_tree_id(tree_id, backward=True, forward=False)

        # Get HDF5 chain
        start_offset = int(hdf5_data['tree_table']['StartOffset'][tree_id])
        length = int(hdf5_data['tree_table']['Length'][tree_id])
        hdf5_chain = walk_main_progenitor_hdf5(hdf5_data['halos'], start_offset, length)

        # Compare lengths
        if len(tess_branch) != len(hdf5_chain):
            print(f"  Tree {tree_id}: FAILED - length mismatch: tessera={len(tess_branch)}, HDF5={len(hdf5_chain)}")
            if verbose:
                print(f"    tessera first 5: {[(h.snap_num, h.subhalo_nr, h.group_nr) for h in tess_branch[:5]]}")
                print(f"    HDF5 first 5: {hdf5_chain[:5]}")
            n_failed += 1
            all_passed = False
            continue

        # Compare each halo
        # tessera branch is root-first (late to early, descending snap)
        # HDF5 chain is also from root (snap N) to leaf (snap 0)
        match = True
        for i, halo in enumerate(tess_branch):
            hdf5_snap, hdf5_subhalo, hdf5_group = hdf5_chain[i]

            if halo.snap_num != hdf5_snap:
                print(f"  Tree {tree_id}: FAILED - snap mismatch at idx {i}: tessera={halo.snap_num}, HDF5={hdf5_snap}")
                match = False
                break

            if halo.subhalo_nr != hdf5_subhalo:
                print(f"  Tree {tree_id}: FAILED - subhalo mismatch at snap {halo.snap_num}: tessera={halo.subhalo_nr}, HDF5={hdf5_subhalo}")
                match = False
                break

            if halo.group_nr != hdf5_group:
                print(f"  Tree {tree_id}: FAILED - group mismatch at snap {halo.snap_num}: tessera={halo.group_nr}, HDF5={hdf5_group}")
                match = False
                break

        if match:
            n_passed += 1
            if verbose:
                print(f"  Tree {tree_id}: PASSED ({len(tess_branch)} halos)")
        else:
            n_failed += 1
            all_passed = False

    print()
    print(f"  Results: {n_passed} passed, {n_failed} failed")
    print()

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    if all_passed:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate tessera MergerTree against raw HDF5 data"
    )
    parser.add_argument('--tree-path', type=Path, required=True,
                        help='Path to trees.hdf5 or directory with split files')
    parser.add_argument('--n-trees', type=int, default=100,
                        help='Number of trees to validate')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    args = parser.parse_args()

    success = validate_tessera_vs_hdf5(args.tree_path, args.n_trees, args.verbose)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
