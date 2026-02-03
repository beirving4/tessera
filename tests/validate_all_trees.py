#!/usr/bin/env python3
"""
Validate tessera MergerTree against raw HDF5 TreeMainProgenitor data.

This script validates that tessera correctly follows the TreeMainProgenitor
chain (the true main branch - most massive progenitor at each step).

NOTE: ytree's `prog` selector uses TreeFirstProgenitor, not TreeMainProgenitor.
These differ when halos have multiple progenitors. For tracking halo evolution,
TreeMainProgenitor (what tessera uses) is the correct choice.

By default, tests ALL trees for complete validation before cluster deployment.

Usage
-----
python compare_tessera_ytree.py --tree-path "/path/to/trees.hdf5"
python compare_tessera_ytree.py --tree-path "/path/to/trees.hdf5" --n-trees 100  # test subset
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import h5py

# Try to import tessera
try:
    import tessera as ts
except ImportError:
    print("ERROR: tessera not installed. Install with: pip install -e /path/to/tessera")
    sys.exit(1)


def walk_hdf5_main_progenitor(snap_num, subhalo_nr, group_nr, main_prog, start, length):
    """Walk TreeMainProgenitor chain from raw HDF5 arrays."""
    chain = []
    tree_idx = 0
    while tree_idx >= 0 and tree_idx < length:
        global_idx = start + tree_idx
        chain.append((int(snap_num[global_idx]),
                      int(subhalo_nr[global_idx]),
                      int(group_nr[global_idx])))
        next_prog = int(main_prog[global_idx])
        if next_prog < 0:
            break
        tree_idx = next_prog
    return chain


def compare_implementations(tree_path: str, n_trees_to_test: int = None, verbose: bool = False) -> bool:
    """
    Validate tessera against raw HDF5 TreeMainProgenitor data.

    Parameters
    ----------
    tree_path : str
        Path to trees.hdf5 file
    n_trees_to_test : int, optional
        Number of trees to compare. If None, tests ALL trees.
    verbose : bool
        Print detailed output for each tree

    Returns
    -------
    bool
        True if all comparisons pass
    """
    print("=" * 70)
    print("TESSERA vs RAW HDF5 TreeMainProgenitor VALIDATION")
    print("=" * 70)
    print(f"Tree file: {tree_path}")
    print()

    # Load with tessera
    print("Loading with tessera...")
    t0 = time.time()
    tess_tree = ts.io.MergerTree(tree_path)
    tess_tracker = ts.io.HaloTracker(tess_tree)
    tess_load_time = time.time() - t0
    print(f"  Loaded: {tess_tree.n_trees()} trees, {tess_tree.n_halos()} halos")
    print(f"  Load time: {tess_load_time:.2f}s")
    print()

    # Load raw HDF5 data
    print("Loading raw HDF5 data...")
    t0 = time.time()
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
    hdf5_load_time = time.time() - t0
    print(f"  Loaded: {hdf5_n_trees} trees, {hdf5_n_halos} halos")
    print(f"  Load time: {hdf5_load_time:.2f}s")
    print()

    # Test 1: Tree counts
    print("=== Test 1: Tree and halo counts ===")
    counts_match = (tess_tree.n_trees() == hdf5_n_trees and
                    tess_tree.n_halos() == hdf5_n_halos)
    if counts_match:
        print(f"  PASSED: Both have {tess_tree.n_trees()} trees, {tess_tree.n_halos()} halos")
    else:
        print(f"  FAILED: tessera=({tess_tree.n_trees()}, {tess_tree.n_halos()}), HDF5=({hdf5_n_trees}, {hdf5_n_halos})")
    print()

    # Determine which trees to test
    total_trees = tess_tree.n_trees()
    if n_trees_to_test is None:
        n_trees = total_trees
        test_ids = list(range(total_trees))
        print(f"=== Test 2: TreeMainProgenitor chain comparison (ALL {n_trees} trees) ===")
    else:
        import random
        random.seed(42)
        n_trees = min(n_trees_to_test, total_trees)
        test_ids = random.sample(range(total_trees), n_trees)
        print(f"=== Test 2: TreeMainProgenitor chain comparison ({n_trees} trees) ===")

    n_passed = 0
    n_failed = 0
    failures = []

    # Progress reporting
    report_interval = max(1, n_trees // 20)  # Report every 5%
    t0 = time.time()

    for idx, tree_id in enumerate(test_ids):
        # Progress update
        if idx > 0 and idx % report_interval == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (n_trees - idx) / rate if rate > 0 else 0
            print(f"  Progress: {idx}/{n_trees} ({100*idx/n_trees:.1f}%) - {n_failed} failures - ETA: {eta:.1f}s")

        # Get tessera branch (backward only = main progenitor chain)
        tess_branch = tess_tracker.trace_from_tree_id(tree_id, backward=True, forward=False)

        # Get HDF5 main progenitor chain
        start = int(starts[tree_id])
        length = int(lengths[tree_id])
        hdf5_chain = walk_hdf5_main_progenitor(snap_num, subhalo_nr, group_nr,
                                                main_prog, start, length)

        # Compare lengths
        if len(tess_branch) != len(hdf5_chain):
            n_failed += 1
            failures.append({
                'tree_id': tree_id,
                'reason': 'length_mismatch',
                'tess_len': len(tess_branch),
                'hdf5_len': len(hdf5_chain)
            })
            if verbose:
                print(f"  Tree {tree_id}: FAILED - length mismatch: tessera={len(tess_branch)}, HDF5={len(hdf5_chain)}")
            continue

        # Compare each halo in the chain
        match = True
        for i, (tess_halo, hdf5_halo) in enumerate(zip(tess_branch, hdf5_chain)):
            hdf5_snap, hdf5_subhalo, hdf5_group = hdf5_halo

            # Compare
            if tess_halo.snap_num != hdf5_snap:
                match = False
                failures.append({
                    'tree_id': tree_id,
                    'reason': 'snap_mismatch',
                    'index': i,
                    'tess': tess_halo.snap_num,
                    'hdf5': hdf5_snap
                })
                if verbose:
                    print(f"  Tree {tree_id}: FAILED - snap mismatch at idx {i}: tessera={tess_halo.snap_num}, HDF5={hdf5_snap}")
                break

            if tess_halo.subhalo_nr != hdf5_subhalo:
                match = False
                failures.append({
                    'tree_id': tree_id,
                    'reason': 'subhalo_mismatch',
                    'index': i,
                    'tess': tess_halo.subhalo_nr,
                    'hdf5': hdf5_subhalo
                })
                if verbose:
                    print(f"  Tree {tree_id}: FAILED - subhalo mismatch at idx {i}: tessera={tess_halo.subhalo_nr}, HDF5={hdf5_subhalo}")
                break

            if tess_halo.group_nr != hdf5_group:
                match = False
                failures.append({
                    'tree_id': tree_id,
                    'reason': 'group_mismatch',
                    'index': i,
                    'tess': tess_halo.group_nr,
                    'hdf5': hdf5_group
                })
                if verbose:
                    print(f"  Tree {tree_id}: FAILED - group mismatch at idx {i}: tessera={tess_halo.group_nr}, HDF5={hdf5_group}")
                break

        if match:
            n_passed += 1
            if verbose:
                print(f"  Tree {tree_id}: PASSED ({len(tess_branch)} halos)")
        else:
            n_failed += 1

    total_time = time.time() - t0
    print(f"  Completed in {total_time:.2f}s ({n_trees/total_time:.1f} trees/s)")
    print(f"  Results: {n_passed}/{n_trees} passed, {n_failed}/{n_trees} failed")

    if failures and not verbose:
        print(f"  First failure: {failures[0]}")
    print()

    # Summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_passed = counts_match and (n_failed == 0)
    if all_passed:
        print("All validations PASSED - tessera matches raw HDF5 TreeMainProgenitor!")
    else:
        print("Some validations FAILED")
        if not counts_match:
            print(f"  - Tree/halo count mismatch")
        if n_failed > 0:
            print(f"  - {n_failed} branch mismatches")
            print(f"  First failure: {failures[0]}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare tessera and ytree merger tree implementations"
    )
    parser.add_argument('--tree-path', type=str, required=True,
                        help='Path to trees.hdf5 file')
    parser.add_argument('--n-trees', type=int, default=None,
                        help='Number of trees to compare (default: ALL trees)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output for each tree')

    args = parser.parse_args()

    if not Path(args.tree_path).exists():
        print(f"ERROR: Tree file not found: {args.tree_path}")
        return 1

    success = compare_implementations(args.tree_path, args.n_trees, args.verbose)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
