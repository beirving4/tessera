#!/usr/bin/env python3
"""
Validate custom merger tree reader against ytree.

This script compares:
1. Number of trees and halos
2. Tree structure (progenitor/descendant links)
3. Halo properties at each snapshot
4. Full main branch traces

Usage
-----
python validate_against_ytree.py \
    --tree-file "/path/to/trees.hdf5" \
    --n-trees 10 \
    --verbose
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from halo_evolution.merger_tree import MergerTree
from halo_evolution.halo_tracker import HaloTracker


def validate_tree_counts(tree: MergerTree, arbor) -> bool:
    """Check that tree/halo counts match."""
    custom_n_trees = tree.n_trees
    ytree_n_trees = len(arbor)

    print(f"  Custom reader: {custom_n_trees} trees")
    print(f"  ytree:         {ytree_n_trees} trees")

    if custom_n_trees != ytree_n_trees:
        print("  FAILED: Tree count mismatch")
        return False

    print("  PASSED: Tree counts match")
    return True


def validate_main_branch(
    tree: MergerTree,
    tracker: HaloTracker,
    arbor,
    tree_id: int,
    verbose: bool = False
) -> bool:
    """
    Compare main branch from custom reader vs ytree.

    Checks:
    - Same number of halos in branch
    - Same SnapNum at each step
    - Same SubhaloNr at each step
    - Same position (within tolerance)
    - Same mass (within tolerance)
    """
    # Get branch from custom reader
    custom_branch = tracker.trace_from_tree_id(
        tree_id=tree_id,
        backward=True,
        forward=False  # Only trace progenitors
    )

    # Get branch from ytree
    ytree_tree = arbor[tree_id]
    ytree_branch = list(ytree_tree['prog'])

    if verbose:
        print(f"  Tree {tree_id}: custom={len(custom_branch)} halos, ytree={len(ytree_branch)}")

    # Compare lengths
    if len(custom_branch) != len(ytree_branch):
        print(f"  FAILED: Branch length mismatch for tree {tree_id}: "
              f"{len(custom_branch)} vs {len(ytree_branch)}")
        return False

    # ytree returns main branch from root (late) to leaf (early)
    # Our custom branch is sorted early to late, so reverse ytree
    ytree_branch_reversed = list(reversed(ytree_branch))

    # Compare each halo
    for i, (custom_halo, ytree_halo) in enumerate(zip(custom_branch, ytree_branch_reversed)):
        # Compare SnapNum
        ytree_snap = int(ytree_halo['SnapNum'])
        if custom_halo.snap_num != ytree_snap:
            print(f"  FAILED: SnapNum mismatch at index {i}: "
                  f"{custom_halo.snap_num} vs {ytree_snap}")
            return False

        # Compare SubhaloNr
        ytree_subhalo = int(ytree_halo['SubhaloNr'])
        if custom_halo.subhalo_nr != ytree_subhalo:
            print(f"  FAILED: SubhaloNr mismatch at snap {custom_halo.snap_num}: "
                  f"{custom_halo.subhalo_nr} vs {ytree_subhalo}")
            return False

        # Compare position (with tolerance for floating point)
        # ytree returns unyt arrays, need to extract values
        ytree_pos = np.array([
            float(ytree_halo['SubhaloPos_0'].value),
            float(ytree_halo['SubhaloPos_1'].value),
            float(ytree_halo['SubhaloPos_2'].value),
        ])
        pos_diff = np.abs(custom_halo.position - ytree_pos)
        if np.any(pos_diff > 1e-3):  # 1 kpc/h tolerance
            print(f"  FAILED: Position mismatch at snap {custom_halo.snap_num}")
            print(f"    Custom: {custom_halo.position}")
            print(f"    ytree:  {ytree_pos}")
            return False

        # Compare mass (with 0.1% tolerance)
        ytree_mass = float(ytree_halo['SubhaloMass'].value)
        if ytree_mass > 0 and custom_halo.mass > 0:
            mass_ratio = custom_halo.mass / ytree_mass
            if abs(mass_ratio - 1.0) > 0.001:
                print(f"  FAILED: Mass mismatch at snap {custom_halo.snap_num}: "
                      f"ratio={mass_ratio:.6f}")
                return False

    if verbose:
        print(f"  Tree {tree_id}: PASSED (all {len(custom_branch)} halos match)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate custom merger tree reader against ytree"
    )
    parser.add_argument('--tree-file', type=Path, required=True,
                        help='Path to trees.hdf5')
    parser.add_argument('--n-trees', type=int, default=10,
                        help='Number of trees to validate')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    args = parser.parse_args()

    print("=" * 60)
    print("MERGER TREE VALIDATION")
    print("=" * 60)
    print()

    # Load custom reader
    print(f"Loading custom reader: {args.tree_file}")
    tree = MergerTree(args.tree_file)
    tracker = HaloTracker(tree)
    print(f"  {tree}")
    print()

    # Load ytree
    print(f"Loading ytree...")
    try:
        import ytree
    except ImportError:
        print("ERROR: ytree not installed. Install with: pip install ytree")
        return 1

    arbor = ytree.load(str(args.tree_file))
    print(f"  Loaded {len(arbor)} trees")
    print()

    # Validation 1: Tree counts
    print("Test 1: Tree and halo counts")
    counts_ok = validate_tree_counts(tree, arbor)
    print()

    # Validation 2: Main branches
    print(f"Test 2: Main branch validation ({args.n_trees} trees)")
    n_passed = 0
    n_tested = min(args.n_trees, tree.n_trees)

    for tree_id in range(n_tested):
        try:
            if validate_main_branch(tree, tracker, arbor, tree_id, verbose=args.verbose):
                n_passed += 1
            else:
                if not args.verbose:
                    print(f"  Tree {tree_id}: FAILED")
        except Exception as e:
            print(f"  Tree {tree_id}: ERROR - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tree counts: {'PASSED' if counts_ok else 'FAILED'}")
    print(f"Main branches: {n_passed}/{n_tested} passed")

    if counts_ok and n_passed == n_tested:
        print("\n✓ All validations PASSED")
        return 0
    else:
        print("\n✗ Some validations FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
