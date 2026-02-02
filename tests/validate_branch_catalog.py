#!/usr/bin/env python3
"""
Validate Branch Catalog Against Original Merger Tree

This script validates that a branch catalog file correctly reproduces
the main progenitor branches from the original GADGET-4 merger tree.

Usage:
    # Validate all branches (default)
    python validate_branch_catalog.py /path/to/trees.hdf5 /path/to/branch_catalog.h5

    # Validate a random subsample of 100 branches
    python validate_branch_catalog.py /path/to/trees.hdf5 /path/to/branch_catalog.h5 --sample 100

    # Validate specific branch indices
    python validate_branch_catalog.py /path/to/trees.hdf5 /path/to/branch_catalog.h5 --indices 0 1 10 100

Requirements:
    - Original trees.hdf5 file used to create the catalog
    - Branch catalog HDF5 file created by extract_branch_catalog.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add examples/halo_evolution to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "halo_evolution"))

from merger_tree import MergerTree
from halo_tracker import HaloTracker
from branch_catalog import BranchCatalog


def validate_branch(
    catalog: BranchCatalog,
    tracker: HaloTracker,
    cat_idx: int,
    verbose: bool = False,
) -> Optional[str]:
    """
    Validate a single branch from catalog against original tree.

    Parameters
    ----------
    catalog : BranchCatalog
        Branch catalog to validate
    tracker : HaloTracker
        Halo tracker for original tree
    cat_idx : int
        Catalog index of branch to validate
    verbose : bool
        Print detailed info for each branch

    Returns
    -------
    error : str or None
        Error message if validation failed, None if passed
    """
    # Get branch from catalog
    cat_branch = catalog.get_branch(cat_idx)

    # Trace the same branch from original tree
    orig_branch = tracker.trace_from_tree_id(
        tree_id=cat_branch.tree_id,
        backward=True,
        forward=False,
    )

    # Compare lengths
    if len(cat_branch) != len(orig_branch):
        return f"tree_idx={cat_idx}: Length mismatch: catalog={len(cat_branch)}, original={len(orig_branch)}"

    if len(cat_branch) == 0:
        if verbose:
            print(f"  OK tree_idx={cat_idx}: Empty branch (tree_id={cat_branch.tree_id})")
        return None

    # Compare all data at once using numpy for speed
    orig_snap = np.array([h.snap_num for h in orig_branch], dtype=np.int32)
    orig_subhalo = np.array([h.subhalo_nr for h in orig_branch], dtype=np.int32)
    orig_scale = np.array([h.scale_factor for h in orig_branch], dtype=np.float32)
    orig_pos = np.array([h.position for h in orig_branch], dtype=np.float32)
    orig_vel = np.array([h.velocity for h in orig_branch], dtype=np.float32)
    orig_m200c = np.array([h.m200c for h in orig_branch], dtype=np.float32)
    orig_r200c = np.array([h.r200c for h in orig_branch], dtype=np.float32)
    orig_mass = np.array([h.mass for h in orig_branch], dtype=np.float32)

    # Check for mismatches
    if not np.array_equal(cat_branch.snap_num, orig_snap):
        return f"tree_idx={cat_idx}: snap_num mismatch"
    if not np.array_equal(cat_branch.subhalo_nr, orig_subhalo):
        return f"tree_idx={cat_idx}: subhalo_nr mismatch"
    if not np.allclose(cat_branch.scale_factor, orig_scale, rtol=1e-5):
        return f"tree_idx={cat_idx}: scale_factor mismatch"
    if not np.allclose(cat_branch.position, orig_pos, rtol=1e-5):
        return f"tree_idx={cat_idx}: position mismatch"
    if not np.allclose(cat_branch.velocity, orig_vel, rtol=1e-5):
        return f"tree_idx={cat_idx}: velocity mismatch"
    if not np.allclose(cat_branch.m200c, orig_m200c, rtol=1e-5):
        return f"tree_idx={cat_idx}: m200c mismatch"
    if not np.allclose(cat_branch.r200c, orig_r200c, rtol=1e-5):
        return f"tree_idx={cat_idx}: r200c mismatch"
    if not np.allclose(cat_branch.mass, orig_mass, rtol=1e-5):
        return f"tree_idx={cat_idx}: mass mismatch"

    if verbose:
        print(
            f"  OK tree_idx={cat_idx}: {len(cat_branch)} halos, "
            f"M200c={cat_branch.root_m200c:.2e}, "
            f"a=[{cat_branch.a_min:.4f}, {cat_branch.a_max:.4f}]"
        )

    return None


def validate_branch_catalog(
    tree_file: Path,
    catalog_file: Path,
    sample_size: Optional[int] = None,
    indices: Optional[List[int]] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Validate branch catalog against original merger tree.

    Parameters
    ----------
    tree_file : Path
        Path to original GADGET-4 trees.hdf5 file
    catalog_file : Path
        Path to branch catalog HDF5 file
    sample_size : int, optional
        If provided, validate a random sample of this many branches
    indices : list of int, optional
        If provided, validate only these specific indices
    verbose : bool
        Print detailed info for each branch
    show_progress : bool
        Show progress bar

    Returns
    -------
    results : dict
        Validation results with keys: n_validated, n_errors, errors, n_halos
    """
    # Load original tree
    print(f"Loading original merger tree from {tree_file}...")
    tree = MergerTree(tree_file)
    tracker = HaloTracker(tree)

    # Load catalog
    print(f"Loading branch catalog from {catalog_file}...")
    catalog = BranchCatalog(catalog_file)
    print(catalog.summary())
    print()

    # Determine which indices to validate
    if indices is not None:
        validate_indices = [i for i in indices if 0 <= i < catalog.n_trees]
        print(f"Validating {len(validate_indices)} specified branches...")
    elif sample_size is not None and sample_size < catalog.n_trees:
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        validate_indices = sorted(rng.choice(catalog.n_trees, size=sample_size, replace=False))
        print(f"Validating random sample of {len(validate_indices)} branches...")
    else:
        validate_indices = list(range(catalog.n_trees))
        print(f"Validating all {len(validate_indices):,} branches...")

    # Validate branches
    errors = []
    n_halos_checked = 0

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(validate_indices, desc="Validating")
        except ImportError:
            iterator = validate_indices
            print("(install tqdm for progress bar)")
    else:
        iterator = validate_indices

    for cat_idx in iterator:
        error = validate_branch(catalog, tracker, cat_idx, verbose=verbose)
        if error:
            errors.append(error)
        else:
            # Count halos for successfully validated branches
            branch = catalog.get_branch(cat_idx)
            n_halos_checked += len(branch)

    # Report results
    print()
    print("=" * 60)
    print("Validation complete!")
    print(f"  Branches validated: {len(validate_indices):,}")
    print(f"  Halos checked: {n_halos_checked:,}")
    print(f"  Errors found: {len(errors)}")

    if errors:
        print(f"\nErrors:")
        for err in errors[:20]:  # Show first 20 errors
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    else:
        print(f"\nAll branches match perfectly!")

    return {
        "n_validated": len(validate_indices),
        "n_errors": len(errors),
        "errors": errors,
        "n_halos": n_halos_checked,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate branch catalog against original merger tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tree_file",
        type=Path,
        help="Original GADGET-4 trees.hdf5 file",
    )
    parser.add_argument(
        "catalog_file",
        type=Path,
        help="Branch catalog HDF5 file to validate",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Validate a random sample of N branches (default: validate all)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Validate specific branch indices",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed info for each validated branch",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    if not args.tree_file.exists():
        raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
    if not args.catalog_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {args.catalog_file}")

    results = validate_branch_catalog(
        tree_file=args.tree_file,
        catalog_file=args.catalog_file,
        sample_size=args.sample,
        indices=args.indices,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )

    # Exit with error code if validation failed
    sys.exit(1 if results["n_errors"] > 0 else 0)


if __name__ == "__main__":
    main()
