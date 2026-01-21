#!/usr/bin/env python3
"""
Generate 2D density projections from N-body snapshots using tessera.

This is Step 1 of the time-series visualization pipeline.

Usage:
    python generate_projections.py --snapshot-dir /path/to/snapshots --output-dir /path/to/output
    python generate_projections.py --snapshot-dir /path/to/snapshots --output-dir /path/to/output --snapshots 0 10 20 30
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import h5py

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SCRIPT_DIR = Path(__file__).parent.resolve()

# Import shared utilities
from tessera.utils import load_snapshot_h5py, infer_grid_size

# Import tessera (try installed package first, then development build)
try:
    import _tessera as ts
except ImportError:
    # Try development build
    REPO_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(REPO_ROOT / 'build'))
    try:
        import _tessera as ts
    except ImportError:
        print("Error: tessera not found. Install with 'pip install .' or build from source.")
        sys.exit(1)

from time_series_config import (
    TimeSeriesConfig,
    get_snapshot_files,
    get_scale_factor,
    save_projection
)


def load_snapshot(snapshot_path: Path) -> tuple:
    """Load particle positions and IDs from a GADGET-4 HDF5 snapshot."""
    # Use utils function for loading
    positions, particle_ids, box_size, _ = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )
    return positions, particle_ids, box_size


def compute_projection(
    positions: np.ndarray,
    particle_ids: np.ndarray,
    config: TimeSeriesConfig,
    scale_factor: float = 1.0
) -> np.ndarray:
    """
    Compute 2D density projection using tessera.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    particle_ids : np.ndarray
        Particle IDs for Lagrangian ordering
    config : TimeSeriesConfig
        Pipeline configuration
    scale_factor : float
        Scale factor (used if computing physical density)

    Returns
    -------
    density_2d : np.ndarray
        2D projected density field
    """
    grid_size = config.n_particles_per_dim

    # Sort particles by Lagrangian ID using Python (workaround for C++ segfault bug)
    sort_order = np.argsort(particle_ids)
    sorted_positions = np.ascontiguousarray(positions[sort_order])

    # Configure density computation
    density_config = ts.density.TetraDensityConfig()
    density_config.lagrangian_grid_size = grid_size
    density_config.box_size = config.box_size
    density_config.output_cells = config.output_cells
    density_config.n_samples = config.n_samples
    density_config.n_threads = config.n_threads
    density_config.periodic = True
    density_config.particle_mass = 1.0

    # Use subbox to extract a slab for projection
    if config.slab_fraction < 1.0:
        density_config.subbox_enabled = True
        slab_width = config.box_size * config.slab_fraction
        slab_start = config.box_size * (config.slab_center - config.slab_fraction / 2)

        # Set subbox based on projection axis
        if config.projection_axis == 2:  # z-projection
            density_config.subbox_origin = (0.0, 0.0, slab_start)
            density_config.subbox_width = (config.box_size, config.box_size, slab_width)
        elif config.projection_axis == 1:  # y-projection
            density_config.subbox_origin = (0.0, slab_start, 0.0)
            density_config.subbox_width = (config.box_size, slab_width, config.box_size)
        else:  # x-projection
            density_config.subbox_origin = (slab_start, 0.0, 0.0)
            density_config.subbox_width = (slab_width, config.box_size, config.box_size)

    # Compute 2D projection directly (more memory efficient)
    result = ts.density.compute_tetra_density_2d_projection(
        sorted_positions, density_config, config.projection_axis
    )
    density_2d = np.array(result.density).reshape(config.output_cells, config.output_cells)

    # Optionally convert to physical density
    if config.use_physical_density:
        density_2d = density_2d / (scale_factor ** 3)

    return density_2d


def generate_all_projections(config: TimeSeriesConfig, verbose: bool = True):
    """Generate and save projections for all snapshots."""

    # Initialize output file
    if config.projections_file.exists():
        config.projections_file.unlink()

    # Get list of snapshots
    snapshot_files = get_snapshot_files(config)

    if not snapshot_files:
        print(f"Error: No snapshots found in {config.snapshot_dir}")
        return

    if verbose:
        print(f"Found {len(snapshot_files)} snapshots")
        print(f"Output resolution: {config.output_cells}x{config.output_cells}")
        print(f"Slab fraction: {config.slab_fraction:.1%}")
        print()

    for i, (idx, path) in enumerate(snapshot_files):
        # Get scale factor
        try:
            a = get_scale_factor(path, idx, config)
        except ValueError as e:
            print(f"Warning: {e}, skipping")
            continue

        if verbose:
            print(f"[{i+1}/{len(snapshot_files)}] Snapshot {idx}: a = {a:.4f} ... ", end="", flush=True)

        # Load snapshot
        positions, particle_ids, box_size = load_snapshot(path)

        # Update config box size if different (shouldn't happen, but just in case)
        if abs(box_size - config.box_size) > 1e-6:
            print(f"Warning: Box size mismatch ({box_size} vs {config.box_size})")

        # Compute projection
        density_2d = compute_projection(positions, particle_ids, config, scale_factor=a)

        # Save
        save_projection(config.projections_file, a, density_2d, idx)

        if verbose:
            print(f"done (density range: {density_2d.min():.2e} - {density_2d.max():.2e})")

    if verbose:
        print(f"\nProjections saved to: {config.projections_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D density projections from N-body snapshots"
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, required=True,
        help="Directory containing snapshot files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for projections"
    )
    parser.add_argument(
        "--box-size", type=float, default=256.0,
        help="Box size in Mpc/h (default: 256)"
    )
    parser.add_argument(
        "--n-particles", type=int, default=256,
        help="Particles per dimension (default: 256)"
    )
    parser.add_argument(
        "--output-cells", type=int, default=512,
        help="Output grid resolution (default: 512)"
    )
    parser.add_argument(
        "--slab-fraction", type=float, default=0.08,
        help="Fraction of box for projection slab (default: 0.08)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Monte Carlo samples per tetrahedron (default: 100)"
    )
    parser.add_argument(
        "--n-threads", type=int, default=0,
        help="Number of OpenMP threads (default: 0 = auto)"
    )
    parser.add_argument(
        "--physical-density", action="store_true",
        help="Compute physical (not comoving) density"
    )
    parser.add_argument(
        "--snapshots", type=int, nargs="+",
        help="Specific snapshot indices to process (default: all)"
    )

    args = parser.parse_args()

    config = TimeSeriesConfig(
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        box_size=args.box_size,
        n_particles_per_dim=args.n_particles,
        output_cells=args.output_cells,
        slab_fraction=args.slab_fraction,
        n_samples=args.n_samples,
        n_threads=args.n_threads,
        use_physical_density=args.physical_density,
        snapshot_indices=args.snapshots
    )

    generate_all_projections(config)


if __name__ == "__main__":
    main()
