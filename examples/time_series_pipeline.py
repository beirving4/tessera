#!/usr/bin/env python3
"""
Complete time-series visualization pipeline.

Runs all steps:
1. Generate 2D density projections from snapshots
2. Build static time-series image
3. Create evolution animation

Usage:
    python time_series_pipeline.py --snapshot-dir /path/to/snapshots --output-dir /path/to/output

For testing with a subset of snapshots:
    python time_series_pipeline.py --snapshot-dir /path/to/snapshots --output-dir /path/to/output \
        --snapshots 0 10 20 30 40 50 60 70
"""

import argparse
import os
from pathlib import Path
import time

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from time_series_config import TimeSeriesConfig, get_snapshot_files
from generate_projections import generate_all_projections
from build_time_series import render_time_series
from animate_evolution import create_animation, check_ffmpeg


def run_pipeline(
    config: TimeSeriesConfig,
    skip_existing: bool = False,
    skip_animation: bool = False
):
    """Run the complete visualization pipeline."""

    print("=" * 70)
    print("TIME-SERIES COSMIC WEB VISUALIZATION PIPELINE")
    print("=" * 70)
    print()

    # Print configuration summary
    print("Configuration:")
    print(f"  Snapshot directory: {config.snapshot_dir}")
    print(f"  Output directory:   {config.output_dir}")
    print(f"  Box size:           {config.box_size} Mpc/h")
    print(f"  Particles:          {config.n_particles_per_dim}^3")
    print(f"  Output resolution:  {config.output_cells}^2")
    print(f"  Slab fraction:      {config.slab_fraction:.1%}")
    print(f"  Box replications:   {config.n_box_replications}")
    print(f"  Scale factor range: {config.a_min} -> {config.a_max}")
    print()

    # Check ffmpeg early if we'll need it
    if not skip_animation:
        try:
            check_ffmpeg()
            print("ffmpeg: found")
        except RuntimeError as e:
            print(f"Warning: {e}")
            print("Animation step will be skipped.")
            skip_animation = True
    print()

    total_start = time.time()

    # Step 1: Generate projections
    print("-" * 70)
    print("STEP 1: Generating density projections")
    print("-" * 70)

    if skip_existing and config.projections_file.exists():
        print(f"Skipping: {config.projections_file} already exists")
    else:
        step_start = time.time()
        generate_all_projections(config)
        print(f"Step 1 completed in {time.time() - step_start:.1f} seconds")
    print()

    # Step 2: Build static image
    print("-" * 70)
    print("STEP 2: Building time-series image")
    print("-" * 70)

    step_start = time.time()
    render_time_series(config)
    print(f"Step 2 completed in {time.time() - step_start:.1f} seconds")
    print()

    # Step 3: Create animation
    if not skip_animation:
        print("-" * 70)
        print("STEP 3: Creating animation")
        print("-" * 70)

        step_start = time.time()
        create_animation(config)
        print(f"Step 3 completed in {time.time() - step_start:.1f} seconds")
        print()

    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {time.time() - total_start:.1f} seconds")
    print()
    print("Output files:")
    print(f"  Projections:   {config.projections_file}")
    print(f"  Static image:  {config.static_image_file}")
    if not skip_animation:
        print(f"  Animation:     {config.animation_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete time-series visualization pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--snapshot-dir", type=Path, required=True,
        help="Directory containing snapshot files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory"
    )

    # Simulation parameters
    parser.add_argument(
        "--box-size", type=float, default=256.0,
        help="Box size in Mpc/h"
    )
    parser.add_argument(
        "--n-particles", type=int, default=256,
        help="Particles per dimension"
    )

    # Density field parameters
    parser.add_argument(
        "--output-cells", type=int, default=512,
        help="Output grid resolution"
    )
    parser.add_argument(
        "--slab-fraction", type=float, default=0.08,
        help="Fraction of box depth for projection slab"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Monte Carlo samples per tetrahedron"
    )

    # Time-series parameters
    parser.add_argument(
        "--n-replications", type=int, default=4,
        help="Number of box replications in time direction"
    )
    parser.add_argument(
        "--a-min", type=float, default=0.02,
        help="Minimum scale factor"
    )
    parser.add_argument(
        "--a-max", type=float, default=100.0,
        help="Maximum scale factor"
    )

    # Animation parameters
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Animation frames per second"
    )
    parser.add_argument(
        "--interpolate", type=int, default=4,
        help="Frame interpolation factor"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Animation output DPI"
    )

    # Visualization
    parser.add_argument(
        "--colormap", type=str, default="cosmic_blue",
        help="Colormap for visualization"
    )
    parser.add_argument(
        "--physical-density", action="store_true",
        help="Show physical (not comoving) density"
    )

    # Snapshot selection
    parser.add_argument(
        "--snapshots", type=int, nargs="+",
        help="Specific snapshot indices to process"
    )

    # Workflow options
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip projection generation if file exists"
    )
    parser.add_argument(
        "--skip-animation", action="store_true",
        help="Skip animation creation"
    )
    parser.add_argument(
        "--n-threads", type=int, default=0,
        help="OpenMP threads (0 = auto)"
    )

    args = parser.parse_args()

    # Build configuration
    config = TimeSeriesConfig(
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        box_size=args.box_size,
        n_particles_per_dim=args.n_particles,
        output_cells=args.output_cells,
        slab_fraction=args.slab_fraction,
        n_samples=args.n_samples,
        n_threads=args.n_threads,
        n_box_replications=args.n_replications,
        a_min=args.a_min,
        a_max=args.a_max,
        use_physical_density=args.physical_density,
        colormap=args.colormap,
        fps=args.fps,
        interpolate_frames=args.interpolate,
        dpi=args.dpi,
        snapshot_indices=args.snapshots
    )

    run_pipeline(
        config,
        skip_existing=args.skip_existing,
        skip_animation=args.skip_animation
    )


if __name__ == "__main__":
    main()
