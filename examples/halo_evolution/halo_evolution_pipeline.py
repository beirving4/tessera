#!/usr/bin/env python3
"""
Halo Evolution Visualization Pipeline

Generates animations and multi-panel figures showing the evolution
of a dark matter halo across cosmic time.

This is the consolidated pipeline that uses:
- C++ MergerTree/HaloTracker for high-performance tree traversal (utils/)
- C++ tessera density computation for density field rendering
- Visualization functions from the visualize/ module

Usage
-----
# Basic usage - trace most massive halo at final snapshot
python halo_evolution_pipeline.py \\
    --tree-file /path/to/trees.hdf5 \\
    --snapshot-dir /path/to/snapshots \\
    --output-dir ./halo_evolution_output

# Specific tree
python halo_evolution_pipeline.py \\
    --tree-file /path/to/trees.hdf5 \\
    --snapshot-dir /path/to/snapshots \\
    --output-dir ./output \\
    --tree-id 0 \\
    --a-min 0.1

# Multi-panel figure only (no animation)
python halo_evolution_pipeline.py \\
    --tree-file /path/to/trees.hdf5 \\
    --snapshot-dir /path/to/snapshots \\
    --output-dir ./output \\
    --no-animation \\
    --epochs 0.1 1.0 10.0 100.0

Note
----
On macOS, there may be HDF5 library conflicts between h5py and tessera.
If you encounter crashes, try running the pipeline in two stages:
  1. First run with --extract-branch-only to save branch info
  2. Then run density rendering separately

This issue does not affect Linux systems.

See Also
--------
- utils.MergerTree: C++ merger tree reader
- utils.HaloTracker: C++ branch tracer
- utils.DensityRenderer: Density field computation
- visualize.create_evolution_multipanel: Multi-panel figure generation
- visualize.create_evolution_animation: Animation generation
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Use consolidated modules from tessera.utils and tessera.visualize
from tessera.utils import (
    MergerTree,
    HaloTracker,
    HaloInfo,
    DensityRenderer,
    DensityRenderConfig,
)
from tessera.visualize import (
    create_evolution_animation,
    create_multipanel_figure,
    VisualizationConfig,
    auto_select_epochs,
)


def main():
    parser = argparse.ArgumentParser(
        description='Halo Evolution Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required paths
    parser.add_argument('--tree-file', type=Path, default=None,
                        help='Path to trees.hdf5 (required unless using --branch-file)')
    parser.add_argument('--snapshot-dir', type=Path, required=True,
                        help='Directory containing snapshots')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory')

    # Halo selection
    parser.add_argument('--tree-id', type=int, default=0,
                        help='Tree ID (0 = most massive z=0 halo)')
    parser.add_argument('--snap-num', type=int, default=None,
                        help='Reference snapshot (alternative to tree-id)')
    parser.add_argument('--subhalo-nr', type=int, default=0,
                        help='Subhalo index at reference snapshot')

    # Time range
    parser.add_argument('--a-min', type=float, default=None,
                        help='Minimum scale factor')
    parser.add_argument('--a-max', type=float, default=None,
                        help='Maximum scale factor')

    # Rendering options
    parser.add_argument('--box-size', type=float, default=1.5,
                        help='Box size in Mpc/h (default: 1.5)')
    parser.add_argument('--output-cells', type=int, default=256,
                        help='Grid resolution (default: 256)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Monte Carlo samples per tet (default: 100)')
    parser.add_argument('--n-threads', type=int, default=0,
                        help='Number of threads (0=auto)')

    # Output options
    parser.add_argument('--no-animation', action='store_true',
                        help='Skip animation generation')
    parser.add_argument('--no-multipanel', action='store_true',
                        help='Skip multi-panel figure')
    parser.add_argument('--epochs', type=float, nargs='+', default=None,
                        help='Scale factors for multi-panel (default: 0.1 1.0 10.0 100.0)')
    parser.add_argument('--n-panels', type=int, default=4,
                        help='Number of panels if epochs not specified')
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation FPS (default: 10)')
    parser.add_argument('--colormap', type=str, default='viridis',
                        help='Colormap (default: viridis)')
    parser.add_argument('--extract-branch-only', action='store_true',
                        help='Only extract branch info (for macOS workaround)')
    parser.add_argument('--branch-file', type=Path, default=None,
                        help='Load branch info from file instead of tracing')

    args = parser.parse_args()

    # Validate arguments
    if args.branch_file is None and args.tree_file is None:
        parser.error("--tree-file is required unless using --branch-file")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HALO EVOLUTION VISUALIZATION PIPELINE")
    print("=" * 60)
    print()

    # Check if loading branch from file (macOS workaround)
    if args.branch_file is not None:
        print(f"Loading branch from file: {args.branch_file}")
        with open(args.branch_file, 'r') as f:
            branch_data = json.load(f)

        # Reconstruct HaloInfo objects using C++ class
        branch = []
        for halo_dict in branch_data['halos']:
            halo = HaloInfo()
            halo.tree_index = halo_dict['tree_index']
            halo.snap_num = halo_dict['snap_num']
            halo.subhalo_nr = halo_dict['subhalo_nr']
            halo.group_nr = halo_dict['group_nr']
            halo.scale_factor = halo_dict['scale_factor']
            halo.redshift = halo_dict['redshift']
            halo.position = tuple(halo_dict['position'])
            halo.velocity = tuple(halo_dict['velocity'])
            halo.mass = halo_dict['mass']
            halo.m200c = halo_dict['m200c']
            halo.r200c = halo_dict['r200c']
            branch.append(halo)
        box_size = branch_data['box_size']
        print(f"  Loaded {len(branch)} halos from branch file")
        print()

    else:
        # Load merger tree (C++ implementation for high performance)
        print(f"Loading merger tree: {args.tree_file}")
        tree = MergerTree(str(args.tree_file))
        tracker = HaloTracker(tree)
        print(f"  Trees: {tree.n_trees()}, Halos: {tree.n_halos()}")
        print()

        # Trace halo branch
        print("Tracing halo main branch...")
        if args.snap_num is not None:
            branch = tracker.trace_main_branch(
                snap_num=args.snap_num,
                subhalo_nr=args.subhalo_nr,
                backward=True,
                forward=True,
                a_min=args.a_min,
                a_max=args.a_max,
            )
        else:
            branch = tracker.trace_from_tree_id(
                tree_id=args.tree_id,
                backward=True,
                forward=True,
                a_min=args.a_min,
                a_max=args.a_max,
            )

        if len(branch) == 0:
            print("Error: No halos found in branch")
            return 1

        # Unwrap coordinates for smooth tracking
        tracker.unwrap_coordinates(branch, box_size=tree.box_size)
        box_size = tree.box_size

        print(f"  Branch spans {len(branch)} snapshots")
        print(f"  Scale factor range: {branch[0].scale_factor:.3f} to {branch[-1].scale_factor:.3f}")
        print(f"  Snapshot range: {branch[0].snap_num} to {branch[-1].snap_num}")
        print()

        # Extract branch only mode (macOS workaround)
        if args.extract_branch_only:
            branch_output = args.output_dir / 'branch_data.json'
            branch_data = {
                'tree_file': str(args.tree_file),
                'tree_id': int(args.tree_id),
                'box_size': float(tree.box_size),
                'halos': [
                    {
                        'tree_index': int(h.tree_index),
                        'snap_num': int(h.snap_num),
                        'subhalo_nr': int(h.subhalo_nr),
                        'group_nr': int(h.group_nr),
                        'scale_factor': float(h.scale_factor),
                        'redshift': float(h.redshift),
                        'position': [float(x) for x in h.position],
                        'velocity': [float(x) for x in h.velocity],
                        'mass': float(h.mass),
                        'm200c': float(h.m200c),
                        'r200c': float(h.r200c),
                    }
                    for h in branch
                ]
            }
            with open(branch_output, 'w') as f:
                json.dump(branch_data, f, indent=2)
            print(f"Saved branch data to {branch_output}")
            print("Run again with --branch-file to render densities (macOS workaround)")
            return 0

    # Configure density renderer
    render_config = DensityRenderConfig(
        box_size=args.box_size,
        output_cells=args.output_cells,
        n_samples=args.n_samples,
        n_threads=args.n_threads,
    )

    renderer = DensityRenderer(
        snapshot_dir=args.snapshot_dir,
        snapshot_pattern='snapshot_{:03d}.hdf5',
        config=render_config,
    )

    # Render density fields
    print("Rendering density fields...")

    def progress_callback(i, total):
        halo = branch[i]
        print(f"  [{i+1}/{total}] a={halo.scale_factor:.3f} (snap {halo.snap_num})")

    densities = renderer.render_branch(
        branch,
        mode='2d',
        progress_callback=progress_callback,
    )

    print(f"  Rendered {len(densities)} density fields")
    print()

    # Compute mean density for overdensity
    # Use first available snapshot's particle count
    first_halo = branch[0]
    try:
        sorted_pos, sim_box_size, grid_size = renderer._load_and_sort_particles(first_halo.snap_num)
        n_particles = len(sorted_pos)
        total_mass = n_particles  # Mass per particle = 1 in code units
        mean_density = total_mass / (sim_box_size ** 3)
    except Exception:
        mean_density = 1.0
        # box_size already defined from branch file or tree loading

    # Visualization config
    vis_config = VisualizationConfig(
        colormap=args.colormap,
        fps=args.fps,
        show_overdensity=True,
        colorbar_per_panel=True,
    )

    # Generate multi-panel figure
    if not args.no_multipanel:
        print("Generating multi-panel figure...")

        epochs = args.epochs
        if epochs is None:
            # Auto-select or use defaults
            available = sorted(densities.keys())
            if len(available) >= 4:
                # Try to match reference epochs
                default_epochs = [0.1, 1.0, 10.0, 100.0]
                epochs = default_epochs
            else:
                epochs = auto_select_epochs(available, args.n_panels)

        multipanel_path = args.output_dir / 'halo_evolution_multipanel.png'
        create_multipanel_figure(
            densities=densities,
            branch=branch,
            output_path=multipanel_path,
            select_epochs=epochs,
            config=vis_config,
            box_size=args.box_size,
            mean_density=mean_density,
        )
        print()

    # Generate animation
    if not args.no_animation:
        print("Generating animation...")
        animation_path = args.output_dir / 'halo_evolution.mp4'
        try:
            create_evolution_animation(
                densities=densities,
                branch=branch,
                output_path=animation_path,
                config=vis_config,
                box_size=args.box_size,
                mean_density=mean_density,
            )
        except Exception as e:
            print(f"Warning: Animation generation failed: {e}")
            print("  (This may require ffmpeg to be installed)")
        print()

    # Save branch info
    info_path = args.output_dir / 'branch_info.txt'
    with open(info_path, 'w') as f:
        f.write("Halo Evolution Branch Information\n")
        f.write("=" * 50 + "\n\n")
        if args.tree_file:
            f.write(f"Tree file: {args.tree_file}\n")
        if args.branch_file:
            f.write(f"Branch file: {args.branch_file}\n")
        f.write(f"Tree ID: {args.tree_id}\n")
        f.write(f"Branch length: {len(branch)} snapshots\n")
        f.write(f"Scale factor range: {branch[0].scale_factor:.4f} - {branch[-1].scale_factor:.4f}\n\n")
        f.write("Halo properties at each epoch:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Snap':>5} {'a':>8} {'z':>8} {'M200c':>12} {'R200c':>8}\n")
        f.write("-" * 50 + "\n")
        for halo in branch:
            f.write(f"{halo.snap_num:>5} {halo.scale_factor:>8.4f} {halo.redshift:>8.4f} "
                   f"{halo.m200c:>12.4e} {halo.r200c:>8.4f}\n")

    print(f"Saved branch info to {info_path}")
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
