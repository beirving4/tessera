#!/usr/bin/env python3
"""
Batch Processing: Density Slices for Multiple Snapshots

This script processes multiple GADGET-4 snapshots to create density slices,
using the phase-space tessellation method. It provides:

- Directory or file list input modes
- Skip list for excluding specific snapshot numbers (use -1 for ICs)
- Option to include initial conditions files
- Ascending or descending processing order
- Progress bar via tqdm
- Continue-on-failure with summary report

Usage:
    # Process all snapshots in a directory
    python density_slice_batch.py /path/to/output/ -o /path/to/thin_slices/

    # Process specific files
    python density_slice_batch.py snap_000.hdf5 snap_010.hdf5 snap_020.hdf5 -o results/

    # Skip certain snapshots (use -1 to skip ICs)
    python density_slice_batch.py /path/to/output/ -o results/ --skip 0 1 2

    # Include initial conditions and process in descending order
    python density_slice_batch.py /path/to/output/ -o results/ --include-ics --order desc
"""

import numpy as np
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

# Try to find the tessera module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

# Import functions from density_slice.py
from density_slice import (
    load_snapshot,
    compute_density_slice_tessellation,
    save_density_hdf5,
    plot_density_slice,
    HAS_CPP
)

# Check for tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm not available."""
        return iterable


def extract_snapshot_number(filepath):
    """
    Extract snapshot number from a GADGET-4 snapshot filename.
    
    Handles patterns like:
        snapshot_034.hdf5 -> 34
        snap_001.hdf5 -> 1
        snapshot_034/snapshot_034.0.hdf5 -> 34
    
    Returns None if no number can be extracted.
    """
    filename = Path(filepath).name
    
    # Try common patterns
    patterns = [
        r'snapshot_(\d+)',
        r'snap_(\d+)',
        r'snapdir_(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    # Try any sequence of digits before extension
    match = re.search(r'(\d+)(?:\.\d+)?\.hdf5$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def find_snapshots(paths, include_ics=False):
    """
    Find all snapshot files from given paths.
    
    Parameters
    ----------
    paths : list of str
        List of file paths or directory paths
    include_ics : bool
        If True, include initial conditions files (assigned snap_num = -1)
    
    Returns
    -------
    list of tuple
        List of (filepath, snapshot_number) tuples, sorted by snapshot number
    """
    snapshots = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_dir():
            # Find all HDF5 files in directory
            for hdf5_file in sorted(path.glob('*.hdf5')):
                # Skip distributed file chunks (e.g., snapshot_034.0.hdf5)
                if re.search(r'\.\d+\.hdf5$', hdf5_file.name):
                    continue
                # Skip non-snapshot files (fof_subhalo_tab, etc.)
                if not hdf5_file.name.startswith('snapshot'):
                    continue
                
                # Handle initial conditions files
                if '_ics' in hdf5_file.name.lower():
                    if include_ics:
                        snapshots.append((str(hdf5_file), -1))  # ICs get snap_num = -1
                    continue
                
                snap_num = extract_snapshot_number(hdf5_file)
                if snap_num is not None:
                    snapshots.append((str(hdf5_file), snap_num))
            
            # Also check for snapshot directories (distributed format)
            for snap_dir in sorted(path.glob('snapdir_*')):
                snap_num = extract_snapshot_number(snap_dir.name)
                if snap_num is not None:
                    snapshots.append((str(snap_dir), snap_num))
        
        elif path.is_file():
            # Handle ICs files passed directly
            if '_ics' in path.name.lower():
                if include_ics:
                    snapshots.append((str(path), -1))
            else:
                snap_num = extract_snapshot_number(path)
                if snap_num is not None:
                    snapshots.append((str(path), snap_num))
                else:
                    # Use index in list as fallback
                    snapshots.append((str(path), len(snapshots)))
        
        else:
            print(f"Warning: Path not found: {path}")
    
    # Sort by snapshot number (ascending by default)
    snapshots.sort(key=lambda x: x[1])
    
    return snapshots


def process_snapshot(
    filepath,
    snap_num,
    output_dir,
    slice_center=None,
    slice_thickness=None,
    projection_axis='z',
    grid_resolution=256,
    n_samples=100,
    particle_type=1,
    cmap='magma',
    overdensity=True,
    vmin=None,
    vmax=None,
    n_threads=0,
    save_plot=True
):
    """
    Process a single snapshot and save results.
    
    Returns
    -------
    dict
        Result info including success status and any error message
    """
    result = {
        'filepath': filepath,
        'snap_num': snap_num,
        'success': False,
        'error': None,
        'output_h5': None,
        'output_png': None
    }
    
    try:
        # Load snapshot
        positions, particle_ids, header = load_snapshot(
            filepath,
            particle_type=particle_type,
            read_ids=True
        )
        
        box_size = header.box_size
        
        # Set defaults based on box size
        center = slice_center if slice_center is not None else box_size / 2
        thickness = slice_thickness if slice_thickness is not None else box_size * 0.1
        
        # Get particle mass
        mass_per_particle = header.mass_table[particle_type]
        if mass_per_particle == 0:
            mass_per_particle = 1.0
        
        # Compute density slice
        density, extent, slice_info = compute_density_slice_tessellation(
            positions,
            particle_ids,
            box_size,
            center,
            thickness,
            projection_axis=projection_axis,
            grid_resolution=grid_resolution,
            n_samples=n_samples,
            mass_per_particle=mass_per_particle,
            n_threads=n_threads
        )
        
        # Output filenames (handle ICs with snap_num = -1)
        if snap_num == -1:
            output_h5 = output_dir / "density_slice_ics.h5"
            output_png = output_dir / "density_slice_ics.png" if save_plot else None
        else:
            output_h5 = output_dir / f"density_slice_{snap_num:03d}.h5"
            output_png = output_dir / f"density_slice_{snap_num:03d}.png" if save_plot else None
        
        # Save HDF5
        save_density_hdf5(str(output_h5), density, header, slice_info, extent)
        result['output_h5'] = str(output_h5)
        
        # Save plot
        if save_plot:
            plot_density_slice(
                density, extent, slice_info, header,
                output_file=str(output_png),
                cmap=cmap,
                overdensity=overdensity,
                vmin=vmin,
                vmax=vmax
            )
            result['output_png'] = str(output_png)
        
        result['success'] = True
        result['redshift'] = header.redshift
        result['n_particles'] = len(positions)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def print_summary(results, start_time, skipped):
    """Print a summary table of processing results."""
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    
    # Counts
    n_success = sum(1 for r in results if r['success'])
    n_failed = sum(1 for r in results if not r['success'])
    n_skipped = len(skipped)
    n_total = len(results) + n_skipped
    
    print(f"\nTotal snapshots found: {n_total}")
    print(f"  Processed successfully: {n_success}")
    print(f"  Failed: {n_failed}")
    print(f"  Skipped: {n_skipped}")
    print(f"\nTotal time: {elapsed}")
    
    if n_success > 0:
        avg_time = elapsed / n_success
        print(f"Average time per snapshot: {avg_time}")
    
    # Successful snapshots
    if n_success > 0:
        print("\n" + "-" * 70)
        print("PROCESSED SNAPSHOTS")
        print("-" * 70)
        print(f"{'Snap #':<8} {'Redshift':<12} {'Particles':<15} {'Output'}")
        print("-" * 70)
        for r in results:
            if r['success']:
                snap_label = 'ICs' if r['snap_num'] == -1 else str(r['snap_num'])
                print(f"{snap_label:<8} {r.get('redshift', 0):<12.4f} "
                      f"{r.get('n_particles', 0):<15,} {Path(r['output_h5']).name}")
    
    # Failed snapshots
    if n_failed > 0:
        print("\n" + "-" * 70)
        print("FAILED SNAPSHOTS")
        print("-" * 70)
        for r in results:
            if not r['success']:
                snap_label = 'ICs' if r['snap_num'] == -1 else str(r['snap_num'])
                print(f"Snapshot {snap_label}: {r['error']}")
    
    # Skipped snapshots
    if n_skipped > 0:
        print("\n" + "-" * 70)
        skip_labels = ['ICs' if s == -1 else str(s) for s in sorted(skipped)]
        print(f"SKIPPED SNAPSHOTS: {skip_labels}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple GADGET-4 snapshots to create density slices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all snapshots in a directory
  python density_slice_batch.py /path/to/snapshots/ -o /path/to/output/

  # Process specific files
  python density_slice_batch.py snap_000.hdf5 snap_010.hdf5 -o results/

  # Skip snapshots 0, 1, 2 (use -1 to skip ICs)
  python density_slice_batch.py /path/to/snapshots/ -o results/ --skip 0 1 2

  # Include initial conditions and process in descending order (latest first)
  python density_slice_batch.py /path/to/snapshots/ -o results/ --include-ics --order desc

  # Custom slice configuration
  python density_slice_batch.py /path/to/snapshots/ -o results/ --center 128 --thickness 25
        """
    )
    
    parser.add_argument('inputs', type=str, nargs='+',
                        help='Snapshot files or directories to process')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory for density slices')
    
    # Slice configuration
    parser.add_argument('--center', type=float, default=None,
                        help='Slice center (default: box center)')
    parser.add_argument('--thickness', type=float, default=None,
                        help='Slice thickness (default: 10%% of box)')
    parser.add_argument('--axis', type=str, default='z', choices=['x', 'y', 'z'],
                        help='Projection axis (default: z)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution (default: 256)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (default: 100)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Threads per snapshot (0=auto)')
    
    # Visualization options
    parser.add_argument('--cmap', type=str, default='magma',
                        help='Colormap (default: magma)')
    parser.add_argument('--overdensity', action='store_true', default=True,
                        help='Plot overdensity 1+delta (default: True)')
    parser.add_argument('--no-overdensity', action='store_false', dest='overdensity',
                        help='Plot raw density instead of overdensity')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating PNG plots')
    
    # Batch options
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='Snapshot numbers to skip (use -1 to skip ICs)')
    parser.add_argument('--include-ics', action='store_true',
                        help='Include initial conditions files (assigned snap_num = -1)')
    parser.add_argument('--order', type=str, default='asc', choices=['asc', 'desc'],
                        help='Processing order: asc (ascending, default) or desc (descending)')
    
    args = parser.parse_args()
    
    # Check C++ module
    if not HAS_CPP:
        print("Error: C++ tessellation module not available.")
        print("Make sure the build directory is in PYTHONPATH.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all snapshots
    print("Scanning for snapshots...")
    all_snapshots = find_snapshots(args.inputs, include_ics=args.include_ics)
    
    if not all_snapshots:
        print("No snapshots found!")
        sys.exit(1)
    
    print(f"Found {len(all_snapshots)} snapshot(s)")
    if args.include_ics:
        n_ics = sum(1 for s in all_snapshots if s[1] == -1)
        if n_ics > 0:
            print(f"  (including {n_ics} initial conditions file(s))")
    
    # Filter out skipped snapshots
    skip_set = set(args.skip)
    skipped = [s for s in all_snapshots if s[1] in skip_set]
    snapshots = [s for s in all_snapshots if s[1] not in skip_set]
    
    if skip_set:
        print(f"Skipping {len(skipped)} snapshot(s): {sorted(skip_set)}")
    
    # Apply sort order
    if args.order == 'desc':
        snapshots.reverse()
        print("Processing in descending order (latest first)")
    
    if not snapshots:
        print("No snapshots to process after applying skip list!")
        sys.exit(1)
    
    print(f"Processing {len(snapshots)} snapshot(s)...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process snapshots
    start_time = datetime.now()
    results = []
    
    # Use tqdm for progress bar
    pbar = tqdm(snapshots, desc="Processing", unit="snap", 
                disable=not HAS_TQDM, ncols=80)
    
    for filepath, snap_num in pbar:
        if HAS_TQDM:
            pbar.set_postfix({'snap': snap_num})
        else:
            print(f"\nProcessing snapshot {snap_num}: {filepath}")
        
        result = process_snapshot(
            filepath=filepath,
            snap_num=snap_num,
            output_dir=output_dir,
            slice_center=args.center,
            slice_thickness=args.thickness,
            projection_axis=args.axis,
            grid_resolution=args.resolution,
            n_samples=args.samples,
            particle_type=args.particle_type,
            cmap=args.cmap,
            overdensity=args.overdensity,
            vmin=args.vmin,
            vmax=args.vmax,
            n_threads=args.threads,
            save_plot=not args.no_plot
        )
        
        results.append(result)
        
        if not result['success'] and HAS_TQDM:
            tqdm.write(f"  Warning: Snapshot {snap_num} failed: {result['error']}")
    
    # Print summary
    skipped_nums = [s[1] for s in skipped]
    print_summary(results, start_time, skipped_nums)


if __name__ == '__main__':
    main()
