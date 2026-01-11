#!/usr/bin/env python3
"""
Batch Processing: Density PDFs for Multiple Snapshots

This script processes multiple GADGET-4 snapshots to compute density PDFs
using the phase-space tessellation method. It provides:

- Directory or file list input modes
- Skip list for excluding specific snapshot numbers (use -1 for ICs)
- Option to include initial conditions files
- Ascending or descending processing order
- Progress bar via tqdm
- Continue-on-failure with summary report

Usage:
    # Process all snapshots in a directory
    python density_pdf_batch.py /path/to/output/ -o /path/to/pdfs/

    # Process specific snapshots
    python density_pdf_batch.py /path/to/output/ -o pdfs/ --only 0 14 34 54 74

    # Include initial conditions
    python density_pdf_batch.py /path/to/output/ -o pdfs/ --include-ics
"""

import numpy as np
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

# Try to find the asymptotic_tetra module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

# Import functions from density_pdf.py
from density_pdf import (
    load_snapshot,
    compute_density_field,
    density_to_physical_units,
    compute_pdf,
    save_pdf_hdf5,
    plot_pdf,
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
    """
    filename = Path(filepath).name
    
    patterns = [
        r'snapshot_(\d+)',
        r'snap_(\d+)',
        r'snapdir_(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    match = re.search(r'(\d+)(?:\.\d+)?\.hdf5$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def find_snapshots(paths, include_ics=False):
    """
    Find all snapshot files from given paths.
    """
    snapshots = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_dir():
            for hdf5_file in sorted(path.glob('*.hdf5')):
                if re.search(r'\.\d+\.hdf5$', hdf5_file.name):
                    continue
                if not hdf5_file.name.startswith('snapshot'):
                    continue
                
                if '_ics' in hdf5_file.name.lower():
                    if include_ics:
                        snapshots.append((str(hdf5_file), -1))
                    continue
                
                snap_num = extract_snapshot_number(hdf5_file)
                if snap_num is not None:
                    snapshots.append((str(hdf5_file), snap_num))
            
            for snap_dir in sorted(path.glob('snapdir_*')):
                snap_num = extract_snapshot_number(snap_dir.name)
                if snap_num is not None:
                    snapshots.append((str(snap_dir), snap_num))
        
        elif path.is_file():
            if '_ics' in path.name.lower():
                if include_ics:
                    snapshots.append((str(path), -1))
            else:
                snap_num = extract_snapshot_number(path)
                if snap_num is not None:
                    snapshots.append((str(path), snap_num))
                else:
                    snapshots.append((str(path), len(snapshots)))
        
        else:
            print(f"Warning: Path not found: {path}")
    
    snapshots.sort(key=lambda x: x[1])
    return snapshots


def process_snapshot(
    filepath,
    snap_num,
    output_dir,
    grid_resolution=None,
    n_samples=100,
    particle_type=1,
    n_bins=100,
    log_min=None,
    log_max=None,
    n_threads=0,
    save_plot=True
):
    """
    Process a single snapshot and save PDF results.
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
            filepath, particle_type=particle_type
        )
        
        # Compute density field
        density_3d, config, density_result = compute_density_field(
            positions, particle_ids, header,
            grid_resolution=grid_resolution,
            n_samples=n_samples,
            n_threads=n_threads,
            particle_type=particle_type
        )
        
        # Convert to physical units
        density_physical = density_to_physical_units(density_3d, header.box_size)
        rho_mean_physical = density_physical.mean()
        
        # Compute overdensity
        overdensity = density_3d / density_3d.mean()
        
        # Compute PDF
        bin_centers, bin_edges, pdf, counts = compute_pdf(
            overdensity,
            n_bins=n_bins,
            log_min=log_min,
            log_max=log_max,
            n_threads=n_threads
        )
        
        # Output filenames
        if snap_num == -1:
            output_h5 = output_dir / "density_pdf_ics.h5"
            output_png = output_dir / "density_pdf_ics.png" if save_plot else None
        else:
            output_h5 = output_dir / f"density_pdf_{snap_num:03d}.h5"
            output_png = output_dir / f"density_pdf_{snap_num:03d}.png" if save_plot else None
        
        # Save HDF5
        save_pdf_hdf5(
            str(output_h5),
            density_physical, overdensity,
            bin_centers, bin_edges, pdf, counts,
            header, config, rho_mean_physical
        )
        result['output_h5'] = str(output_h5)
        
        # Save plot
        if save_plot:
            plot_pdf(bin_centers, bin_edges, pdf, counts, header,
                     output_file=str(output_png),
                     log_min=log_min, log_max=log_max)
            result['output_png'] = str(output_png)
        
        result['success'] = True
        result['redshift'] = header.redshift
        result['n_particles'] = len(positions)
        result['overdensity_max'] = float(overdensity.max())
        
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
    
    if n_success > 0:
        print("\n" + "-" * 70)
        print("PROCESSED SNAPSHOTS")
        print("-" * 70)
        print(f"{'Snap #':<8} {'Redshift':<12} {'Max δ+1':<15} {'Output'}")
        print("-" * 70)
        for r in results:
            if r['success']:
                snap_label = 'ICs' if r['snap_num'] == -1 else str(r['snap_num'])
                print(f"{snap_label:<8} {r.get('redshift', 0):<12.4f} "
                      f"{r.get('overdensity_max', 0):<15.1f} {Path(r['output_h5']).name}")
    
    if n_failed > 0:
        print("\n" + "-" * 70)
        print("FAILED SNAPSHOTS")
        print("-" * 70)
        for r in results:
            if not r['success']:
                snap_label = 'ICs' if r['snap_num'] == -1 else str(r['snap_num'])
                print(f"Snapshot {snap_label}: {r['error']}")
    
    if n_skipped > 0:
        print("\n" + "-" * 70)
        skip_labels = ['ICs' if s == -1 else str(s) for s in sorted(skipped)]
        print(f"SKIPPED SNAPSHOTS: {skip_labels}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch compute density PDFs from multiple GADGET-4 snapshots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all snapshots in a directory
  python density_pdf_batch.py /path/to/snapshots/ -o /path/to/output/

  # Process only specific snapshots
  python density_pdf_batch.py /path/to/snapshots/ -o results/ --only 0 14 34 54 74

  # Include initial conditions
  python density_pdf_batch.py /path/to/snapshots/ -o results/ --include-ics

  # Skip certain snapshots
  python density_pdf_batch.py /path/to/snapshots/ -o results/ --skip 0 1 2
        """
    )
    
    parser.add_argument('inputs', type=str, nargs='+',
                        help='Snapshot files or directories to process')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory for PDF results')
    
    # Grid configuration
    parser.add_argument('--resolution', type=int, default=None,
                        help='Grid resolution (default: same as Lagrangian grid)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (default: 100)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Threads per snapshot (0=auto)')
    
    # PDF configuration
    parser.add_argument('--n-bins', type=int, default=100,
                        help='Number of PDF bins (default: 100)')
    parser.add_argument('--log-min', type=float, default=None,
                        help='Log10 of minimum bin edge (auto-detect if not set)')
    parser.add_argument('--log-max', type=float, default=None,
                        help='Log10 of maximum bin edge (auto-detect if not set)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating PNG plots')
    
    # Batch options
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='Snapshot numbers to skip (use -1 to skip ICs)')
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Process only these snapshot numbers (use -1 for ICs)')
    parser.add_argument('--include-ics', action='store_true',
                        help='Include initial conditions files (assigned snap_num = -1)')
    parser.add_argument('--order', type=str, default='asc', choices=['asc', 'desc'],
                        help='Processing order: asc (ascending) or desc (descending)')
    
    args = parser.parse_args()
    
    if not HAS_CPP:
        print("Error: C++ tessellation module not available.")
        print("Make sure the build directory is in PYTHONPATH.")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Filter by --only if specified
    if args.only is not None:
        only_set = set(args.only)
        all_snapshots = [s for s in all_snapshots if s[1] in only_set]
        print(f"Filtering to {len(all_snapshots)} snapshot(s): {sorted(args.only)}")
    
    # Filter out skipped snapshots
    skip_set = set(args.skip)
    skipped = [s for s in all_snapshots if s[1] in skip_set]
    snapshots = [s for s in all_snapshots if s[1] not in skip_set]
    
    if skip_set:
        print(f"Skipping {len(skipped)} snapshot(s): {sorted(skip_set)}")
    
    if not snapshots:
        print("No snapshots to process after applying filters!")
        sys.exit(1)
    
    # Apply sort order
    if args.order == 'desc':
        snapshots.reverse()
        print("Processing in descending order (latest first)")
    
    print(f"Processing {len(snapshots)} snapshot(s)...")
    print(f"Output directory: {output_dir}")
    print()
    
    start_time = datetime.now()
    results = []
    
    pbar = tqdm(snapshots, desc="Processing", unit="snap",
                disable=not HAS_TQDM, ncols=80)
    
    for filepath, snap_num in pbar:
        if HAS_TQDM:
            snap_label = 'ICs' if snap_num == -1 else snap_num
            pbar.set_postfix({'snap': snap_label})
        else:
            snap_label = 'ICs' if snap_num == -1 else snap_num
            print(f"\nProcessing snapshot {snap_label}: {filepath}")
        
        result = process_snapshot(
            filepath=filepath,
            snap_num=snap_num,
            output_dir=output_dir,
            grid_resolution=args.resolution,
            n_samples=args.samples,
            particle_type=args.particle_type,
            n_bins=args.n_bins,
            log_min=args.log_min,
            log_max=args.log_max,
            n_threads=args.threads,
            save_plot=not args.no_plot
        )
        
        results.append(result)
        
        if not result['success'] and HAS_TQDM:
            tqdm.write(f"  Warning: Snapshot {snap_label} failed: {result['error']}")
    
    skipped_nums = [s[1] for s in skipped]
    print_summary(results, start_time, skipped_nums)


if __name__ == '__main__':
    main()
