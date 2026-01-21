#!/usr/bin/env python3
"""
Overdensity PDF Analysis with ORIGAMI Morphology Classification

Computes overdensity PDFs for:
1. The full density field (all particles)
2. Each ORIGAMI morphological class (void, wall, filament, halo)

Supports jackknife resampling for uncertainty estimation.

Outputs:
- HDF5 file with histogram and PDF data (and errors if --jackknife)
- 2x1 figure: histograms (top) and PDFs (bottom) with shared x-axis

Usage:
    python overdensity_pdf_origami.py
    python overdensity_pdf_origami.py --jackknife  # With uncertainty estimation
    python overdensity_pdf_origami.py --jackknife --n-subboxes 3  # 27 sub-boxes
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()

# Import shared utilities
from tessera.utils import infer_grid_size, load_snapshot_h5py

# Try to import local config, fall back to example paths
try:
    from config import SNAPSHOT_BASE
except ImportError:
    raise ImportError(
        "Please create examples/config.py with your local paths.\n"
        "Copy config.example.py to config.py and update the paths."
    )

# Import tessera (try installed package first, then development build)
try:
    import _tessera as ts
except ImportError:
    REPO_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(REPO_ROOT / 'build'))
    import _tessera as ts

# Snapshots to analyze
SNAPSHOTS = [
    {'name': 'snapshot_034', 'description': 'a=1, z=0'},
    {'name': 'snapshot_074', 'description': 'a=100, z=-0.99'},
]

# ORIGAMI class names and colors
ORIGAMI_CLASSES = ['Void', 'Wall', 'Filament', 'Halo']
ORIGAMI_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_snapshot(snapshot_name):
    """Load particle data from GADGET-4 snapshot."""
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"
    print(f"Loading {snapshot_path}...")

    # Use utils function for loading
    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )

    # Read redshift separately (not returned by load_snapshot_h5py)
    with h5py.File(snapshot_path, 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])

    # Infer grid size using utils function
    grid_size = infer_grid_size(len(positions))

    print(f"  Loaded {len(positions):,} particles (grid: {grid_size}^3)")
    print(f"  Scale factor: {scale_factor:.4f}, Redshift: {redshift:.4f}")

    return positions, particle_ids, box_size, redshift, scale_factor, grid_size


def run_origami_with_pdf(positions, particle_ids, box_size, grid_size,
                         use_jackknife=False, n_subboxes_per_dim=2,
                         output_cells=256, n_samples=100, n_bins=100):
    """Run unified ORIGAMI pipeline with integrated PDF computation.

    Args:
        positions: Particle positions
        particle_ids: Particle IDs
        box_size: Simulation box size
        grid_size: Lagrangian grid size
        use_jackknife: Enable jackknife resampling
        n_subboxes_per_dim: Sub-boxes per dimension for jackknife
        output_cells: Density grid resolution
        n_samples: Monte Carlo samples for density
        n_bins: Number of histogram bins

    Returns:
        result: Pipeline result with all outputs
        hist_results: Dict formatted for compatibility with save/plot functions
    """
    print("  Running unified ORIGAMI pipeline with PDF computation...")

    # Configure pipeline
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = output_cells
    config.density_n_samples = n_samples
    config.sample_density_at_particles = True
    config.n_threads = 1
    config.n_split = 1

    # PDF configuration
    config.pdf_n_bins = n_bins
    config.pdf_log_bins = True
    config.pdf_jackknife = use_jackknife
    config.pdf_jackknife_subboxes = n_subboxes_per_dim

    # Run pipeline
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"    Detected ID ordering: {ts.origami.id_ordering_name(result.detected_ordering)}")
    print(f"    Pipeline completed in {result.total_time_ms:.1f} ms")
    print(f"    PDF computation: {result.pdf_time_ms:.1f} ms")

    print(f"\n  ORIGAMI Results:")
    print(f"    Void:     {result.n_void:>10,} ({result.f_void:>6.2%})")
    print(f"    Wall:     {result.n_wall:>10,} ({result.f_wall:>6.2%})")
    print(f"    Filament: {result.n_filament:>10,} ({result.f_filament:>6.2%})")
    print(f"    Halo:     {result.n_halo:>10,} ({result.f_halo:>6.2%})")

    print(f"\n  Mean overdensity: {result.overdensity_mean:.6f}")

    # Convert pipeline result to hist_results format for compatibility
    bin_edges = np.array(result.pdf_bin_edges)
    bin_centers = np.array(result.pdf_bin_centers)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    hist_results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'mean_density': result.mean_density,
        'jackknife': use_jackknife,
    }

    if use_jackknife:
        hist_results['n_subboxes'] = result.pdf_jackknife_n_subboxes
        hist_results['n_subboxes_per_dim'] = n_subboxes_per_dim

    # All particles
    particle_density = np.array(result.particle_density)
    mean_density = particle_density.mean()
    overdensity = particle_density / mean_density

    hist_results['all'] = {
        'histogram': np.array(result.hist_all),
        'pdf': np.array(result.pdf_all),
        'n_particles': len(particle_density),
        'mean_overdensity': result.overdensity_mean,
        'median_overdensity': float(np.median(overdensity)),
    }
    if use_jackknife:
        hist_results['all']['pdf_error'] = np.array(result.pdf_all_error)

    # Per-class results
    morphology = np.array(result.morphology)
    class_data = [
        ('void', result.hist_void, result.pdf_void, result.pdf_void_error if use_jackknife else None, result.n_void),
        ('wall', result.hist_wall, result.pdf_wall, result.pdf_wall_error if use_jackknife else None, result.n_wall),
        ('filament', result.hist_filament, result.pdf_filament, result.pdf_filament_error if use_jackknife else None, result.n_filament),
        ('halo', result.hist_halo, result.pdf_halo, result.pdf_halo_error if use_jackknife else None, result.n_halo),
    ]

    for i, (key, hist, pdf, pdf_err, n_particles) in enumerate(class_data):
        mask = morphology == i
        od_class = overdensity[mask] if mask.sum() > 0 else np.array([0.0])
        hist_results[key] = {
            'histogram': np.array(hist),
            'pdf': np.array(pdf),
            'n_particles': n_particles,
            'mean_overdensity': float(od_class.mean()),
            'median_overdensity': float(np.median(od_class)),
        }
        if use_jackknife and pdf_err is not None:
            hist_results[key]['pdf_error'] = np.array(pdf_err)

    return result, hist_results


def save_to_hdf5(output_path, hist_results, snapshot_info):
    """Save histogram and PDF results to HDF5."""
    print(f"  Saving to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Header
        hdr = f.create_group('Header')
        hdr.attrs['created'] = datetime.now().isoformat()
        hdr.attrs['snapshot'] = snapshot_info['name']
        hdr.attrs['description'] = snapshot_info['description']
        hdr.attrs['scale_factor'] = snapshot_info['scale_factor']
        hdr.attrs['redshift'] = snapshot_info['redshift']
        hdr.attrs['box_size'] = snapshot_info['box_size']
        hdr.attrs['grid_size'] = snapshot_info['grid_size']
        hdr.attrs['mean_density'] = hist_results['mean_density']

        # Linear regime detection
        # In the linear regime (no shell-crossing), ORIGAMI classifies all
        # particles as voids, so the classification is not physically meaningful.
        hdr.attrs['is_linear_regime'] = snapshot_info.get('is_linear_regime', False)
        hdr.attrs['linear_regime_threshold'] = snapshot_info.get('linear_regime_threshold', 0.99)

        # Jackknife info
        hdr.attrs['jackknife'] = hist_results.get('jackknife', False)
        if hist_results.get('jackknife', False):
            hdr.attrs['n_subboxes'] = hist_results['n_subboxes']
            hdr.attrs['n_subboxes_per_dim'] = hist_results['n_subboxes_per_dim']

        # Bin information
        bins = f.create_group('Bins')
        bins.create_dataset('edges', data=hist_results['bin_edges'])
        bins.create_dataset('centers', data=hist_results['bin_centers'])
        bins.create_dataset('widths', data=hist_results['bin_widths'])

        # Helper to save group data
        def save_group(grp, data):
            grp.attrs['n_particles'] = data['n_particles']
            grp.attrs['mean_overdensity'] = data['mean_overdensity']
            grp.attrs['median_overdensity'] = data['median_overdensity']
            grp.create_dataset('histogram', data=data['histogram'])
            grp.create_dataset('pdf', data=data['pdf'])
            # Save errors if available (jackknife mode)
            if 'histogram_error' in data:
                grp.create_dataset('histogram_error', data=data['histogram_error'])
            if 'pdf_error' in data:
                grp.create_dataset('pdf_error', data=data['pdf_error'])

        # All particles
        all_grp = f.create_group('All')
        save_group(all_grp, hist_results['all'])

        # Per-class data
        for class_name in ORIGAMI_CLASSES:
            key = class_name.lower()
            grp = f.create_group(class_name)
            save_group(grp, hist_results[key])

    print(f"    Saved successfully")


def create_figure(hist_results, snapshot_info, output_path):
    """Create 2x1 figure with histograms (top) and PDFs (bottom), shared x-axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_hist, ax_pdf) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'hspace': 0})

    bin_centers = hist_results['bin_centers']
    has_errors = hist_results.get('jackknife', False)

    # Top panel: Histograms (counts)
    # Plot all particles
    ax_hist.step(bin_centers, hist_results['all']['histogram'], where='mid',
                 color='black', linewidth=2, label='All', linestyle='--')

    # Plot each class
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        key = class_name.lower()
        ax_hist.step(bin_centers, hist_results[key]['histogram'], where='mid',
                     color=ORIGAMI_COLORS[i], linewidth=1.5, label=class_name)

    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_ylabel('Counts')
    ax_hist.legend(loc='upper right', fontsize=9)

    title = f"Overdensity Distribution: {snapshot_info['name']} ({snapshot_info['description']})"
    if has_errors:
        title += f" [Jackknife: {hist_results['n_subboxes']} sub-boxes]"
    ax_hist.set_title(title)
    ax_hist.grid(True, alpha=0.3, which='both')
    ax_hist.set_ylim(bottom=0.5)

    # Bottom panel: PDFs (probability density)
    # Plot all particles with error band if available
    pdf_all = hist_results['all']['pdf']
    ax_pdf.plot(bin_centers, pdf_all, color='black', linewidth=2, label='All', linestyle='--')
    if has_errors and 'pdf_error' in hist_results['all']:
        pdf_err = hist_results['all']['pdf_error']
        # Only fill where PDF is positive
        mask = pdf_all > 0
        ax_pdf.fill_between(bin_centers[mask],
                            np.maximum(pdf_all[mask] - pdf_err[mask], pdf_all[mask] * 0.01),
                            pdf_all[mask] + pdf_err[mask],
                            color='black', alpha=0.15)

    for i, class_name in enumerate(ORIGAMI_CLASSES):
        key = class_name.lower()
        pdf_class = hist_results[key]['pdf']
        # Only plot if there's data
        if pdf_class.max() > 0:
            ax_pdf.plot(bin_centers, pdf_class,
                        color=ORIGAMI_COLORS[i], linewidth=1.5, label=class_name)
            # Add error band if available
            if has_errors and 'pdf_error' in hist_results[key]:
                pdf_err = hist_results[key]['pdf_error']
                mask = pdf_class > 0
                ax_pdf.fill_between(bin_centers[mask],
                                    np.maximum(pdf_class[mask] - pdf_err[mask], pdf_class[mask] * 0.01),
                                    pdf_class[mask] + pdf_err[mask],
                                    color=ORIGAMI_COLORS[i], alpha=0.2)

    ax_pdf.set_xscale('log')
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel(r'$1 + \delta$')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.legend(loc='upper right', fontsize=9)
    ax_pdf.grid(True, alpha=0.3, which='both')

    # Add statistics annotation
    stats_text = (
        f"Mass fractions:\n"
        f"  Void: {hist_results['void']['n_particles'] / hist_results['all']['n_particles']:.1%}\n"
        f"  Wall: {hist_results['wall']['n_particles'] / hist_results['all']['n_particles']:.1%}\n"
        f"  Filament: {hist_results['filament']['n_particles'] / hist_results['all']['n_particles']:.1%}\n"
        f"  Halo: {hist_results['halo']['n_particles'] / hist_results['all']['n_particles']:.1%}"
    )
    ax_pdf.text(0.02, 0.02, stats_text, transform=ax_pdf.transAxes,
                fontsize=8, verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Remove top x-axis ticks (they're shared)
    ax_hist.tick_params(axis='x', which='both', bottom=True, labelbottom=False)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {output_path}")


def process_snapshot(snapshot_info, use_jackknife=False, n_subboxes_per_dim=2):
    """Process a single snapshot: compute ORIGAMI, density, and PDFs using unified pipeline.

    Args:
        snapshot_info: Dict with snapshot name and description
        use_jackknife: If True, use jackknife resampling for uncertainty estimation
        n_subboxes_per_dim: Sub-boxes per dimension for jackknife (2->8, 3->27)
    """
    print(f"\n{'='*70}")
    print(f"Processing {snapshot_info['name']} ({snapshot_info['description']})")
    if use_jackknife:
        print(f"  [Jackknife enabled: {n_subboxes_per_dim}^3 = {n_subboxes_per_dim**3} sub-boxes]")
    print('='*70)

    # Load data
    positions, particle_ids, box_size, redshift, scale_factor, grid_size = \
        load_snapshot(snapshot_info['name'])

    # Update snapshot info
    snapshot_info['scale_factor'] = scale_factor
    snapshot_info['redshift'] = redshift
    snapshot_info['box_size'] = box_size
    snapshot_info['grid_size'] = grid_size

    # Run unified pipeline with integrated PDF computation
    result, hist_results = run_origami_with_pdf(
        positions, particle_ids, box_size, grid_size,
        use_jackknife=use_jackknife,
        n_subboxes_per_dim=n_subboxes_per_dim,
        output_cells=256,
        n_samples=100,
        n_bins=100
    )

    # Store linear regime info in snapshot_info for saving
    snapshot_info['is_linear_regime'] = result.is_linear_regime
    snapshot_info['linear_regime_threshold'] = result.linear_regime_threshold

    if result.is_linear_regime:
        print(f"  WARNING: Field is in LINEAR REGIME ({result.f_void*100:.1f}% void)")
        print(f"           ORIGAMI classification may not be physically meaningful")

    # Save to HDF5
    suffix = "_jackknife" if use_jackknife else ""
    h5_path = SCRIPT_DIR / f"{snapshot_info['name']}_overdensity_pdf{suffix}.h5"
    save_to_hdf5(h5_path, hist_results, snapshot_info)

    # Create figure
    fig_path = SCRIPT_DIR / f"{snapshot_info['name']}_overdensity_pdf{suffix}.png"
    create_figure(hist_results, snapshot_info, fig_path)

    return hist_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Overdensity PDF Analysis with ORIGAMI Morphology Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python overdensity_pdf_origami.py
    python overdensity_pdf_origami.py --jackknife
    python overdensity_pdf_origami.py --jackknife --n-subboxes 3
        """
    )
    parser.add_argument('--jackknife', action='store_true',
                        help='Enable jackknife resampling for uncertainty estimation')
    parser.add_argument('--n-subboxes', type=int, default=2, dest='n_subboxes_per_dim',
                        help='Sub-boxes per dimension for jackknife (default: 2, giving 8 sub-boxes)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("Overdensity PDF Analysis with ORIGAMI Morphology")
    if args.jackknife:
        print(f"  [Jackknife enabled: {args.n_subboxes_per_dim}^3 = {args.n_subboxes_per_dim**3} sub-boxes]")
    print("="*70)

    all_results = {}

    for snap_info in SNAPSHOTS:
        results = process_snapshot(
            snap_info.copy(),
            use_jackknife=args.jackknife,
            n_subboxes_per_dim=args.n_subboxes_per_dim
        )
        all_results[snap_info['name']] = results

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

    # Summary
    print("\nSummary:")
    for snap_name, results in all_results.items():
        print(f"\n  {snap_name}:")
        print(f"    Total particles: {results['all']['n_particles']:,}")
        print(f"    Mean overdensity: {results['all']['mean_overdensity']:.4f}")
        if results.get('jackknife', False):
            print(f"    Jackknife sub-boxes: {results['n_subboxes']}")
        for class_name in ORIGAMI_CLASSES:
            key = class_name.lower()
            n = results[key]['n_particles']
            frac = n / results['all']['n_particles']
            mean_od = results[key]['mean_overdensity']
            print(f"    {class_name}: {n:,} ({frac:.1%}), <1+delta>={mean_od:.2f}")


if __name__ == "__main__":
    main()
