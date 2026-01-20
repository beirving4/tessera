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

    with h5py.File(snapshot_path, 'r') as f:
        positions = np.ascontiguousarray(f['PartType1/Coordinates'][:], dtype=np.float64)
        particle_ids = np.ascontiguousarray(f['PartType1/ParticleIDs'][:], dtype=np.int64)
        box_size = float(f['Header'].attrs['BoxSize'])
        redshift = float(f['Header'].attrs['Redshift'])
        scale_factor = float(f['Header'].attrs['Time'])

    grid_size = int(round(len(positions) ** (1/3)))
    print(f"  Loaded {len(positions):,} particles (grid: {grid_size}^3)")
    print(f"  Scale factor: {scale_factor:.4f}, Redshift: {redshift:.4f}")

    return positions, particle_ids, box_size, redshift, scale_factor, grid_size


def detect_id_ordering(positions, particle_ids, box_size, grid_size):
    """Detect whether particle IDs use x-major or z-major ordering.

    This method uses displacement correlations between neighboring IDs to
    robustly detect the ordering even when particles have moved significantly
    and wrapped around periodic boundaries.
    """
    grid_size = int(grid_size)
    n_particles = len(positions)

    particle_ids_int = particle_ids.astype(np.int64)
    id_to_idx = {int(pid): i for i, pid in enumerate(particle_ids_int)}

    def get_pos(pid):
        if pid in id_to_idx:
            return positions[id_to_idx[pid]]
        return None

    def periodic_diff(p1, p2, box):
        diff = p1 - p2
        diff = diff - box * np.round(diff / box)
        return diff

    x_major_score = 0.0
    z_major_score = 0.0
    n_valid = 0

    spacing = box_size / grid_size
    test_ids = [1, grid_size//4, grid_size//2, grid_size*10, grid_size*100]

    for base_id in test_ids:
        if base_id < 1 or base_id >= n_particles:
            continue

        p1 = get_pos(base_id)
        p2 = get_pos(base_id + 1)

        if p1 is None or p2 is None:
            continue

        diff = periodic_diff(p2, p1, box_size)

        z_major_expected = np.array([0, 0, spacing])
        x_major_expected = np.array([spacing, 0, 0])

        z_major_score += np.linalg.norm(diff - z_major_expected)
        x_major_score += np.linalg.norm(diff - x_major_expected)
        n_valid += 1

    for base_id in test_ids:
        next_y_id = base_id + grid_size

        p1 = get_pos(base_id)
        p2 = get_pos(next_y_id)

        if p1 is None or p2 is None:
            continue

        diff = periodic_diff(p2, p1, box_size)
        z_major_score += abs(abs(diff[0]) - 0)
        x_major_score += abs(abs(diff[2]) - 0)

    if n_valid == 0:
        return 'z-major'

    return 'x-major' if x_major_score < z_major_score else 'z-major'


def sort_to_lagrangian(positions, particle_ids, grid_size, id_ordering):
    """Sort particles to x-major Lagrangian order."""
    n_particles = len(positions)
    sorted_positions = np.zeros((n_particles, 3), dtype=np.float64)

    # Ensure integer types to prevent float indexing errors
    grid_size = int(grid_size)
    particle_ids = particle_ids.astype(np.int64)

    if id_ordering == 'x-major':
        sorted_positions[particle_ids - 1] = positions
    else:  # z-major
        for i in range(n_particles):
            idx = int(particle_ids[i]) - 1
            z = idx % grid_size
            y = (idx // grid_size) % grid_size
            x = idx // (grid_size * grid_size)
            xmajor_idx = x + y * grid_size + z * grid_size * grid_size
            sorted_positions[xmajor_idx] = positions[i]

    return np.ascontiguousarray(sorted_positions)


def compute_origami(sorted_positions, box_size, grid_size):
    """Compute ORIGAMI morphology classification."""
    print("  Computing ORIGAMI morphology...")

    config = ts.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    result = ts.origami.compute_morphology(sorted_positions, config)

    print(f"    Void:     {result.n_void:>10,} ({result.f_void:>6.2%})")
    print(f"    Wall:     {result.n_wall:>10,} ({result.f_wall:>6.2%})")
    print(f"    Filament: {result.n_filament:>10,} ({result.f_filament:>6.2%})")
    print(f"    Halo:     {result.n_halo:>10,} ({result.f_halo:>6.2%})")

    return result


def compute_density_at_particles(sorted_positions, box_size, grid_size, origami_result,
                                  output_cells=256, n_samples=100):
    """Compute 3D density field and sample at particle positions."""
    print(f"  Computing density field ({output_cells}^3)...")

    config = ts.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.output_cells = output_cells
    config.box_size = float(box_size)
    config.particle_mass = 1.0
    config.n_samples = n_samples
    config.n_threads = 1
    config.periodic = True

    density_result = ts.density.compute_tetra_density_3d(sorted_positions, config)

    density_3d = np.array(density_result.density).reshape(
        output_cells, output_cells, output_cells
    )

    print(f"    Density range: [{density_3d.min():.4e}, {density_3d.max():.4e}]")
    print(f"    Mean density: {density_3d.mean():.4e}")

    # Sample at particle positions
    print("  Sampling density at particle positions...")
    ts.origami.sample_density_at_particles(
        density_3d, sorted_positions, box_size, origami_result
    )

    return density_3d


def compute_overdensity_histograms(particle_density, morphology, n_bins=100):
    """Compute overdensity histograms for all particles and by ORIGAMI class.

    Returns:
        dict with bin_edges, bin_centers, histograms (counts), and PDFs (normalized)
    """
    # Normalize to overdensity (1 + delta)
    mean_density = particle_density.mean()
    overdensity = particle_density / mean_density

    # Use log-spaced bins
    log_min = np.log10(max(overdensity.min(), 1e-3))
    log_max = np.log10(overdensity.max())
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'mean_density': mean_density,
        'jackknife': False,
    }

    # Full field histogram
    hist_all, _ = np.histogram(overdensity, bins=bin_edges)
    pdf_all = hist_all / (hist_all.sum() * bin_widths)  # Normalize to PDF

    results['all'] = {
        'histogram': hist_all,
        'pdf': pdf_all,
        'n_particles': len(overdensity),
        'mean_overdensity': overdensity.mean(),
        'median_overdensity': np.median(overdensity),
    }

    # Per-class histograms
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        mask = morphology == i
        if mask.sum() > 0:
            od_class = overdensity[mask]
            hist_class, _ = np.histogram(od_class, bins=bin_edges)
            pdf_class = hist_class / (hist_class.sum() * bin_widths) if hist_class.sum() > 0 else np.zeros_like(hist_class, dtype=float)

            results[class_name.lower()] = {
                'histogram': hist_class,
                'pdf': pdf_class,
                'n_particles': int(mask.sum()),
                'mean_overdensity': od_class.mean(),
                'median_overdensity': np.median(od_class),
            }
        else:
            results[class_name.lower()] = {
                'histogram': np.zeros(n_bins, dtype=int),
                'pdf': np.zeros(n_bins, dtype=float),
                'n_particles': 0,
                'mean_overdensity': 0.0,
                'median_overdensity': 0.0,
            }

    return results


def compute_overdensity_histograms_jackknife(positions, particle_density, morphology,
                                              box_size, n_bins=100, n_subboxes_per_dim=2):
    """Compute overdensity histograms with jackknife uncertainty estimation.

    Uses spatial sub-box resampling with leave-one-out jackknife to estimate
    uncertainties on both histogram counts and PDFs.

    Args:
        positions: Particle positions (Eulerian), shape (n_particles, 3)
        particle_density: Density at each particle position
        morphology: ORIGAMI morphology classification (0-3)
        box_size: Simulation box size
        n_bins: Number of histogram bins
        n_subboxes_per_dim: Sub-boxes per dimension (2 -> 8, 3 -> 27 total)

    Returns:
        dict with bin_edges, bin_centers, histograms, PDFs, and errors
    """
    # Normalize to overdensity (1 + delta)
    mean_density = particle_density.mean()
    overdensity = particle_density / mean_density

    # Determine range for log bins
    od_positive = overdensity[overdensity > 0]
    range_min = max(od_positive.min(), 1e-3)
    range_max = overdensity.max() * 1.001

    print(f"    Using jackknife with {n_subboxes_per_dim}^3 = {n_subboxes_per_dim**3} sub-boxes")

    # Use C++ jackknife conditional histogram
    jk_result = ts.stats.compute_jackknife_conditional_histogram(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(overdensity, dtype=np.float64),
        np.ascontiguousarray(morphology, dtype=np.uint8),
        box_size,
        n_bins,
        range_min,
        range_max,
        log_bins=True,
        n_subboxes_per_dim=n_subboxes_per_dim,
        n_threads=0
    )

    # Build results dict
    bin_edges = np.array(jk_result.all.bin_edges)
    bin_centers = np.array(jk_result.all.bin_centers)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'mean_density': mean_density,
        'jackknife': True,
        'n_subboxes': jk_result.all.n_subboxes,
        'n_subboxes_per_dim': n_subboxes_per_dim,
    }

    # Helper to extract results from JackknifeHistogramResult
    def extract_jk_results(jk_hist, od_values):
        return {
            'histogram': np.array(jk_hist.global_counts),
            'pdf': np.array(jk_hist.global_pdf),
            'histogram_error': np.array(jk_hist.counts_error),
            'pdf_error': np.array(jk_hist.pdf_error),
            'n_particles': int(jk_hist.global_n_total),
            'mean_overdensity': float(od_values.mean()) if len(od_values) > 0 else 0.0,
            'median_overdensity': float(np.median(od_values)) if len(od_values) > 0 else 0.0,
        }

    # All particles
    results['all'] = extract_jk_results(jk_result.all, overdensity)

    # Per-class results
    class_map = {
        'void': jk_result.void_class,
        'wall': jk_result.wall_class,
        'filament': jk_result.filament_class,
        'halo': jk_result.halo_class,
    }

    for class_name in ORIGAMI_CLASSES:
        key = class_name.lower()
        class_idx = ORIGAMI_CLASSES.index(class_name)
        mask = morphology == class_idx
        od_class = overdensity[mask] if mask.sum() > 0 else np.array([])
        results[key] = extract_jk_results(class_map[key], od_class)

    return results


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
    """Process a single snapshot: compute ORIGAMI, density, and PDFs.

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

    # Sort to Lagrangian order
    print("  Sorting to Lagrangian order...")
    id_ordering = detect_id_ordering(positions, particle_ids, box_size, grid_size)
    print(f"    Detected ID ordering: {id_ordering}")
    sorted_positions = sort_to_lagrangian(positions, particle_ids, grid_size, id_ordering)

    # Compute ORIGAMI morphology
    origami_result = compute_origami(sorted_positions, box_size, grid_size)

    # Store linear regime info in snapshot_info for saving
    snapshot_info['is_linear_regime'] = origami_result.is_linear_regime
    snapshot_info['linear_regime_threshold'] = origami_result.linear_regime_threshold

    if origami_result.is_linear_regime:
        print(f"  WARNING: Field is in LINEAR REGIME ({origami_result.f_void*100:.1f}% void)")
        print(f"           ORIGAMI classification may not be physically meaningful")

    # Compute density and sample at particles
    density_3d = compute_density_at_particles(
        sorted_positions, box_size, grid_size, origami_result,
        output_cells=256, n_samples=100
    )

    # Get particle densities and morphology
    particle_density = np.array(origami_result.particle_density)
    morphology = np.array(origami_result.morphology)

    # Compute histograms and PDFs
    print("  Computing overdensity histograms...")
    if use_jackknife:
        # Use Eulerian positions (sorted_positions) for sub-box assignment
        hist_results = compute_overdensity_histograms_jackknife(
            sorted_positions, particle_density, morphology, box_size,
            n_bins=100, n_subboxes_per_dim=n_subboxes_per_dim
        )
    else:
        hist_results = compute_overdensity_histograms(particle_density, morphology, n_bins=100)

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
