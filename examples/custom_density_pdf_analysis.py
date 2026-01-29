#!/usr/bin/env python3
"""
Custom Density PDF Analysis from ORIGAMI Pipeline Output

This example demonstrates how to:
1. Run the ORIGAMI pipeline to extract per-particle density and morphology
2. Save the raw data to HDF5 for later analysis
3. Compute custom histograms/PDFs with different binning
4. Compute jackknife uncertainties with different sub-box configurations
5. Compare different binning strategies

This approach gives you full control over the PDF computation while
leveraging the optimized C++ pipeline for density estimation.

Usage:
    python custom_density_pdf_analysis.py /path/to/snapshot.hdf5 [options]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import h5py
import argparse
from pathlib import Path
from datetime import datetime
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add tessera to path if needed
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


# =============================================================================
# ORIGAMI Pipeline: Extract Raw Density Data
# =============================================================================

def run_origami_density_extraction(
    snapshot_path: Path,
    n_threads: int = 0,
) -> dict:
    """
    Run ORIGAMI pipeline to extract per-particle density and morphology.

    This skips the internal PDF computation (pdf_n_bins=0) for efficiency
    when you want to compute custom PDFs later.

    Args:
        snapshot_path: Path to GADGET-4 HDF5 snapshot
        n_threads: Number of threads (0=auto)

    Returns:
        Dictionary with positions, density, morphology, and metadata
    """
    print(f"\n{'='*70}")
    print(f"Extracting density data from: {snapshot_path.name}")
    print(f"{'='*70}")

    # Load snapshot
    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])
        redshift = float(f['Header'].attrs['Redshift'])

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    print(f"  Particles: {n_particles:,} ({grid_size}^3)")
    print(f"  Box size: {box_size:.1f} Mpc/h")
    print(f"  Scale factor: {scale_factor:.4f}, z={redshift:.2f}")

    # Ensure contiguous arrays
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    particle_ids = np.ascontiguousarray(particle_ids, dtype=np.int64)

    # Configure pipeline: density extraction only, skip PDF computation
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.density_output_cells = grid_size
    config.sample_density_at_particles = True  # Get per-particle density
    config.pdf_n_bins = 0                       # Skip internal PDF computation
    config.n_threads = n_threads

    print(f"\n  Running ORIGAMI pipeline (density extraction mode)...")
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    print(f"  Pipeline completed in {result.total_time_ms:.1f} ms")
    print(f"  Detected ID ordering: {ts.origami.id_ordering_name(result.detected_ordering)}")

    # Extract arrays
    density = np.array(result.particle_density)
    morphology = np.array(result.morphology)

    # Note: positions are now sorted to Lagrangian order
    print(f"\n  ORIGAMI fractions:")
    print(f"    Void:     {result.f_void:>6.2%}")
    print(f"    Wall:     {result.f_wall:>6.2%}")
    print(f"    Filament: {result.f_filament:>6.2%}")
    print(f"    Halo:     {result.f_halo:>6.2%}")

    print(f"\n  Density statistics:")
    print(f"    Mean density: {result.mean_density:.4e}")
    print(f"    Density range: [{density.min():.4e}, {density.max():.4e}]")

    return {
        'positions': positions,  # Now in Lagrangian order
        'density': density,
        'morphology': morphology,
        'box_size': box_size,
        'mean_density': result.mean_density,
        'scale_factor': scale_factor,
        'redshift': redshift,
        'grid_size': grid_size,
        'n_particles': n_particles,
        'f_void': result.f_void,
        'f_wall': result.f_wall,
        'f_filament': result.f_filament,
        'f_halo': result.f_halo,
    }


# =============================================================================
# Save/Load Raw Data
# =============================================================================

def save_density_data(data: dict, output_path: Path) -> None:
    """
    Save extracted density data to HDF5 for later custom analysis.

    File structure:
        /Header (attrs: box_size, mean_density, scale_factor, redshift, ...)
        /Data/positions  (N, 3) float64
        /Data/density    (N,)   float64
        /Data/morphology (N,)   uint8
    """
    print(f"\n  Saving density data to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Header with metadata
        hdr = f.create_group('Header')
        hdr.attrs['created'] = datetime.now().isoformat()
        hdr.attrs['box_size'] = data['box_size']
        hdr.attrs['mean_density'] = data['mean_density']
        hdr.attrs['scale_factor'] = data['scale_factor']
        hdr.attrs['redshift'] = data['redshift']
        hdr.attrs['grid_size'] = data['grid_size']
        hdr.attrs['n_particles'] = data['n_particles']
        hdr.attrs['f_void'] = data['f_void']
        hdr.attrs['f_wall'] = data['f_wall']
        hdr.attrs['f_filament'] = data['f_filament']
        hdr.attrs['f_halo'] = data['f_halo']

        # Data arrays with compression
        grp = f.create_group('Data')
        grp.create_dataset('positions', data=data['positions'],
                          compression='gzip', compression_opts=4)
        grp.create_dataset('density', data=data['density'],
                          compression='gzip', compression_opts=4)
        grp.create_dataset('morphology', data=data['morphology'],
                          compression='gzip', compression_opts=4)

    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    File size: {file_size_mb:.1f} MB")


def load_density_data(input_path: Path) -> dict:
    """Load previously saved density data from HDF5."""
    print(f"\n  Loading density data from: {input_path}")

    with h5py.File(input_path, 'r') as f:
        data = {
            'positions': f['Data/positions'][:],
            'density': f['Data/density'][:],
            'morphology': f['Data/morphology'][:],
            'box_size': f['Header'].attrs['box_size'],
            'mean_density': f['Header'].attrs['mean_density'],
            'scale_factor': f['Header'].attrs['scale_factor'],
            'redshift': f['Header'].attrs['redshift'],
            'grid_size': f['Header'].attrs['grid_size'],
            'n_particles': f['Header'].attrs['n_particles'],
            'f_void': f['Header'].attrs['f_void'],
            'f_wall': f['Header'].attrs['f_wall'],
            'f_filament': f['Header'].attrs['f_filament'],
            'f_halo': f['Header'].attrs['f_halo'],
        }

    print(f"    Loaded {data['n_particles']:,} particles")
    return data


# =============================================================================
# Custom PDF Computation
# =============================================================================

def compute_density_pdf(
    positions: np.ndarray,
    density: np.ndarray,
    morphology: np.ndarray,
    mean_density: float,
    box_size: float,
    n_bins: int = 100,
    log_bins: bool = True,
    od_range: tuple = (1e-3, 1e5),
    jackknife: bool = False,
    n_subboxes_per_dim: int = 2,
) -> dict:
    """
    Compute overdensity histograms and PDFs per morphology class.

    Args:
        positions: (N, 3) particle positions
        density: (N,) density at each particle
        morphology: (N,) morphology class (0=void, 1=wall, 2=filament, 3=halo)
        mean_density: mean density for overdensity normalization
        box_size: simulation box size (required for jackknife)
        n_bins: number of histogram bins
        log_bins: if True, use logarithmic bins
        od_range: (min, max) overdensity range for bins
        jackknife: if True, compute jackknife uncertainties
        n_subboxes_per_dim: sub-boxes per dimension for jackknife (2->8, 3->27, 4->64)

    Returns:
        dict with bin_edges, bin_centers, and per-class histogram/pdf data
    """
    # Compute overdensity (1 + delta)
    overdensity = density / mean_density

    # Create bins
    if log_bins:
        bin_edges = np.logspace(np.log10(od_range[0]), np.log10(od_range[1]), n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
    else:
        bin_edges = np.linspace(od_range[0], od_range[1], n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Arithmetic mean

    bin_widths = bin_edges[1:] - bin_edges[:-1]

    results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'n_bins': n_bins,
        'log_bins': log_bins,
        'od_range': od_range,
        'jackknife': jackknife,
    }

    if jackknife:
        results['n_subboxes_per_dim'] = n_subboxes_per_dim
        results['n_subboxes'] = n_subboxes_per_dim ** 3

    # Classes: all + 4 morphology types
    class_configs = [
        ('all', np.ones(len(overdensity), dtype=bool)),
        ('void', morphology == 0),
        ('wall', morphology == 1),
        ('filament', morphology == 2),
        ('halo', morphology == 3),
    ]

    for name, mask in class_configs:
        od_class = overdensity[mask]
        n_particles = len(od_class)

        # Histogram (counts)
        hist, _ = np.histogram(od_class, bins=bin_edges)

        # PDF: normalized so integral over overdensity = 1
        # PDF(x) = counts / (N * dx) where dx is bin width
        if n_particles > 0:
            pdf = hist.astype(float) / (n_particles * bin_widths)
        else:
            pdf = np.zeros(n_bins, dtype=float)

        results[name] = {
            'histogram': hist,
            'pdf': pdf,
            'n_particles': n_particles,
            'mean_overdensity': float(od_class.mean()) if n_particles > 0 else 0.0,
            'median_overdensity': float(np.median(od_class)) if n_particles > 0 else 0.0,
        }

        # Jackknife uncertainties
        if jackknife:
            pos_class = positions[mask]
            pdf_samples = _jackknife_resample(
                pos_class, od_class, bin_edges, bin_widths,
                n_subboxes_per_dim, box_size
            )

            # Jackknife variance estimator: (n-1)/n * sum((x_i - x_mean)^2)
            n_subboxes = n_subboxes_per_dim ** 3
            pdf_mean = pdf_samples.mean(axis=0)
            pdf_var = ((n_subboxes - 1) / n_subboxes) * np.sum((pdf_samples - pdf_mean)**2, axis=0)

            results[name]['pdf_error'] = np.sqrt(pdf_var)
            results[name]['pdf_samples'] = pdf_samples  # Keep all samples for diagnostics

    return results


def _jackknife_resample(
    positions: np.ndarray,
    overdensity: np.ndarray,
    bin_edges: np.ndarray,
    bin_widths: np.ndarray,
    n_subboxes_per_dim: int,
    box_size: float,
) -> np.ndarray:
    """
    Compute jackknife PDF samples by leaving out one sub-box at a time.

    The box is divided into n_subboxes_per_dim^3 equal sub-boxes.
    Each jackknife sample excludes particles from one sub-box.

    Args:
        positions: (N, 3) particle positions
        overdensity: (N,) overdensity values
        bin_edges: histogram bin edges
        bin_widths: histogram bin widths
        n_subboxes_per_dim: sub-boxes per dimension
        box_size: simulation box size

    Returns:
        (n_subboxes, n_bins) array of PDF samples
    """
    n_subboxes = n_subboxes_per_dim ** 3
    subbox_size = box_size / n_subboxes_per_dim

    # Assign each particle to a sub-box index
    # Index = ix * n^2 + iy * n + iz
    ix = np.clip((positions[:, 0] / subbox_size).astype(int), 0, n_subboxes_per_dim - 1)
    iy = np.clip((positions[:, 1] / subbox_size).astype(int), 0, n_subboxes_per_dim - 1)
    iz = np.clip((positions[:, 2] / subbox_size).astype(int), 0, n_subboxes_per_dim - 1)
    subbox_idx = ix * n_subboxes_per_dim**2 + iy * n_subboxes_per_dim + iz

    n_bins = len(bin_edges) - 1
    pdf_samples = np.zeros((n_subboxes, n_bins))

    for i in range(n_subboxes):
        # Leave out sub-box i
        mask = subbox_idx != i
        od_subset = overdensity[mask]
        n_subset = len(od_subset)

        if n_subset > 0:
            hist, _ = np.histogram(od_subset, bins=bin_edges)
            pdf_samples[i] = hist.astype(float) / (n_subset * bin_widths)

    return pdf_samples


# =============================================================================
# Visualization
# =============================================================================

ORIGAMI_COLORS = {
    'void': '#3b82f6',
    'wall': '#22c55e',
    'filament': '#f59e0b',
    'halo': '#ef4444',
    'all': '#374151',
}


def plot_pdf_comparison(
    results_list: list[dict],
    labels: list[str],
    output_path: Path,
    title: str = None,
) -> None:
    """
    Compare PDFs from different binning/jackknife configurations.

    Args:
        results_list: List of results dicts from compute_density_pdf()
        labels: Labels for each configuration
        output_path: Path to save figure
        title: Optional figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    classes = ['void', 'wall', 'filament', 'halo']

    for ax, cls_name in zip(axes, classes):
        for results, label in zip(results_list, labels):
            bin_centers = results['bin_centers']
            pdf = results[cls_name]['pdf']

            line, = ax.plot(bin_centers, pdf, label=label, linewidth=1.5)

            # Add error band if available
            if 'pdf_error' in results[cls_name]:
                pdf_err = results[cls_name]['pdf_error']
                ax.fill_between(
                    bin_centers,
                    np.maximum(pdf - pdf_err, pdf * 0.01),  # Avoid negative on log scale
                    pdf + pdf_err,
                    alpha=0.2, color=line.get_color()
                )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$1 + \delta$')
        ax.set_ylabel('PDF')
        ax.set_title(cls_name.capitalize(), fontweight='bold', color=ORIGAMI_COLORS[cls_name])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved comparison figure: {output_path}")


def plot_single_pdf(
    results: dict,
    output_path: Path,
    title: str = None,
) -> None:
    """
    Plot PDFs for all morphology classes on a single figure.
    """
    fig, (ax_hist, ax_pdf) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    bin_centers = results['bin_centers']

    # Top: Histograms
    ax_hist.step(bin_centers, results['all']['histogram'], where='mid',
                 color=ORIGAMI_COLORS['all'], linewidth=2, linestyle='--', label='All')
    for cls_name in ['void', 'wall', 'filament', 'halo']:
        ax_hist.step(bin_centers, results[cls_name]['histogram'], where='mid',
                     color=ORIGAMI_COLORS[cls_name], linewidth=1.5, label=cls_name.capitalize())

    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_ylabel('Counts')
    ax_hist.legend(loc='upper right', fontsize=9)
    ax_hist.grid(True, alpha=0.3, which='both')
    if title:
        ax_hist.set_title(title, fontweight='bold')

    # Bottom: PDFs with error bands
    ax_pdf.plot(bin_centers, results['all']['pdf'],
                color=ORIGAMI_COLORS['all'], linewidth=2, linestyle='--', label='All')

    for cls_name in ['void', 'wall', 'filament', 'halo']:
        pdf = results[cls_name]['pdf']
        ax_pdf.plot(bin_centers, pdf, color=ORIGAMI_COLORS[cls_name],
                    linewidth=1.5, label=cls_name.capitalize())

        if 'pdf_error' in results[cls_name]:
            pdf_err = results[cls_name]['pdf_error']
            mask = pdf > 0
            ax_pdf.fill_between(
                bin_centers[mask],
                np.maximum(pdf[mask] - pdf_err[mask], pdf[mask] * 0.01),
                pdf[mask] + pdf_err[mask],
                color=ORIGAMI_COLORS[cls_name], alpha=0.2
            )

    ax_pdf.set_xscale('log')
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel(r'$1 + \delta$')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.legend(loc='upper right', fontsize=9)
    ax_pdf.grid(True, alpha=0.3, which='both')
    ax_pdf.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved PDF figure: {output_path}")


def print_pdf_summary(results: dict, label: str = "") -> None:
    """Print summary statistics for PDF results."""
    print(f"\n{'='*70}")
    print(f"PDF Summary{': ' + label if label else ''}")
    print(f"{'='*70}")
    print(f"  Bins: {results['n_bins']}, Log: {results['log_bins']}, Range: {results['od_range']}")
    if results['jackknife']:
        print(f"  Jackknife: {results['n_subboxes']} sub-boxes ({results['n_subboxes_per_dim']}^3)")

    print(f"\n  {'Class':>10}  {'N':>12}  {'<1+δ>':>8}  {'median':>8}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}")
    for cls_name in ['all', 'void', 'wall', 'filament', 'halo']:
        d = results[cls_name]
        print(f"  {cls_name:>10}: {d['n_particles']:>12,}  {d['mean_overdensity']:>8.2f}  {d['median_overdensity']:>8.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Custom density PDF analysis from ORIGAMI pipeline output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract density data and compute custom PDFs
  python custom_density_pdf_analysis.py snapshot_074.hdf5

  # Load previously saved data and recompute with different settings
  python custom_density_pdf_analysis.py --load density_data_074.h5 --n-bins 200

  # Compare different jackknife configurations
  python custom_density_pdf_analysis.py snapshot_074.hdf5 --compare-jackknife
        """
    )

    parser.add_argument('input', type=str,
                        help='Path to GADGET-4 snapshot or saved density data HDF5')
    parser.add_argument('--load', action='store_true',
                        help='Input is a saved density data file (not a snapshot)')
    parser.add_argument('--save-data', type=str, default=None,
                        help='Save extracted density data to this path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: same as input)')

    # PDF computation options
    parser.add_argument('--n-bins', type=int, default=100,
                        help='Number of histogram bins (default: 100)')
    parser.add_argument('--linear-bins', action='store_true',
                        help='Use linear bins instead of logarithmic')
    parser.add_argument('--od-min', type=float, default=1e-3,
                        help='Minimum overdensity for bins (default: 1e-3)')
    parser.add_argument('--od-max', type=float, default=1e5,
                        help='Maximum overdensity for bins (default: 1e5)')

    # Jackknife options
    parser.add_argument('--jackknife', action='store_true',
                        help='Compute jackknife uncertainties')
    parser.add_argument('--n-subboxes', type=int, default=2,
                        help='Sub-boxes per dimension for jackknife (default: 2, giving 8 sub-boxes)')

    # Comparison mode
    parser.add_argument('--compare-jackknife', action='store_true',
                        help='Compare different jackknife configurations (2^3, 3^3, 4^3)')
    parser.add_argument('--compare-binning', action='store_true',
                        help='Compare different binning (50, 100, 200 bins)')

    parser.add_argument('--n-threads', type=int, default=0,
                        help='Number of threads for ORIGAMI (0=auto)')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or extract density data
    if args.load:
        data = load_density_data(input_path)
        base_name = input_path.stem.replace('_density_data', '')
    else:
        data = run_origami_density_extraction(input_path, n_threads=args.n_threads)
        base_name = input_path.stem

        # Save data if requested
        if args.save_data:
            save_path = Path(args.save_data)
        else:
            save_path = output_dir / f"{base_name}_density_data.h5"
        save_density_data(data, save_path)

    # Comparison modes
    if args.compare_jackknife:
        print(f"\n{'='*70}")
        print("Comparing Jackknife Configurations")
        print(f"{'='*70}")

        results_list = []
        labels = []

        for n_sub in [2, 3, 4]:
            print(f"\n  Computing with {n_sub}^3 = {n_sub**3} sub-boxes...")
            results = compute_density_pdf(
                data['positions'], data['density'], data['morphology'],
                data['mean_density'], data['box_size'],
                n_bins=args.n_bins,
                log_bins=not args.linear_bins,
                od_range=(args.od_min, args.od_max),
                jackknife=True,
                n_subboxes_per_dim=n_sub,
            )
            results_list.append(results)
            labels.append(f"{n_sub}³ = {n_sub**3} sub-boxes")

        plot_pdf_comparison(
            results_list, labels,
            output_dir / f"{base_name}_jackknife_comparison.png",
            title=f"Jackknife Comparison (a={data['scale_factor']:.2f})"
        )

        for results, label in zip(results_list, labels):
            print_pdf_summary(results, label)

    elif args.compare_binning:
        print(f"\n{'='*70}")
        print("Comparing Binning Configurations")
        print(f"{'='*70}")

        results_list = []
        labels = []

        for n_bins in [50, 100, 200]:
            print(f"\n  Computing with {n_bins} bins...")
            results = compute_density_pdf(
                data['positions'], data['density'], data['morphology'],
                data['mean_density'], data['box_size'],
                n_bins=n_bins,
                log_bins=not args.linear_bins,
                od_range=(args.od_min, args.od_max),
                jackknife=args.jackknife,
                n_subboxes_per_dim=args.n_subboxes,
            )
            results_list.append(results)
            labels.append(f"{n_bins} bins")

        plot_pdf_comparison(
            results_list, labels,
            output_dir / f"{base_name}_binning_comparison.png",
            title=f"Binning Comparison (a={data['scale_factor']:.2f})"
        )

        for results, label in zip(results_list, labels):
            print_pdf_summary(results, label)

    else:
        # Single PDF computation
        print(f"\n{'='*70}")
        print("Computing Custom PDF")
        print(f"{'='*70}")

        results = compute_density_pdf(
            data['positions'], data['density'], data['morphology'],
            data['mean_density'], data['box_size'],
            n_bins=args.n_bins,
            log_bins=not args.linear_bins,
            od_range=(args.od_min, args.od_max),
            jackknife=args.jackknife,
            n_subboxes_per_dim=args.n_subboxes,
        )

        print_pdf_summary(results)

        suffix = "_jackknife" if args.jackknife else ""
        plot_single_pdf(
            results,
            output_dir / f"{base_name}_custom_pdf{suffix}.png",
            title=f"Overdensity PDF (a={data['scale_factor']:.2f}, {args.n_bins} bins)"
        )

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
