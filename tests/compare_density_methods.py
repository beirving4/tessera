#!/usr/bin/env python3
"""
Compare Direct Particle Density vs Grid-Based Monte Carlo Density.

This script compares the new memory-efficient direct particle density method
against the traditional grid-based Monte Carlo approach for computing
overdensity PDFs per ORIGAMI morphology class.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add tessera build directory
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'build'))

try:
    import _tessera as ts
except ImportError:
    raise ImportError("Could not import _tessera. Build the project first.")


# ORIGAMI class names and colors
ORIGAMI_CLASSES = ['Void', 'Wall', 'Filament', 'Halo']
ORIGAMI_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_snapshot(snapshot_path: Path) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Load particle positions, IDs, and metadata from snapshot."""
    print(f"  Loading snapshot: {snapshot_path}")

    with h5py.File(snapshot_path, 'r') as f:
        positions = f['PartType1/Coordinates'][:]
        particle_ids = f['PartType1/ParticleIDs'][:]
        box_size = f['Header'].attrs['BoxSize']
        scale_factor = f['Header'].attrs['Time']
        redshift = f['Header'].attrs['Redshift']

    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    print(f"    Particles: {n_particles:,} ({grid_size}^3)")
    print(f"    Box size: {box_size:.1f} Mpc/h")
    print(f"    Scale factor: {scale_factor:.4f}, z={redshift:.4f}")

    return (
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(particle_ids, dtype=np.int64),
        float(box_size),
        float(scale_factor),
        float(redshift),
    )


def load_existing_pdf_results(results_path: Path) -> dict:
    """Load existing grid-based PDF results from HDF5."""
    print(f"  Loading existing PDF results: {results_path}")

    results = {}
    with h5py.File(results_path, 'r') as f:
        # Bins
        results['bin_edges'] = f['Bins/edges'][:]
        results['bin_centers'] = f['Bins/centers'][:]
        results['bin_widths'] = f['Bins/widths'][:]

        # Header info
        results['mean_density'] = f['Header'].attrs['mean_density']
        results['scale_factor'] = f['Header'].attrs['scale_factor']
        results['redshift'] = f['Header'].attrs['redshift']

        # Load data for each class
        for class_name in ['All'] + ORIGAMI_CLASSES:
            grp = f[class_name]
            results[class_name.lower()] = {
                'histogram': grp['histogram'][:],
                'pdf': grp['pdf'][:],
                'n_particles': grp.attrs['n_particles'],
                'mean_overdensity': grp.attrs['mean_overdensity'],
                'median_overdensity': grp.attrs['median_overdensity'],
            }
            if 'pdf_error' in grp:
                results[class_name.lower()]['pdf_error'] = grp['pdf_error'][:]

    return results


def compute_direct_density_pdfs(
    positions: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
    grid_size: int,
    bin_edges: np.ndarray,
    n_threads: int = 1,
) -> dict:
    """Compute density PDFs using the direct particle density method.

    This method computes density directly at particle positions from tetrahedra
    volumes, without constructing an intermediate 3D grid.
    """
    print("  Computing direct particle density...")

    # Sort positions to Lagrangian order using particle IDs
    start = time.perf_counter()
    sorted_positions = ts.density.sort_by_lagrangian_id(positions, particle_ids, grid_size)
    sort_time = time.perf_counter() - start
    print(f"    Sort to Lagrangian order: {sort_time*1000:.1f} ms")

    # Run ORIGAMI to get morphology classification
    origami_config = ts.origami.OrigamiConfig(grid_size, box_size)
    origami_config.n_threads = n_threads

    start = time.perf_counter()
    origami_result = ts.origami.compute_morphology(sorted_positions, origami_config)
    origami_time = time.perf_counter() - start
    print(f"    ORIGAMI classification: {origami_time*1000:.1f} ms")

    morphology = np.array(origami_result.morphology)

    # Now compute direct particle density
    config = ts.density.ParticleDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = box_size
    config.particle_mass = 1.0
    config.n_threads = n_threads
    config.periodic = True

    start = time.perf_counter()
    result = ts.density.compute_particle_density(sorted_positions, config)
    density_time = time.perf_counter() - start
    print(f"    Direct density computation: {density_time*1000:.1f} ms")
    print(f"    Total time: {(origami_time + density_time)*1000:.1f} ms")

    particle_density = np.array(result.density)
    mean_density = result.mean_density

    # Compute overdensity
    overdensity = particle_density / mean_density

    # Compute histograms and PDFs
    results = {
        'bin_edges': bin_edges,
        'bin_centers': 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        'bin_widths': bin_edges[1:] - bin_edges[:-1],
        'mean_density': mean_density,
        'density_time_ms': density_time * 1000,
        'origami_time_ms': origami_time * 1000,
    }

    # All particles
    hist_all, _ = np.histogram(overdensity, bins=bin_edges)
    pdf_all = hist_all / (len(overdensity) * results['bin_widths'])
    results['all'] = {
        'histogram': hist_all,
        'pdf': pdf_all,
        'n_particles': len(overdensity),
        'mean_overdensity': float(np.mean(overdensity)),
        'median_overdensity': float(np.median(overdensity)),
    }

    # Per-class results
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        mask = morphology == i
        od_class = overdensity[mask]

        if len(od_class) > 0:
            hist_class, _ = np.histogram(od_class, bins=bin_edges)
            pdf_class = hist_class / (len(od_class) * results['bin_widths'])
            results[class_name.lower()] = {
                'histogram': hist_class,
                'pdf': pdf_class,
                'n_particles': len(od_class),
                'mean_overdensity': float(np.mean(od_class)),
                'median_overdensity': float(np.median(od_class)),
            }
        else:
            results[class_name.lower()] = {
                'histogram': np.zeros(len(bin_edges) - 1, dtype=np.int64),
                'pdf': np.zeros(len(bin_edges) - 1),
                'n_particles': 0,
                'mean_overdensity': 0.0,
                'median_overdensity': 0.0,
            }

    print(f"    Mean overdensity: {results['all']['mean_overdensity']:.6f}")

    return results


def create_comparison_figure(
    grid_results: dict,
    direct_results: dict,
    snapshot_info: dict,
    output_path: Path,
) -> None:
    """Create figure comparing grid-based and direct density methods."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bin_centers = grid_results['bin_centers']

    # Top-left: All particles PDF comparison
    ax = axes[0, 0]
    ax.plot(bin_centers, grid_results['all']['pdf'], 'k-', linewidth=2, label='Grid-based MC')
    ax.plot(bin_centers, direct_results['all']['pdf'], 'r--', linewidth=2, label='Direct volume')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel('PDF')
    ax.set_title('All Particles')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Top-right: Per-class PDFs (grid-based)
    ax = axes[0, 1]
    ax.plot(bin_centers, grid_results['all']['pdf'], 'k--', linewidth=1.5, alpha=0.5, label='All')
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        key = class_name.lower()
        if grid_results[key]['pdf'].max() > 0:
            ax.plot(bin_centers, grid_results[key]['pdf'], color=ORIGAMI_COLORS[i],
                   linewidth=1.5, label=class_name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel('PDF')
    ax.set_title('Grid-Based MC (per class)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Bottom-left: Per-class PDFs (direct)
    ax = axes[1, 0]
    ax.plot(bin_centers, direct_results['all']['pdf'], 'k--', linewidth=1.5, alpha=0.5, label='All')
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        key = class_name.lower()
        if direct_results[key]['pdf'].max() > 0:
            ax.plot(bin_centers, direct_results[key]['pdf'], color=ORIGAMI_COLORS[i],
                   linewidth=1.5, label=class_name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel('PDF')
    ax.set_title('Direct Volume (per class)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Bottom-right: Ratio of PDFs
    ax = axes[1, 1]
    # Compute ratio where both have data
    pdf_grid = grid_results['all']['pdf']
    pdf_direct = direct_results['all']['pdf']
    mask = (pdf_grid > 0) & (pdf_direct > 0)
    ratio = np.ones_like(pdf_grid)
    ratio[mask] = pdf_direct[mask] / pdf_grid[mask]

    ax.plot(bin_centers, ratio, 'b-', linewidth=2, label='All')

    # Also plot per-class ratios
    for i, class_name in enumerate(ORIGAMI_CLASSES):
        key = class_name.lower()
        pdf_g = grid_results[key]['pdf']
        pdf_d = direct_results[key]['pdf']
        mask = (pdf_g > 0) & (pdf_d > 0)
        if mask.sum() > 5:
            r = np.ones_like(pdf_g)
            r[mask] = pdf_d[mask] / pdf_g[mask]
            ax.plot(bin_centers, r, color=ORIGAMI_COLORS[i], linewidth=1, alpha=0.7, label=class_name)

    ax.axhline(1.0, color='k', linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel('Direct / Grid-based')
    ax.set_title('PDF Ratio')
    ax.set_ylim(0.5, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Add overall title
    snap_idx = snapshot_info.get('snapshot_idx', '?')
    scale_factor = snapshot_info.get('scale_factor', 1.0)
    redshift = snapshot_info.get('redshift', 0.0)
    fig.suptitle(
        f"Density Method Comparison: Snapshot {snap_idx} (a={scale_factor:.4f}, z={redshift:.4f})",
        fontsize=12, fontweight='bold'
    )

    # Add statistics box
    stats_text = (
        f"Mean overdensity:\n"
        f"  Grid-based: {grid_results['all']['mean_overdensity']:.6f}\n"
        f"  Direct:     {direct_results['all']['mean_overdensity']:.6f}\n"
        f"\nComputation time:\n"
        f"  Direct density: {direct_results['density_time_ms']:.1f} ms\n"
        f"  (+ ORIGAMI: {direct_results['origami_time_ms']:.1f} ms)"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison figure: {output_path}")


def process_snapshot(
    snapshot_idx: int,
    snapshot_dir: Path,
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Process a single snapshot and create comparison."""

    print(f"\n{'='*70}")
    print(f"Processing Snapshot {snapshot_idx}")
    print('='*70)

    snapshot_path = snapshot_dir / f"snapshot_{snapshot_idx:03d}.hdf5"
    results_path = results_dir / f"L256_N256_primary_density_pdf_{snapshot_idx:03d}_jackknife.hdf5"
    output_path = output_dir / f"density_method_comparison_{snapshot_idx:03d}.png"

    if not snapshot_path.exists():
        print(f"  ERROR: Snapshot not found: {snapshot_path}")
        return

    if not results_path.exists():
        print(f"  ERROR: Existing results not found: {results_path}")
        return

    # Load snapshot
    positions, particle_ids, box_size, scale_factor, redshift = load_snapshot(snapshot_path)
    grid_size = int(round(len(positions) ** (1/3)))

    snapshot_info = {
        'snapshot_idx': snapshot_idx,
        'scale_factor': scale_factor,
        'redshift': redshift,
        'box_size': box_size,
        'grid_size': grid_size,
    }

    # Load existing grid-based results
    grid_results = load_existing_pdf_results(results_path)

    # Compute direct density PDFs using the same bins
    direct_results = compute_direct_density_pdfs(
        positions=positions,
        particle_ids=particle_ids,
        box_size=box_size,
        grid_size=grid_size,
        bin_edges=grid_results['bin_edges'],
        n_threads=1,
    )

    # Create comparison figure
    create_comparison_figure(
        grid_results=grid_results,
        direct_results=direct_results,
        snapshot_info=snapshot_info,
        output_path=output_path,
    )

    # Print comparison summary
    print(f"\n  Comparison Summary:")
    print(f"    Grid-based mean overdensity: {grid_results['all']['mean_overdensity']:.6f}")
    print(f"    Direct mean overdensity:     {direct_results['all']['mean_overdensity']:.6f}")

    # Compute KL divergence as a measure of difference
    pdf_g = grid_results['all']['pdf']
    pdf_d = direct_results['all']['pdf']
    mask = (pdf_g > 0) & (pdf_d > 0)
    if mask.sum() > 0:
        kl_div = np.sum(pdf_d[mask] * np.log(pdf_d[mask] / pdf_g[mask]) * grid_results['bin_widths'][mask])
        print(f"    KL divergence (direct || grid): {kl_div:.6f}")


def main():
    print("=" * 70)
    print("Direct vs Grid-Based Density Method Comparison")
    print("=" * 70)

    # Paths
    base_dir = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly")
    sim_dir = base_dir / "Uniform_L256_N256_primary_sandbox/gadget4/output"
    snapshot_dir = sim_dir
    results_dir = sim_dir / "density_pdf_results/results"
    output_dir = sim_dir / "density_method_comparison"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process snapshots 34 and 74
    for snap_idx in [34, 74]:
        process_snapshot(
            snapshot_idx=snap_idx,
            snapshot_dir=snapshot_dir,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
