#!/usr/bin/env python3
"""
Density Method Comparison: CIC vs DTFE vs VTFE

Compares three density estimation methods on the same snapshot:
  1. CIC  (Cloud-in-Cell) — Klypin et al. 2018: Grid-based Eulerian density
  2. DTFE (Tessellation)  — Abel, Hahn & Kaehler 2012: Lagrangian tetrahedra
  3. VTFE (Voronoi)       — Falck et al. 2012: Eulerian Voronoi cell volumes

Outputs:
- 3-panel figure: (1) Overdensity PDF, (2) rho^2 P(rho) plot, (3) Timing bars
- Summary statistics printed to console

Usage:
    python density_method_comparison.py
    python density_method_comparison.py --snapshot snapshot_074
    python density_method_comparison.py --cic-cells 128
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

SCRIPT_DIR = Path(__file__).parent.resolve()

# Import shared utilities
from tessera.utils import infer_grid_size, load_snapshot_h5py

# Import local config
try:
    import importlib.util
    _config_path = SCRIPT_DIR / 'config.py'
    if not _config_path.exists():
        raise FileNotFoundError
    _spec = importlib.util.spec_from_file_location('examples_config', str(_config_path))
    _config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_config)
    SNAPSHOT_BASE = _config.SNAPSHOT_BASE
except (FileNotFoundError, AttributeError):
    raise ImportError(
        "Please create examples/config.py with your local paths.\n"
        "Copy config.example.py to config.py and update the paths."
    )

# Import tessera
try:
    import tessera as ts
except ImportError:
    REPO_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(REPO_ROOT / 'build'))
    import tessera as ts


def load_snapshot(snapshot_name):
    """Load particle data from GADGET-4 snapshot."""
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"
    print(f"Loading {snapshot_path}...")

    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )
    grid_size = infer_grid_size(len(positions))

    print(f"  {len(positions):,} particles (grid: {grid_size}^3), box: {box_size}, a={scale_factor:.4f}")
    return positions, particle_ids, box_size, grid_size


def compute_overdensity_pdf(values, n_bins=100, range_min=None, range_max=None):
    """Compute overdensity PDF with log bins following Klypin et al. normalization.

    Returns bin_centers, pdf, bin_edges where:
      - pdf is normalized so integral P(rho) d(rho) = 1
      - bin_centers are geometric means of bin edges
    """
    positive = values[values > 0]
    if range_min is None:
        range_min = max(positive.min(), 1e-3)
    if range_max is None:
        range_max = positive.max() * 1.001

    bin_edges = np.logspace(np.log10(range_min), np.log10(range_max), n_bins + 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    counts, _ = np.histogram(positive, bins=bin_edges)
    pdf = counts / (len(values) * bin_widths)

    return bin_centers, pdf, bin_edges


def run_cic(positions, box_size, cic_cells):
    """Run CIC density and return per-particle overdensity + timing."""
    t0 = time.time()
    result = ts.density.compute_cic_density(
        positions, box_size=float(box_size), output_cells=cic_cells,
        sample_at_particles=True)
    elapsed = time.time() - t0

    overdensity = np.array(result.particle_density) / result.mean_density
    print(f"  CIC:  {elapsed:.2f}s, mean(1+d)={overdensity.mean():.6f}, "
          f"min={overdensity.min():.4f}, max={overdensity.max():.1f}")
    return overdensity, elapsed


def run_dtfe(positions, particle_ids, box_size, grid_size):
    """Run DTFE (tessellation) density and return per-particle overdensity + timing."""
    # DTFE requires Lagrangian sorting
    pos_copy = positions.copy()
    t0 = time.time()

    # Sort to Lagrangian order
    sorted_pos, ordering, offset = ts.origami.sort_particles_to_lagrangian(
        pos_copy, particle_ids, grid_size, float(box_size))

    # Compute direct particle density from tetrahedra
    result = ts.density.compute_particle_density(
        sorted_pos, lagrangian_grid_size=grid_size, box_size=float(box_size))
    elapsed = time.time() - t0

    overdensity = np.array(result.density) / result.mean_density
    print(f"  DTFE: {elapsed:.2f}s, mean(1+d)={overdensity.mean():.6f}, "
          f"min={overdensity.min():.4f}, max={overdensity.max():.1f}")
    return overdensity, elapsed


def run_vtfe(positions, box_size):
    """Run VTFE (Voronoi) density and return per-particle overdensity + timing."""
    t0 = time.time()
    result = ts.density.compute_voronoi_density(
        positions, box_size=float(box_size))
    elapsed = time.time() - t0

    overdensity = np.array(result.density) / result.mean_density
    print(f"  VTFE: {elapsed:.2f}s, mean(1+d)={overdensity.mean():.6f}, "
          f"min={overdensity.min():.4f}, max={overdensity.max():.1f}")
    return overdensity, elapsed


def create_comparison_figure(results, snapshot_name, output_path, n_bins=100):
    """Create 3-panel comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {'CIC': '#1f77b4', 'DTFE': '#ff7f0e', 'VTFE': '#2ca02c'}

    # Compute common bin range across all methods
    all_od = np.concatenate([r['overdensity'] for r in results.values()])
    positive = all_od[all_od > 0]
    range_min = max(positive.min(), 1e-3)
    range_max = positive.max() * 1.001

    # Panel 1: Overdensity PDF P(1+delta)
    ax = axes[0]
    for name, r in results.items():
        centers, pdf, edges = compute_overdensity_pdf(
            r['overdensity'], n_bins=n_bins, range_min=range_min, range_max=range_max)
        mask = pdf > 0
        ax.plot(centers[mask], pdf[mask], color=colors[name], linewidth=2, label=name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel(r'$P(1+\delta)$')
    ax.set_title('Overdensity PDF')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: rho^2 P(rho) — removes power-law trend (Klypin normalization)
    ax = axes[1]
    for name, r in results.items():
        centers, pdf, edges = compute_overdensity_pdf(
            r['overdensity'], n_bins=n_bins, range_min=range_min, range_max=range_max)
        rho2_pdf = centers**2 * pdf
        mask = pdf > 0
        ax.plot(centers[mask], rho2_pdf[mask], color=colors[name], linewidth=2, label=name)

    ax.set_xscale('log')
    ax.set_xlabel(r'$1 + \delta$')
    ax.set_ylabel(r'$(1+\delta)^2 \, P(1+\delta)$')
    ax.set_title(r'$\rho^2 P(\rho)$ (Klypin normalization)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Timing comparison
    ax = axes[2]
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    bars = ax.bar(names, times, color=[colors[n] for n in names], width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Density Method Comparison: {snapshot_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Density Method Comparison: CIC vs DTFE vs VTFE")
    parser.add_argument('--snapshot', default='snapshot_034',
                        help='Snapshot name (default: snapshot_034)')
    parser.add_argument('--cic-cells', type=int, default=256,
                        help='CIC grid resolution (default: 256)')
    parser.add_argument('--n-bins', type=int, default=100,
                        help='Number of PDF bins (default: 100)')
    parser.add_argument('--skip-vtfe', action='store_true',
                        help='Skip VTFE (Voronoi) if it is too slow')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Density Method Comparison: CIC vs DTFE vs VTFE")
    print("=" * 70)

    positions, particle_ids, box_size, grid_size = load_snapshot(args.snapshot)

    results = {}

    # CIC
    print("\nRunning CIC density...")
    od_cic, t_cic = run_cic(positions, box_size, args.cic_cells)
    results['CIC'] = {'overdensity': od_cic, 'time': t_cic}

    # DTFE
    print("\nRunning DTFE density...")
    od_dtfe, t_dtfe = run_dtfe(positions, particle_ids, box_size, grid_size)
    results['DTFE'] = {'overdensity': od_dtfe, 'time': t_dtfe}

    # VTFE
    if not args.skip_vtfe:
        print("\nRunning VTFE density...")
        od_vtfe, t_vtfe = run_vtfe(positions, box_size)
        results['VTFE'] = {'overdensity': od_vtfe, 'time': t_vtfe}
    else:
        print("\nSkipping VTFE (--skip-vtfe)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<8} {'Time (s)':<12} {'Mean(1+d)':<14} {'Min':<12} {'Max':<12}")
    print("-" * 58)
    for name, r in results.items():
        od = r['overdensity']
        print(f"{name:<8} {r['time']:<12.2f} {od.mean():<14.6f} {od.min():<12.4f} {od.max():<12.1f}")

    # Create figure
    fig_path = SCRIPT_DIR / f"{args.snapshot}_density_comparison.png"
    create_comparison_figure(results, args.snapshot, fig_path, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
