#!/usr/bin/env python3
"""
PDF Resolution Comparison: CIC grid at N=64, 128, 256

Checks whether the low-density flattening in the unsmoothed PDF is a
resolution artifact by comparing CIC overdensity PDFs at different mesh sizes.
No additional Fourier smoothing — the CIC cell width IS the smoothing scale.

Produces a 2-panel figure (a=1 and a=100), each with 3 PDFs overlaid.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

SCRIPT_DIR = Path(__file__).parent.resolve()

from tessera.utils import load_snapshot_h5py, infer_grid_size

try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location('examples_config', str(SCRIPT_DIR / 'config.py'))
    _config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_config)
    SNAPSHOT_BASE = _config.SNAPSHOT_BASE
except (FileNotFoundError, AttributeError):
    raise ImportError("Please create examples/config.py with your local paths.")

try:
    import tessera as ts
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR.parent / 'build'))
    import tessera as ts


def compute_overdensity_pdf(delta, n_bins=120):
    """Volume-weighted PDF of ρ/ρ̄ = 1+δ with log-spaced bins."""
    rho_over_rhobar = 1.0 + delta.ravel()
    positive = rho_over_rhobar[rho_over_rhobar > 0]

    log_min = np.log10(max(positive.min(), 1e-4))
    log_max = np.log10(positive.max() * 1.001)
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    counts, _ = np.histogram(positive, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    pdf = counts / (len(positive) * bin_widths)

    return bin_centers, pdf


def run_snapshot(snapshot_name, ax):
    """Compute and plot PDFs at multiple resolutions for one snapshot."""
    import h5py as h5
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"

    positions, _, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=False)

    with h5.File(snapshot_path, 'r') as f:
        particle_mass = float(f['Header'].attrs['MassTable'][1])

    n_particles = len(positions)
    grid_sizes = [64, 128, 256]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    linestyles = ['-', '-', '-']

    for N_grid, color, ls in zip(grid_sizes, colors, linestyles):
        cell_width = box_size / N_grid
        print(f"  N_grid={N_grid} (Δ={cell_width:.2f} Mpc/h)...", end=' ', flush=True)
        t0 = time.time()

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=N_grid,
            particle_mass=particle_mass, n_threads=1)

        grid = np.array(result.grid_density).reshape(N_grid, N_grid, N_grid)
        delta = grid / grid.mean() - 1.0

        # Fraction of empty cells
        empty_frac = np.sum(grid == 0) / grid.size

        bc, pdf = compute_overdensity_pdf(delta)
        dt = time.time() - t0

        var = np.var(delta)
        print(f"σ²={var:.4f}, empty={empty_frac:.1%}, {dt:.2f}s")

        label = f'$N_{{\\rm grid}}={N_grid}$  ($\\Delta={cell_width:.1f}$, $\\sigma^2={var:.3f}$)'
        ax.plot(bc, pdf, color=color, ls=ls, lw=1.8, label=label, alpha=0.85)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\rho / \\bar{\\rho}$')
    ax.set_ylabel('$P(\\rho / \\bar{\\rho})$')
    ax.set_xlim(1e-3, 1e4)
    ax.legend(fontsize=8, loc='upper right')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, snap_name, title in [
        (axes[0], 'snapshot_034', '$a = 1$ (present day)'),
        (axes[1], 'snapshot_074', '$a = 100$ (far future)'),
    ]:
        print(f"\n{snap_name}:")
        ax.set_title(title, fontsize=12)
        run_snapshot(snap_name, ax)

    fig.suptitle('CIC Overdensity PDF — Resolution Dependence', fontsize=13, y=1.02)
    fig.tight_layout()

    output_path = SCRIPT_DIR / 'pdf_resolution_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


if __name__ == '__main__':
    main()
