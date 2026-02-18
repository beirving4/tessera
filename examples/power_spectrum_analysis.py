#!/usr/bin/env python3
"""
Power Spectrum Analysis: CIC grid → P(k) → Fourier smoothing → σ_R² cross-check

Demonstrates the full P(k) and smoothed-PDF pipeline:
  1. Load snapshot → CIC density grid (C++ compute_cic_density)
  2. Compute P(k) from the CIC overdensity field
  3. Compare tessera P(k) to GADGET-4's built-in P(k)
  4. Smooth δ at multiple scales R via Fourier-space convolution
  5. Compute PDFs at each smoothing scale
  6. Cross-check: σ_R² from PDF variance vs. P(k) integral

Outputs a 3-panel figure:
  Panel 1: P(k) comparison (tessera CIC vs GADGET-4)
  Panel 2: Overdensity PDFs at different smoothing scales
  Panel 3: σ_R² cross-check (field variance vs P(k) integral)

Usage:
    python power_spectrum_analysis.py
    python power_spectrum_analysis.py --snapshot snapshot_074
    python power_spectrum_analysis.py --cic-cells 128
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
else:
    os.environ.setdefault('OMP_NUM_THREADS', '4')

SCRIPT_DIR = Path(__file__).parent.resolve()

# Import shared utilities
from tessera.utils import load_snapshot_h5py, infer_grid_size
from tessera.utils.fourier import (
    smooth_field_fourier,
    compute_power_spectrum,
    sigma_R_squared_from_pk,
    sigma_R_squared_from_field,
)
from tessera.utils.gadget4_pk import read_gadget4_powerspec

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
    import h5py as h5
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"
    print(f"Loading {snapshot_path}...")

    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )
    grid_size = infer_grid_size(len(positions))

    with h5.File(snapshot_path, 'r') as f:
        mass_table = f['Header'].attrs['MassTable']
        particle_mass = float(mass_table[1])

    print(f"  {len(positions):,} particles (grid: {grid_size}^3), box: {box_size}, a={scale_factor:.4f}")
    print(f"  Particle mass: {particle_mass:.6f} [10^10 Msun/h]")
    return positions, particle_ids, box_size, grid_size, particle_mass, scale_factor


def compute_cic_overdensity(positions, box_size, output_cells, particle_mass):
    """Compute CIC density grid and convert to overdensity δ = ρ/ρ̄ − 1."""
    print(f"\nComputing CIC density on {output_cells}^3 grid...")
    t0 = time.time()

    result = ts.density.compute_cic_density(
        positions, box_size=box_size, output_cells=output_cells,
        particle_mass=particle_mass, n_threads=1,
    )

    grid = np.array(result.grid_density).reshape(output_cells, output_cells, output_cells)
    mean_density = grid.mean()
    delta = grid / mean_density - 1.0

    dt = time.time() - t0
    print(f"  CIC done in {dt:.2f}s")
    print(f"  Mean density: {mean_density:.6e}")
    print(f"  δ range: [{delta.min():.3f}, {delta.max():.3f}]")
    print(f"  δ variance: {np.var(delta):.6f}")

    return delta, mean_density


def compute_overdensity_pdf(delta, n_bins=100):
    """Compute volume-weighted PDF of overdensity ρ/ρ̄ = 1+δ."""
    rho_over_rhobar = 1.0 + delta.ravel()
    positive = rho_over_rhobar[rho_over_rhobar > 0]

    log_min = np.log10(max(positive.min(), 1e-3))
    log_max = np.log10(positive.max() * 1.001)
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    counts, _ = np.histogram(positive, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    pdf = counts / (len(positive) * bin_widths)

    return bin_centers, pdf


def main():
    parser = argparse.ArgumentParser(description="Power spectrum analysis with tessera")
    parser.add_argument('--snapshot', default='snapshot_034',
                        help='Snapshot name (default: snapshot_034)')
    parser.add_argument('--cic-cells', type=int, default=256,
                        help='CIC grid resolution (default: 256)')
    parser.add_argument('--output', default=None,
                        help='Output figure path (default: <snapshot>_pk_analysis.png)')
    args = parser.parse_args()

    snapshot_name = args.snapshot
    output_cells = args.cic_cells
    snap_num = snapshot_name.split('_')[-1]

    if args.output is None:
        output_path = SCRIPT_DIR / f"{snapshot_name}_pk_analysis.png"
    else:
        output_path = Path(args.output)

    # ── 1. Load snapshot ──
    positions, particle_ids, box_size, grid_size, particle_mass, scale_factor = \
        load_snapshot(snapshot_name)
    n_particles = len(positions)

    # ── 2. CIC overdensity ──
    delta, mean_density = compute_cic_overdensity(
        positions, box_size, output_cells, particle_mass)
    cell_width = box_size / output_cells

    # ── 3. Compute P(k) from CIC grid ──
    print("\nComputing power spectrum from CIC grid...")
    t0 = time.time()
    pk_tessera = compute_power_spectrum(
        delta, box_size=box_size, n_bins=output_cells // 2,
        deconvolve_cic=True, subtract_shot_noise=True, n_particles=n_particles,
    )
    dt = time.time() - t0
    print(f"  P(k) done in {dt:.2f}s, {len(pk_tessera['k'])} bins")

    # ── 4. Read GADGET-4 P(k) ──
    powerspec_dir = SNAPSHOT_BASE / 'powerspecs'
    powerspec_file = powerspec_dir / f'powerspec_{snap_num}.txt'
    pk_gadget = None
    if powerspec_file.exists():
        print(f"\nReading GADGET-4 P(k) from {powerspec_file.name}...")
        pk_gadget = read_gadget4_powerspec(powerspec_file)
        print(f"  {len(pk_gadget['k'])} stitched bins, a={pk_gadget['scale_factor']}")
    else:
        print(f"\nGADGET-4 powerspec not found at {powerspec_file}")

    # ── 5. Smooth at multiple scales ──
    R_values = [2 * cell_width, 4 * cell_width, 8 * cell_width, 16 * cell_width]
    print(f"\nSmoothing at R = {[f'{R:.2f}' for R in R_values]} Mpc/h...")

    smoothed_fields = {}
    for R in R_values:
        t0 = time.time()
        smoothed = smooth_field_fourier(delta, box_size=box_size, R=R, window='gaussian')
        dt = time.time() - t0
        smoothed_fields[R] = smoothed
        print(f"  R={R:.2f}: σ²={np.var(smoothed):.6f}, time={dt:.2f}s")

    # ── 6. PDFs at each smoothing scale ──
    print("\nComputing PDFs...")
    pdfs = {}
    for R in R_values:
        bin_centers, pdf = compute_overdensity_pdf(smoothed_fields[R])
        pdfs[R] = (bin_centers, pdf)

    # Also compute unsmoothed PDF
    bin_centers_raw, pdf_raw = compute_overdensity_pdf(delta)

    # ── 7. σ_R² cross-check ──
    print("\nσ_R² cross-check (field variance vs P(k) integral):")
    print(f"  {'R':>8s}  {'σ²(field)':>12s}  {'σ²(P(k))':>12s}  {'ratio':>8s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")

    sigma_field_list = []
    sigma_pk_list = []
    for R in R_values:
        sig_field = sigma_R_squared_from_field(smoothed_fields[R])
        sig_pk = sigma_R_squared_from_pk(
            pk_tessera['k'], pk_tessera['Pk'], R=R, window='gaussian',
        )
        ratio = sig_pk / sig_field if sig_field > 0 else float('nan')
        sigma_field_list.append(sig_field)
        sigma_pk_list.append(sig_pk)
        print(f"  {R:8.2f}  {sig_field:12.6f}  {sig_pk:12.6f}  {ratio:8.4f}")

    # ── 8. Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: P(k) comparison
    ax = axes[0]
    ax.loglog(pk_tessera['k'], pk_tessera['delta_squared'],
              'b-', lw=1.5, label='Tessera CIC', alpha=0.8)
    if pk_gadget is not None:
        ax.loglog(pk_gadget['k'], pk_gadget['delta_squared'],
                  'r--', lw=1.5, label='GADGET-4', alpha=0.8)
    ax.axvline(pk_tessera['k_nyquist'], color='gray', ls=':', alpha=0.5, label='$k_{\\rm Nyq}$')
    ax.set_xlabel('$k$ [h/Mpc]')
    ax.set_ylabel('$\\Delta^2(k)$')
    ax.set_title(f'Power Spectrum ($a={scale_factor:.2f}$)')
    ax.legend(fontsize=9)
    ax.set_xlim(pk_tessera['k_fundamental'] * 0.8, pk_tessera['k_nyquist'] * 1.5)

    # Panel 2: PDFs at different smoothing scales
    ax = axes[1]
    ax.plot(bin_centers_raw, pdf_raw, 'k-', lw=1, alpha=0.4, label='Unsmoothed')
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(R_values)))
    for (R, (bc, pdf)), color in zip(pdfs.items(), colors):
        ax.plot(bc, pdf, color=color, lw=1.5, label=f'$R={R:.1f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\rho / \\bar{\\rho}$')
    ax.set_ylabel('$P(\\rho / \\bar{\\rho})$')
    ax.set_title('Volume-weighted PDF')
    ax.legend(fontsize=8)
    ax.set_xlim(1e-2, 1e3)

    # Panel 3: σ_R² cross-check
    ax = axes[2]
    ax.plot(R_values, sigma_field_list, 'bo-', lw=1.5, ms=6, label='$\\sigma^2$ from field')
    ax.plot(R_values, sigma_pk_list, 'r^--', lw=1.5, ms=6, label='$\\sigma^2$ from $P(k)$')
    ax.set_xlabel('Smoothing scale $R$ [Mpc/h]')
    ax.set_ylabel('$\\sigma_R^2$')
    ax.set_title('$\\sigma_R^2$ Cross-check')
    ax.legend(fontsize=9)
    ax.set_xscale('log')

    fig.suptitle(f'{snapshot_name} — Power Spectrum & Smoothed PDF Analysis', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


if __name__ == '__main__':
    main()
