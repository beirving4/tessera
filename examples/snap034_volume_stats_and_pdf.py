#!/usr/bin/env python
"""Run tetrahedra volume stats and mass/volume-weighted PDF comparison on snapshot_034."""

import numpy as np
import tessera as ts
from tessera.utils import load_snapshot, infer_grid_size
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Load snapshot_034
# ═══════════════════════════════════════════════════════════════════════════
snapshot_path = "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/snapshot_034.hdf5"
n_threads = 8

positions, particle_ids, header = load_snapshot(snapshot_path, read_ids=True)
grid_size = infer_grid_size(len(positions))
box_size = header.box_size

print(f"Loaded {len(positions):,} particles, grid: {grid_size}^3, box: {box_size}")
print(f"Scale factor a = {header.time}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 1) Direct particle density (tessellation) — prints volume stats
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TESSELLATION (DTFE direct) -- snapshot_034")
print("=" * 70)

pos_work = positions.copy()
config = ts.origami.PipelineConfig()
config.box_size = box_size
config.lagrangian_grid_size = grid_size
config.use_direct_particle_density = True
config.particle_mass = 1.0
config.n_threads = n_threads

result_dtfe = ts.origami.run_pipeline(pos_work, particle_ids, config)

print(f"\nTetrahedra volume statistics:")
print(f"  n_tetrahedra:            {6 * grid_size**3:,}")
print(f"  min_tetra_volume:        {result_dtfe.min_tetra_volume:.6e}")
print(f"  max_tetra_volume:        {result_dtfe.max_tetra_volume:.6e}")
print(f"  mean_tetra_volume (geom):{result_dtfe.mean_tetra_volume:.6e}")
print(f"  stddev (linear):         {result_dtfe.stddev_tetra_volume:.6e}")
print(f"  2*stddev (linear):       {result_dtfe.stddev2_tetra_volume:.6e}")
print(f"  log_stddev:              {result_dtfe.log_stddev_tetra_volume:.6f}")
print(f"  2*log_stddev:            {result_dtfe.log_stddev2_tetra_volume:.6f}")
print(f"  mean_density:            {result_dtfe.mean_density:.6e}")
print(f"  density_time_ms:         {result_dtfe.density_time_ms:.1f}")

geom_mean = result_dtfe.mean_tetra_volume
log_sig = result_dtfe.log_stddev_tetra_volume
print(f"\n  Geometric mean +/- 1sig range: [{geom_mean * np.exp(-log_sig):.4e}, {geom_mean * np.exp(log_sig):.4e}]")
print(f"  Geometric mean +/- 2sig range: [{geom_mean * np.exp(-2*log_sig):.4e}, {geom_mean * np.exp(2*log_sig):.4e}]")

lag_cell_vol = (box_size / grid_size)**3
lag_tetra_vol = lag_cell_vol / 6.0
print(f"\n  Expected Lagrangian tetra volume: {lag_tetra_vol:.6e}")
print(f"  Ratio geom_mean / Lagrangian:     {geom_mean / lag_tetra_vol:.4f}")

dtfe_density = np.array(result_dtfe.particle_density)

# ═══════════════════════════════════════════════════════════════════════════
# 2) CIC density (256^3 grid)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CIC (256^3 grid) -- snapshot_034")
print("=" * 70)

cic_result = ts.density.compute_cic_density(
    positions,
    box_size=box_size,
    output_cells=256,
    particle_mass=1.0,
    n_threads=n_threads,
    sample_at_particles=True
)
cic_density = np.array(cic_result.particle_density)
print(f"  CIC mean density: {cic_result.mean_density:.6e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3) Compute mass-weighted and volume-weighted PDFs
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PDF COMPARISON: Mass-weighted vs Volume-weighted")
print("=" * 70)

def compute_pdfs(density_arr, label, n_bins=100):
    rho_M = np.mean(density_arr)
    od = density_arr / rho_M
    od_pos = od[od > 0]

    bin_edges = np.logspace(np.log10(od_pos.min()), np.log10(od_pos.max()), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    hist_mw, _ = np.histogram(od_pos, bins=bin_edges)
    pdf_mw = hist_mw / (len(od_pos) * bin_widths)

    weights_vw = 1.0 / od_pos
    hist_vw, _ = np.histogram(od_pos, bins=bin_edges, weights=weights_vw)
    pdf_vw = hist_vw / (np.sum(weights_vw) * bin_widths)

    integral_mw = np.sum(pdf_mw * bin_widths)
    integral_vw = np.sum(pdf_vw * bin_widths)
    mean_od_mw = np.mean(od_pos)
    mean_od_vw = np.sum(od_pos * weights_vw) / np.sum(weights_vw)

    print(f"\n  {label}:")
    print(f"    rho_M (particle mean):       {rho_M:.6e}")
    print(f"    Mass-weighted <1+d>:         {mean_od_mw:.6f}  (integral={integral_mw:.4f})")
    print(f"    Volume-weighted <1+d>:       {mean_od_vw:.6f}  (integral={integral_vw:.4f})")
    print(f"    Overdensity range:           [{od_pos.min():.4e}, {od_pos.max():.4e}]")

    return bin_centers, pdf_mw, pdf_vw, bin_edges

bc_dtfe, mw_dtfe, vw_dtfe, be_dtfe = compute_pdfs(dtfe_density, "DTFE (tessellation)")
bc_cic, mw_cic, vw_cic, be_cic = compute_pdfs(cic_density, "CIC (256^3)")

# ═══════════════════════════════════════════════════════════════════════════
# 4) Plot comparison
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

ax = axes[0]
ax.loglog(bc_dtfe, mw_dtfe, 'b-', lw=1.5, alpha=0.85, label='DTFE (tessellation)')
ax.loglog(bc_cic, mw_cic, 'r-', lw=1.5, alpha=0.85, label='CIC (256$^3$)')
ax.set_xlabel(r'$1 + \delta$', fontsize=13)
ax.set_ylabel(r'$P(1+\delta)$', fontsize=13)
ax.set_title('Mass-weighted PDF', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(1e-3, 1e4)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.loglog(bc_dtfe, vw_dtfe, 'b-', lw=1.5, alpha=0.85, label='DTFE (tessellation)')
ax.loglog(bc_cic, vw_cic, 'r-', lw=1.5, alpha=0.85, label='CIC (256$^3$)')
ax.set_xlabel(r'$1 + \delta$', fontsize=13)
ax.set_title('Volume-weighted PDF', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(1e-3, 1e4)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle(r'Density PDF comparison -- snapshot\_034 ($a=1$, $256^3$ particles)', fontsize=14, y=1.02)
plt.tight_layout()
outpath = '/Users/bryen/Documents/GitHub/tessera/examples/snap034_mass_vs_volume_pdf_comparison.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {outpath}")
