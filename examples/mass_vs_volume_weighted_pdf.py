#!/usr/bin/env python
"""Compare mass-weighted vs volume-weighted overdensity PDFs.

For each density method (DTFE grid-MC, CIC base, CIC optimal):
  - Left column: mass-weighted (1 count per particle)
  - Right column: volume-weighted (weight = 1/(1+delta) per particle)
  - Top row: histogram counts
  - Bottom row: PDF

Usage:
    python mass_vs_volume_weighted_pdf.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

# Setup tessera
sys.path.insert(0, '/Users/bryen/Documents/GitHub/tessera/build/cp312-cp312-macosx_26_0_arm64')
os.environ['DYLD_LIBRARY_PATH'] = '/Users/bryen/Documents/GitHub/tessera/build/cp312-cp312-macosx_26_0_arm64/src:' + os.environ.get('DYLD_LIBRARY_PATH', '')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import _tessera as ts

# ── Configuration ──
SNAPSHOT = '/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/snapshot_034.hdf5'
BOX_SIZE = 256.0
N_GRID = 256
N_THREADS = 8
N_BINS = 100
SAVE_DIR = '/Users/bryen/Documents/GitHub/tessera/examples'

# ── Load snapshot ──
print("Loading snapshot_034...")
with h5py.File(SNAPSHOT, 'r') as f:
    positions = np.ascontiguousarray(f['PartType1/Coordinates'][()], dtype=np.float64)
    particle_ids = np.array(f['PartType1/ParticleIDs'][()], dtype=np.int64)
N = len(positions)
rho_V = N * 1.0 / BOX_SIZE**3  # volume-weighted mean density (=1.0 for this sim)
print(f"  {N} particles, rho_V = {rho_V:.6f}")

# ── Run ORIGAMI pipeline (for morphology + DTFE density) ──
print("Running ORIGAMI pipeline (~100s)...")
config = ts.origami.PipelineConfig()
config.box_size = BOX_SIZE
config.density_output_cells = N_GRID
config.lagrangian_grid_size = N_GRID
config.n_threads = N_THREADS
config.sample_density_at_particles = True
config.pdf_n_bins = 0  # we compute PDFs ourselves

pos_work = positions.copy()  # pipeline sorts in-place
result = ts.origami.run_pipeline(pos_work, particle_ids, config)

morphology = np.array(result.morphology)
dtfe_density = np.array(result.particle_density)  # raw density (not normalized)
print(f"  Morphology: V={np.mean(morphology==0):.3f}, W={np.mean(morphology==1):.3f}, "
      f"F={np.mean(morphology==2):.3f}, H={np.mean(morphology==3):.3f}")

# ── Compute CIC densities (on the Lagrangian-sorted positions) ──
print("Computing CIC densities...")
cic_base_result = ts.density.compute_cic_density(
    pos_work, box_size=BOX_SIZE, output_cells=N_GRID,
    particle_mass=1.0, n_threads=N_THREADS, sample_at_particles=True)
cic_base_density = np.array(cic_base_result.particle_density)

cic_opt_result = ts.density.compute_cic_density(
    pos_work, box_size=BOX_SIZE, output_cells=64,
    particle_mass=1.0, n_threads=N_THREADS, sample_at_particles=True)
cic_opt_density = np.array(cic_opt_result.particle_density)

# ── Standard overdensity: rho / rho_V ──
# rho_V = 1.0 for this sim, so standard_od = raw_density numerically
methods = {
    'DTFE (grid MC)': dtfe_density / rho_V,
    r'CIC ($N_{grid}=256^3$)': cic_base_density / rho_V,
    r'CIC ($N_{grid}=64^3$, optimal)': cic_opt_density / rho_V,
}

for name, od in methods.items():
    print(f"  {name}: mass-weighted mean(1+δ) = {np.mean(od):.4f}, median = {np.median(od):.4f}")

# ── Morphology classes ──
classes = [
    ('All', np.ones(N, dtype=bool)),
    ('Void', morphology == 0),
    ('Wall', morphology == 1),
    ('Filament', morphology == 2),
    ('Halo', morphology == 3),
]
class_colors = {'All': 'black', 'Void': 'tab:blue', 'Wall': 'tab:orange',
                'Filament': 'tab:green', 'Halo': 'tab:red'}
class_linestyle = {'All': '--', 'Void': '-', 'Wall': '-', 'Filament': '-', 'Halo': '-'}
class_linewidth = {'All': 2.0, 'Void': 1.5, 'Wall': 1.5, 'Filament': 1.5, 'Halo': 1.5}


def compute_weighted_hist(overdensity, bin_edges, weight_type='mass'):
    """Compute histogram with mass or volume weighting."""
    od = overdensity[overdensity > 0]
    if weight_type == 'mass':
        weights = np.ones(len(od))
    else:  # volume
        weights = 1.0 / od
    hist, _ = np.histogram(od, bins=bin_edges, weights=weights)
    return hist, weights


def hist_to_pdf(hist, bin_edges, total_weight):
    """Convert histogram to PDF normalized to integrate to 1."""
    bin_widths = np.diff(bin_edges)
    pdf = hist / (total_weight * bin_widths)
    return pdf


# ── Plot each method ──
for method_name, overdensity in methods.items():
    # Determine bin edges from this method's full range
    od_pos = overdensity[overdensity > 0]
    lo = np.percentile(od_pos, 0.005)
    hi = np.percentile(od_pos, 99.995)
    bin_edges = np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{method_name} — Snapshot 34 (a=1.0, z=0.0)', fontsize=14, y=0.98)

    for col, wt in enumerate(['mass', 'volume']):
        ax_hist = axes[0, col]
        ax_pdf = axes[1, col]

        for cls_name, mask in classes:
            od_cls = overdensity[mask]
            od_cls = od_cls[od_cls > 0]

            if wt == 'mass':
                weights = np.ones(len(od_cls))
            else:
                weights = 1.0 / od_cls

            hist, _ = np.histogram(od_cls, bins=bin_edges, weights=weights)
            total_w = np.sum(weights)
            pdf = hist / (total_w * np.diff(bin_edges))

            ls = class_linestyle[cls_name]
            lw = class_linewidth[cls_name]
            color = class_colors[cls_name]

            # Mask zero bins for log scale
            nonzero = hist > 0
            ax_hist.step(bin_centers[nonzero], hist[nonzero], where='mid',
                        color=color, linestyle=ls, linewidth=lw, label=cls_name)
            ax_pdf.step(bin_centers[nonzero], pdf[nonzero], where='mid',
                       color=color, linestyle=ls, linewidth=lw, label=cls_name)

        # Formatting
        wt_label = 'Mass-weighted' if wt == 'mass' else 'Volume-weighted'
        ax_hist.set_xscale('log')
        ax_hist.set_yscale('log')
        ax_hist.set_title(wt_label, fontsize=12)
        ax_hist.set_ylabel('Counts' if wt == 'mass' else 'Effective volume')
        ax_hist.legend(fontsize=9, loc='upper right')
        ax_hist.grid(True, alpha=0.3, which='both')

        ax_pdf.set_xscale('log')
        ax_pdf.set_yscale('log')
        ax_pdf.set_xlabel(r'$1+\delta$')
        ax_pdf.set_ylabel('PDF')
        ax_pdf.grid(True, alpha=0.3, which='both')

        # Annotate weighted mean
        od_pos = overdensity[overdensity > 0]
        if wt == 'mass':
            mean_val = np.mean(od_pos)
        else:
            w = 1.0 / od_pos
            mean_val = np.sum(od_pos * w) / np.sum(w)
        ax_pdf.text(0.02, 0.02, f'$\\langle 1+\\delta \\rangle$ = {mean_val:.4f}',
                   transform=ax_pdf.transAxes, fontsize=10, va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Mass fractions annotation (bottom-left of PDF panel)
        if col == 0:
            frac_text = 'Mass fractions:\n'
            for cls_name, mask in classes[1:]:  # skip 'All'
                frac_text += f'  {cls_name}: {np.sum(mask)/N*100:.1f}%\n'
            ax_pdf.text(0.02, 0.15, frac_text.strip(),
                       transform=ax_pdf.transAxes, fontsize=8, va='bottom',
                       family='monospace',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Share x-axes within columns
    axes[0, 0].sharex(axes[1, 0])
    axes[0, 1].sharex(axes[1, 1])

    plt.tight_layout()
    safe = method_name.replace(' ', '_').replace('$', '').replace('{', '').replace('}', '')
    safe = safe.replace('(', '').replace(')', '').replace('^', '').replace(',', '').replace('\\', '')
    save_path = os.path.join(SAVE_DIR, f'mass_vs_volume_pdf_{safe}_034.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

print("\nDone!")
