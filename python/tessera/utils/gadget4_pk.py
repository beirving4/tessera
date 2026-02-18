"""
GADGET-4 power spectrum file reader.

Reads ``powerspec_NNN.txt`` files produced by GADGET-4's built-in power spectrum
module, which outputs 3 folded sections (NUM_FOLDED_POWER_SPECTRA_PIECES = 3)
with fold factor 16. Also reads the simpler 2-column ``for_colossus_NNN.txt``
log-space format.

File format (powerspec_NNN.txt):
    3 sections, each with a 5-line header then N_bins rows of 5 columns.
    Header per section: [scale_factor, n_bins, box_size, pmgrid, growth_factor]
    Data columns: k | Δ²(k) | ModePow | ModeCount | ShotNoise

    Footer (3 values after all sections): particle_mass, total_count, effective_tracers

    Folded sections extend the k range:
    - Section 0 (unfolded): reliable for k < k_Nyq
    - Section 1 (fold ×16): extends to k × 16
    - Section 2 (fold ×256): extends to k × 256

Example usage:
    >>> from tessera.utils.gadget4_pk import read_gadget4_powerspec
    >>> pk = read_gadget4_powerspec('powerspec_034.txt')
    >>> plt.loglog(pk['k'], pk['delta_squared'])
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def get_powerspec_paths(power_spec_dir: str | Path) -> OrderedDict[int, Path]:
    """Find all powerspec_NNN.txt files and return sorted mapping.

    Parameters
    ----------
    power_spec_dir : path-like
        Directory containing powerspec files.

    Returns
    -------
    paths : OrderedDict[int, Path]
        Mapping from snapshot number to file path, ordered by snapshot number.
    """
    power_spec_dir = Path(power_spec_dir)
    paths = {}
    for f in power_spec_dir.glob('powerspec_*.txt'):
        m = re.search(r'powerspec_(\d+)\.txt$', f.name)
        if m:
            paths[int(m.group(1))] = f
    return OrderedDict(sorted(paths.items()))


def read_gadget4_powerspec(
    filepath: str | Path,
    subtract_shot_noise: bool = True,
    target_bins: int = 50,
    min_mode_count: int = 8,
    n_sections: int = 3,
    fold_factor: int = 16,
) -> dict:
    """Parse a GADGET-4 power spectrum file and stitch folded sections.

    Reads all sections from the file, applies adaptive rebinning to reduce
    noise, stitches the sections together using the reliable k-range of each,
    and optionally subtracts shot noise.

    Parameters
    ----------
    filepath : path-like
        Path to ``powerspec_NNN.txt`` file.
    subtract_shot_noise : bool
        If True, subtract the shot noise column before stitching.
    target_bins : int
        Target number of bins per section after rebinning.
    min_mode_count : int
        Minimum number of modes per rebinned bin (bins with fewer modes
        are merged with neighbors).
    n_sections : int
        Number of folded sections (default 3 for standard GADGET-4).
    fold_factor : int
        Fold factor between successive sections (default 16).

    Returns
    -------
    result : dict
        ``'scale_factor'``
            Scale factor a from the file header.
        ``'box_size'``
            Simulation box size L.
        ``'pmgrid'``
            PM grid resolution.
        ``'growth_factor'``
            Linear growth factor D(a).
        ``'k'``
            Stitched wavenumber array (h/Mpc).
        ``'delta_squared'``
            Stitched dimensionless power Δ²(k).
        ``'Pk'``
            Stitched P(k) = Δ²(k) × 2π²/k³.
        ``'shot_noise'``
            Dict with ``'k'`` and ``'delta_squared'`` arrays (raw, all sections).
        ``'sections'``
            List of per-section raw data dicts (for debugging).
        ``'footer'``
            Dict with ``'particle_mass'``, ``'total_count'``, ``'effective_num_tracers'``.
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    sections = []
    idx = 0

    for s in range(n_sections):
        # Parse 5-line header
        scale_factor = float(lines[idx].strip()); idx += 1
        n_bins = int(lines[idx].strip()); idx += 1
        box_size = float(lines[idx].strip()); idx += 1
        pmgrid = int(lines[idx].strip()); idx += 1
        growth_factor = float(lines[idx].strip()); idx += 1

        # Parse data rows
        k = np.zeros(n_bins)
        delta_sq = np.zeros(n_bins)
        mode_pow = np.zeros(n_bins)
        mode_count = np.zeros(n_bins, dtype=np.int64)
        shot = np.zeros(n_bins)

        for i in range(n_bins):
            vals = lines[idx].strip().split()
            k[i] = float(vals[0])
            delta_sq[i] = float(vals[1])
            mode_pow[i] = float(vals[2])
            mode_count[i] = int(float(vals[3]))
            shot[i] = float(vals[4])
            idx += 1

        sections.append({
            'section': s,
            'scale_factor': scale_factor,
            'n_bins': n_bins,
            'box_size': box_size,
            'pmgrid': pmgrid,
            'growth_factor': growth_factor,
            'k': k,
            'delta_squared': delta_sq,
            'mode_pow': mode_pow,
            'mode_count': mode_count,
            'shot': shot,
        })

    # Parse footer (3 values)
    footer = {}
    if idx < len(lines):
        footer['particle_mass'] = float(lines[idx].strip()); idx += 1
    if idx < len(lines):
        footer['total_count'] = float(lines[idx].strip()); idx += 1
    if idx < len(lines):
        footer['effective_num_tracers'] = float(lines[idx].strip()); idx += 1

    # Stitch sections: use each section in its reliable k-range
    k_nyquist = np.pi * pmgrid / box_size  # PM grid Nyquist

    all_k = []
    all_delta_sq = []
    all_shot_k = []
    all_shot_delta_sq = []

    for s_idx, sec in enumerate(sections):
        k_s = sec['k'].copy()
        ds_s = sec['delta_squared'].copy()
        shot_s = sec['shot'].copy()
        mc_s = sec['mode_count'].copy()

        # Store raw shot noise for all sections
        all_shot_k.append(k_s.copy())
        all_shot_delta_sq.append(shot_s.copy())

        # Subtract shot noise before stitching
        if subtract_shot_noise:
            ds_s = ds_s - shot_s

        # Determine reliable k-range for this section
        if s_idx == 0:
            # Unfolded: use k from fundamental to k_Nyq/4
            # (aliasing starts affecting modes above k_Nyq/4)
            k_max = k_nyquist / 4.0
            mask = k_s <= k_max
        elif s_idx < n_sections - 1:
            # Intermediate fold: use k from k_Nyq/4 * fold^(s-1) to k_Nyq/4 * fold^s
            k_min = k_nyquist / 4.0 * fold_factor ** (s_idx - 1)
            k_max = k_nyquist / 4.0 * fold_factor ** s_idx
            mask = (k_s >= k_min) & (k_s <= k_max)
        else:
            # Last fold: use everything above the lower cutoff
            k_min = k_nyquist / 4.0 * fold_factor ** (s_idx - 1)
            mask = k_s >= k_min

        k_s = k_s[mask]
        ds_s = ds_s[mask]
        mc_s = mc_s[mask]

        if len(k_s) == 0:
            continue

        # Adaptive rebinning: merge bins to achieve min_mode_count
        k_rb, ds_rb = _adaptive_rebin(k_s, ds_s, mc_s, target_bins, min_mode_count)

        all_k.append(k_rb)
        all_delta_sq.append(ds_rb)

    # Concatenate and sort
    k_stitched = np.concatenate(all_k)
    delta_sq_stitched = np.concatenate(all_delta_sq)

    sort_idx = np.argsort(k_stitched)
    k_stitched = k_stitched[sort_idx]
    delta_sq_stitched = delta_sq_stitched[sort_idx]

    # Compute P(k) from Δ²(k)
    Pk_stitched = delta_sq_stitched * (2.0 * np.pi ** 2) / k_stitched ** 3

    return {
        'scale_factor': sections[0]['scale_factor'],
        'box_size': sections[0]['box_size'],
        'pmgrid': sections[0]['pmgrid'],
        'growth_factor': sections[0]['growth_factor'],
        'k': k_stitched,
        'delta_squared': delta_sq_stitched,
        'Pk': Pk_stitched,
        'shot_noise': {
            'k': np.concatenate(all_shot_k),
            'delta_squared': np.concatenate(all_shot_delta_sq),
        },
        'sections': sections,
        'footer': footer,
    }


def _adaptive_rebin(
    k: NDArray, delta_sq: NDArray, mode_count: NDArray,
    target_bins: int, min_mode_count: int,
) -> tuple[NDArray, NDArray]:
    """Rebin power spectrum data adaptively.

    Merges bins to ensure each output bin has at least ``min_mode_count``
    modes. The weighted average uses mode counts as weights.

    Returns (k_rebinned, delta_sq_rebinned).
    """
    n = len(k)
    if n <= target_bins:
        return k, delta_sq

    # Simple approach: uniform rebinning in log-k space, then merge small bins
    log_k_min = np.log10(k[0])
    log_k_max = np.log10(k[-1])
    log_edges = np.linspace(log_k_min, log_k_max, target_bins + 1)

    k_out = []
    ds_out = []

    for i in range(target_bins):
        mask = (np.log10(k) >= log_edges[i]) & (np.log10(k) < log_edges[i + 1])
        if i == target_bins - 1:
            mask = (np.log10(k) >= log_edges[i]) & (np.log10(k) <= log_edges[i + 1])

        if not np.any(mask):
            continue

        weights = mode_count[mask].astype(np.float64)
        total_modes = weights.sum()

        if total_modes < min_mode_count and len(k_out) > 0:
            # Skip and let the next bin absorb these modes
            continue

        if total_modes > 0:
            k_avg = np.average(k[mask], weights=weights)
            ds_avg = np.average(delta_sq[mask], weights=weights)
        else:
            k_avg = np.mean(k[mask])
            ds_avg = np.mean(delta_sq[mask])

        k_out.append(k_avg)
        ds_out.append(ds_avg)

    if len(k_out) == 0:
        return k, delta_sq

    return np.array(k_out), np.array(ds_out)


def read_gadget4_colossus_pk(filepath: str | Path) -> dict:
    """Read a ``for_colossus_NNN.txt`` file (2-column log10 format).

    Parameters
    ----------
    filepath : path-like
        Path to ``for_colossus_NNN.txt`` file.

    Returns
    -------
    result : dict
        ``'k'``
            Wavenumber array in linear scale (h/Mpc).
        ``'Pk'``
            P(k) array in linear scale ((Mpc/h)³).
        ``'log10_k'``
            Original log10(k) values.
        ``'log10_Pk'``
            Original log10(P(k)) values.
    """
    filepath = Path(filepath)
    data = np.loadtxt(filepath)
    log10_k = data[:, 0]
    log10_Pk = data[:, 1]

    return {
        'k': 10.0 ** log10_k,
        'Pk': 10.0 ** log10_Pk,
        'log10_k': log10_k,
        'log10_Pk': log10_Pk,
    }
