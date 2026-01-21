#!/usr/bin/env python3
"""
Density PDF Computation using Phase-Space Tessellation

This script computes the one-point probability distribution function (PDF)
of the matter density field from a GADGET-4 snapshot, using the phase-space
tessellation method.

The PDF is computed following standard cosmological conventions:
- Density is expressed as overdensity: ρ/ρ_mean = 1 + δ
- Bins are logarithmically spaced in (1 + δ)
- The PDF P(ρ) satisfies: ∫P(ρ)dρ = 1 and ∫ρP(ρ)dρ = 1

Output includes:
- HDF5 file with density values (physical and overdensity) and PDF
- Visualization of ρ²P(ρ) vs (1 + δ)

Reference: The PDF definition follows conventions used in cosmological
literature (e.g., CIC density assignment schemes).
"""

import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared utilities
from utils import (
    load_snapshot,
    infer_grid_size,
    sort_positions_lagrangian,
    check_tessera_available,
)

# Import tessera
try:
    import tessera as ts
except ImportError:
    import _tessera as ts

HAS_CPP = check_tessera_available()


def compute_density_field(positions, particle_ids, header, grid_resolution=None,
                          n_samples=100, n_threads=0, particle_type=1):
    """
    Compute 3D density field using tessellation.

    Parameters
    ----------
    positions : ndarray
        Particle positions (N, 3)
    particle_ids : ndarray
        Particle IDs for Lagrangian ordering
    header : Gadget4Header
        Snapshot header
    grid_resolution : int, optional
        Output grid resolution (default: same as Lagrangian grid)
    n_samples : int
        Monte Carlo samples per tetrahedron
    n_threads : int
        Number of threads (0 = auto)
    particle_type : int
        Particle type for mass lookup

    Returns
    -------
    density_3d : ndarray, shape (N_cell, N_cell, N_cell)
        3D density field in code units (10^10 M_sun/h / (kpc/h)^3)
    config : TetraDensityConfig
        Configuration used for computation
    """
    # Infer grid size from particle count
    lagrangian_grid_size = infer_grid_size(len(positions))

    if grid_resolution is None:
        grid_resolution = lagrangian_grid_size

    box_size = header.box_size

    # Get particle mass from header
    mass_per_particle = header.mass_table[particle_type]
    if mass_per_particle == 0:
        mass_per_particle = 1.0

    # Sort by Lagrangian ID using utils function
    print("Sorting particles by Lagrangian ID...")
    sorted_positions = sort_positions_lagrangian(
        positions, particle_ids, lagrangian_grid_size
    )

    # Set up tessellation config
    config = ts.density.TetraDensityConfig()
    config.box_size = box_size
    config.lagrangian_grid_size = lagrangian_grid_size
    config.output_cells = grid_resolution
    config.n_samples = n_samples
    config.particle_mass = mass_per_particle
    config.n_threads = n_threads

    # Compute 3D density
    print(f"Computing 3D tessellation density field ({grid_resolution}^3)...")
    result = ts.density.compute_tetra_density_3d(sorted_positions, config)
    
    density_3d = np.array(result.density).reshape(
        (grid_resolution, grid_resolution, grid_resolution)
    )
    
    print(f"  Tetrahedra processed: {result.n_tetrahedra:,}")
    print(f"  Mean density: {result.mean_density:.6e}")
    
    return density_3d, config, result


def density_to_physical_units(density, box_size, h=0.6774):
    """
    Convert density from code units to physical units.
    
    GADGET-4 code units:
    - Mass: 10^10 M_sun/h
    - Length: kpc/h (typically) or Mpc/h
    
    This function converts to h^-2 M_sun / Mpc^3.
    
    Parameters
    ----------
    density : ndarray
        Density in code units (10^10 M_sun/h / (length_unit)^3)
    box_size : float
        Box size in code units. If < 1000, assumed to be Mpc/h; else kpc/h.
    h : float
        Hubble parameter h = H0/100
    
    Returns
    -------
    density_physical : ndarray
        Density in h^-2 M_sun / Mpc^3
    """
    # Determine if box_size is in Mpc/h or kpc/h
    # Convention: boxes < 1000 are in Mpc/h, larger in kpc/h
    if box_size < 1000:
        # Box in Mpc/h, density in 10^10 M_sun/h / (Mpc/h)^3
        # Convert to h^-2 M_sun / Mpc^3
        # density * 10^10 / h^2 [M_sun/Mpc^3 * h^2] -> h^-2 M_sun/Mpc^3
        density_physical = density * 1e10  # Now in M_sun/h / (Mpc/h)^3 = M_sun h^2 / Mpc^3
    else:
        # Box in kpc/h, density in 10^10 M_sun/h / (kpc/h)^3
        # 1 Mpc = 1000 kpc, so (kpc/h)^3 = (Mpc/h)^3 / 10^9
        # density * 10^10 * 10^9 = density * 10^19 [M_sun h^2 / Mpc^3]
        density_physical = density * 1e10 * 1e9  # M_sun h^2 / Mpc^3
    
    return density_physical


def compute_pdf(density, n_bins=100, log_min=None, log_max=None, n_threads=0):
    """
    Compute the PDF of the density field using logarithmic bins.
    
    Parameters
    ----------
    density : ndarray
        Flattened density values (overdensity 1+δ)
    n_bins : int
        Number of logarithmic bins
    log_min, log_max : float, optional
        Log10 of bin range (auto-detect if None)
    n_threads : int
        Number of threads
    
    Returns
    -------
    bin_centers : ndarray
        Bin center values (geometric mean)
    bin_edges : ndarray
        Bin edge values
    pdf : ndarray
        PDF values P(ρ)
    counts : ndarray
        Raw bin counts
    """
    density_flat = density.ravel().astype(np.float64)
    
    # Filter out non-positive values
    mask = density_flat > 0
    n_zero = np.sum(~mask)
    if n_zero > 0:
        print(f"  Warning: {n_zero} cells with density <= 0 (excluded from PDF)")
    density_flat = density_flat[mask]
    
    # Auto-detect range
    if log_min is None:
        log_min = np.log10(density_flat.min())
    if log_max is None:
        log_max = np.log10(density_flat.max()) + 0.01  # Small padding
    
    range_min = 10 ** log_min
    range_max = 10 ** log_max
    
    # Use C++ histogram
    hist_result = ts.stats.histogram_log(
        density_flat, n_bins, range_min, range_max, n_threads
    )
    
    # Compute PDF
    pdf = ts.stats.histogram_to_pdf(hist_result)
    
    # Get bin centers (geometric mean for log bins)
    bin_centers = ts.stats.bin_centers(hist_result.bin_edges, log_space=True)
    
    return bin_centers, hist_result.bin_edges, pdf, hist_result.counts


def save_pdf_hdf5(filename, density_3d, overdensity, bin_centers, bin_edges, 
                  pdf, counts, header, config, rho_mean_physical):
    """
    Save PDF results to HDF5 file.
    """
    with h5py.File(filename, 'w') as f:
        # Store density statistics (not full field - too large)
        stats = f.create_group('DensityStats')
        stats.attrs['min'] = float(density_3d.min())
        stats.attrs['max'] = float(density_3d.max())
        stats.attrs['mean'] = float(density_3d.mean())
        stats.attrs['std'] = float(density_3d.std())
        stats.attrs['n_cells'] = density_3d.size
        stats.attrs['grid_resolution'] = config.output_cells
        stats.attrs['rho_mean_physical'] = rho_mean_physical
        stats.attrs['units'] = 'h^-2 M_sun / Mpc^3'
        
        # Overdensity statistics
        od_stats = f.create_group('OverdensityStats')
        od_stats.attrs['min'] = float(overdensity.min())
        od_stats.attrs['max'] = float(overdensity.max())
        od_stats.attrs['mean'] = float(overdensity.mean())
        od_stats.attrs['std'] = float(overdensity.std())
        od_stats.attrs['description'] = '1 + delta = rho / rho_mean'
        
        # PDF data
        pdf_grp = f.create_group('PDF')
        pdf_grp.create_dataset('bin_centers', data=bin_centers)
        pdf_grp.create_dataset('bin_edges', data=bin_edges)
        pdf_grp.create_dataset('pdf', data=pdf)
        pdf_grp.create_dataset('counts', data=counts)
        pdf_grp.attrs['n_bins'] = len(bin_centers)
        pdf_grp.attrs['variable'] = '1 + delta (overdensity)'
        pdf_grp.attrs['normalization'] = 'integral P(rho) drho = 1'
        
        # Header info
        hdr = f.create_group('Header')
        hdr.attrs['BoxSize'] = header.box_size
        hdr.attrs['Redshift'] = header.redshift
        hdr.attrs['Time'] = header.time
        hdr.attrs['NumPartTotal'] = list(header.num_part_total)
        hdr.attrs['MassTable'] = list(header.mass_table)
        
        # Config info
        cfg = f.create_group('Config')
        cfg.attrs['lagrangian_grid_size'] = config.lagrangian_grid_size
        cfg.attrs['output_cells'] = config.output_cells
        cfg.attrs['n_samples'] = config.n_samples
        cfg.attrs['particle_mass'] = config.particle_mass
    
    print(f"Saved PDF results to {filename}")


def plot_pdf(bin_centers, bin_edges, pdf, counts, header, output_file=None, 
             log_min=None, log_max=None):
    """
    Plot the density PDF as ρ²P(ρ) vs (1+δ) with histogram overlay.
    
    Plotting ρ²P(ρ) is common practice to reduce dynamic range
    and highlight features in the PDF.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute bin widths for bar plot
    bin_widths = np.diff(bin_edges)
    
    # Normalize counts to match PDF scale for overlay
    counts_normalized = counts / (counts.sum() * bin_widths)  # Same normalization as PDF
    
    # Left panel: P(ρ) vs (1+δ)
    mask = pdf > 0
    
    # Histogram bars (using step plot for log-scale compatibility)
    ax1.fill_between(bin_centers, 0, counts_normalized, alpha=0.3, color='blue', 
                     step='mid', label='Histogram')
    # PDF line
    ax1.loglog(bin_centers[mask], pdf[mask], 'b-', lw=2, label='PDF')
    ax1.set_xlabel(r'$1 + \delta$', fontsize=12)
    ax1.set_ylabel(r'$P(\rho)$', fontsize=12)
    ax1.set_title(f'Density PDF (z = {header.redshift:.2f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    if log_min is not None and log_max is not None:
        ax1.set_xlim(10**log_min, 10**log_max)
    
    # Right panel: ρ²P(ρ) vs (1+δ) - standard plotting convention
    rho2_P = bin_centers**2 * pdf
    rho2_counts = bin_centers**2 * counts_normalized
    
    # Histogram bars
    ax2.fill_between(bin_centers, 0, rho2_counts, alpha=0.3, color='red',
                     step='mid', label='Histogram')
    # PDF line
    ax2.semilogx(bin_centers[mask], rho2_P[mask], 'r-', lw=2, label=r'$\rho^2 P(\rho)$')
    ax2.set_xlabel(r'$1 + \delta$', fontsize=12)
    ax2.set_ylabel(r'$\rho^2 P(\rho)$', fontsize=12)
    ax2.set_title(r'Density PDF ($\rho^2$-weighted)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(1.0, color='k', ls='--', alpha=0.5, label=r'$\rho = \bar{\rho}$')
    ax2.legend(loc='upper right')
    
    if log_min is not None and log_max is not None:
        ax2.set_xlim(10**log_min, 10**log_max)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute density PDF from a GADGET-4 snapshot using tessellation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python density_pdf.py snapshot_034.hdf5 -o density_pdf_034.h5

  # With visualization
  python density_pdf.py snapshot_034.hdf5 -o density_pdf.h5 --plot density_pdf.png

  # Custom binning
  python density_pdf.py snapshot_034.hdf5 -o pdf.h5 --n-bins 200 --log-min -2 --log-max 3
        """
    )
    
    parser.add_argument('snapshot', type=str,
                        help='Path to snapshot file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 filename')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output image filename (optional)')
    
    # Grid configuration
    parser.add_argument('--resolution', type=int, default=None,
                        help='Grid resolution (default: same as Lagrangian grid)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples per tetrahedron (default: 100)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='Particle type (default: 1 for DM)')
    
    # PDF configuration
    parser.add_argument('--n-bins', type=int, default=100,
                        help='Number of PDF bins (default: 100)')
    parser.add_argument('--log-min', type=float, default=None,
                        help='Log10 of minimum bin edge (auto-detect if not set)')
    parser.add_argument('--log-max', type=float, default=None,
                        help='Log10 of maximum bin edge (auto-detect if not set)')
    
    args = parser.parse_args()
    
    # Check C++ module
    if not HAS_CPP:
        print("Error: C++ tessellation module not available.")
        print("Make sure the build directory is in PYTHONPATH.")
        sys.exit(1)
    
    # Load snapshot
    print(f"Loading snapshot from {args.snapshot}...")
    positions, particle_ids, header = load_snapshot(
        args.snapshot, particle_type=args.particle_type, read_ids=True
    )
    
    print(f"Box size: {header.box_size}")
    print(f"Redshift: {header.redshift:.4f}")
    print(f"Particles: {len(positions):,}")
    
    # Compute density field
    density_3d, config, result = compute_density_field(
        positions, particle_ids, header,
        grid_resolution=args.resolution,
        n_samples=args.samples,
        n_threads=args.threads,
        particle_type=args.particle_type
    )
    
    # Convert to physical units
    print("\nConverting to physical units...")
    density_physical = density_to_physical_units(density_3d, header.box_size)
    rho_mean_physical = density_physical.mean()
    print(f"  Mean density: {rho_mean_physical:.6e} h^-2 M_sun/Mpc^3")
    
    # Compute overdensity (1 + δ)
    overdensity = density_3d / density_3d.mean()  # Use actual mean for normalization
    
    print(f"\nOverdensity (1+δ) statistics:")
    print(f"  Min: {overdensity.min():.6f}")
    print(f"  Max: {overdensity.max():.6f}")
    print(f"  Mean: {overdensity.mean():.6f}")
    
    # Compute PDF
    print(f"\nComputing PDF with {args.n_bins} bins...")
    bin_centers, bin_edges, pdf, counts = compute_pdf(
        overdensity,
        n_bins=args.n_bins,
        log_min=args.log_min,
        log_max=args.log_max,
        n_threads=args.threads
    )
    
    print(f"  Bin range: [{bin_edges[0]:.4f}, {bin_edges[-1]:.4f}]")
    print(f"  Peak at 1+δ = {bin_centers[np.argmax(pdf)]:.4f}")
    
    # Verify normalization
    bin_widths = np.diff(bin_edges)
    integral_1 = np.sum(pdf * bin_widths)
    integral_rho = np.sum(bin_centers * pdf * bin_widths)
    print(f"  ∫P(ρ)dρ = {integral_1:.6f} (should be ~1)")
    print(f"  ∫ρP(ρ)dρ = {integral_rho:.6f} (should be ~1)")
    
    # Save results
    save_pdf_hdf5(
        args.output,
        density_physical, overdensity,
        bin_centers, bin_edges, pdf, counts,
        header, config, rho_mean_physical
    )
    
    # Plot
    if args.plot:
        print("\nGenerating visualization...")
        plot_pdf(bin_centers, bin_edges, pdf, counts, header, 
                 output_file=args.plot,
                 log_min=args.log_min, log_max=args.log_max)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
