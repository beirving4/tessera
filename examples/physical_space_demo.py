#!/usr/bin/env python3
"""
Physical Space Density Computation Demo

This script demonstrates computing tetrahedron-based density fields in physical
coordinates rather than comoving coordinates. This is useful for comparing
structures across different scale factors at the same physical scale.

At late times (a > 1), fixed structures in physical space appear to shrink in
comoving coordinates. By computing density in physical space, we can:
1. Compare the same halo at different epochs at consistent physical scales
2. See the true physical density distribution

The output grid uses comoving coordinates for the extent (matching simulation
conventions), but the density values represent physical space computations.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()

# Import shared utilities
from tessera.utils import load_snapshot_h5py, infer_grid_size

# Try to import local config
try:
    from config import SNAPSHOT_BASE, HALO_CENTERS
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


def load_snapshot(snapshot_name):
    """Load particle data from GADGET-4 snapshot."""
    import h5py
    snapshot_path = SNAPSHOT_BASE / f"{snapshot_name}.hdf5"

    # Use utils function for loading
    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=1, read_ids=True
    )

    # Read redshift separately (not returned by load_snapshot_h5py)
    with h5py.File(snapshot_path, 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])

    return positions, particle_ids, box_size, redshift, scale_factor


def compute_density_2d(positions, particle_ids, box_size, grid_size, output_cells,
                       subbox=None, scale_factor_compute=1.0):
    """
    Compute 2D z-projected density field.

    Args:
        positions: Particle positions (comoving coordinates)
        particle_ids: Particle IDs for Lagrangian sorting
        box_size: Box size (comoving)
        grid_size: Lagrangian grid size
        output_cells: Output grid resolution
        subbox: Dict with 'center' and 'width' (comoving), or None for full box
        scale_factor_compute: Scale factor for physical space computation.
                              If 1.0, compute in comoving space.
                              If != 1.0, scale positions and box to physical coords.

    Returns:
        2D density array
    """
    # Sort positions by Lagrangian ID
    sorted_positions = ts.density.sort_by_lagrangian_id(positions, particle_ids, grid_size)

    # Scale to physical coordinates if requested
    a = scale_factor_compute
    if a != 1.0:
        positions_use = sorted_positions * a
        box_size_use = box_size * a
    else:
        positions_use = sorted_positions
        box_size_use = box_size

    config = ts.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = box_size_use
    config.output_cells = output_cells
    config.n_threads = 1
    config.n_samples = 50
    config.periodic = True
    config.particle_mass = 1.0

    if subbox is not None:
        config.subbox_enabled = True
        center = subbox['center']
        width = subbox['width']
        if a != 1.0:
            center_use = tuple(c * a for c in center)
            width_use = width * a
        else:
            center_use = center
            width_use = width
        config.subbox_origin = (center_use[0] - width_use/2,
                                center_use[1] - width_use/2,
                                center_use[2] - width_use/2)
        config.subbox_width = (width_use, width_use, width_use)

    result = ts.density.compute_tetra_density_2d_projection(positions_use, config, 2)
    return result.density


def create_comparison_figure(density_comoving, density_physical, extent_comoving,
                             title, output_path, scale_factor):
    """Create side-by-side comparison of comoving and physical space densities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize for display
    com_norm = density_comoving / density_comoving.mean()
    phys_norm = density_physical / density_physical.mean()

    # Use same color scale
    vmin = min(np.log10(np.maximum(com_norm, 0.01).min()),
               np.log10(np.maximum(phys_norm, 0.01).min()))
    vmax = max(np.log10(com_norm.max()), np.log10(phys_norm.max()))

    # Left: Comoving space
    im1 = axes[0].imshow(np.log10(np.maximum(com_norm, 0.01)), origin='lower',
                         cmap='magma', vmin=vmin, vmax=vmax, extent=extent_comoving)
    axes[0].set_title(f'Comoving Space')
    axes[0].set_xlabel('cMpc/h')
    axes[0].set_ylabel('cMpc/h')
    plt.colorbar(im1, ax=axes[0], label=r'$\log_{10}(1+\delta)$')

    # Middle: Physical space (same comoving extent for grid)
    im2 = axes[1].imshow(np.log10(np.maximum(phys_norm, 0.01)), origin='lower',
                         cmap='magma', vmin=vmin, vmax=vmax, extent=extent_comoving)
    axes[1].set_title(f'Physical Space (a={scale_factor:.0f})')
    axes[1].set_xlabel('cMpc/h')
    axes[1].set_ylabel('cMpc/h')
    plt.colorbar(im2, ax=axes[1], label=r'$\log_{10}(1+\delta)$')

    # Right: Ratio (physical/comoving normalized)
    # After normalization, these should be identical (correlation = 1)
    ratio = phys_norm / com_norm
    ratio_centered = ratio - 1  # Center around 0
    abs_max = np.percentile(np.abs(ratio_centered[np.isfinite(ratio_centered)]), 99)
    im3 = axes[2].imshow(ratio_centered, origin='lower', cmap='RdBu_r',
                         vmin=-abs_max, vmax=abs_max, extent=extent_comoving)
    axes[2].set_title('Normalized Ratio - 1')
    axes[2].set_xlabel('cMpc/h')
    axes[2].set_ylabel('cMpc/h')
    plt.colorbar(im3, ax=axes[2], label='(Physical/Comoving) - 1')

    # Compute correlation
    corr = np.corrcoef(com_norm.flatten(), phys_norm.flatten())[0, 1]

    # Physical width annotation
    width_comoving = extent_comoving[1] - extent_comoving[0]
    width_physical = width_comoving * scale_factor

    plt.suptitle(f'{title}\n'
                 f'Comoving width: {width_comoving:.1f} cMpc/h | '
                 f'Physical width: {width_physical:.1f} Mpc/h | '
                 f'Correlation: {corr:.6f}',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("Physical Space Density Computation Demo")
    print("=" * 70)

    output_cells = 256  # Resolution for density grid

    # ==========================================================================
    # Full Box Comparisons
    # ==========================================================================
    print("\n--- Full Box Comparisons ---")

    for snap_name, desc in [('snapshot_034', 'a=1'), ('snapshot_074', 'a=100')]:
        print(f"\n{snap_name} ({desc}):")

        positions, particle_ids, box_size, redshift, scale_factor = load_snapshot(snap_name)
        grid_size = int(round(len(positions) ** (1/3)))

        print(f"  Scale factor: {scale_factor:.4f}")
        print(f"  Box size: {box_size} cMpc/h (comoving)")
        print(f"  Physical box: {box_size * scale_factor:.1f} Mpc/h")

        # Compute in comoving space
        density_com = compute_density_2d(positions, particle_ids, box_size, grid_size,
                                         output_cells, scale_factor_compute=1.0)

        # Compute in physical space
        density_phys = compute_density_2d(positions, particle_ids, box_size, grid_size,
                                          output_cells, scale_factor_compute=scale_factor)

        print(f"  Comoving density: mean={density_com.mean():.4f}, max={density_com.max():.4f}")
        print(f"  Physical density: mean={density_phys.mean():.6f}, max={density_phys.max():.4f}")
        print(f"  Ratio of means: {density_com.mean() / density_phys.mean():.4f} (expected a²={scale_factor**2:.4f})")

        # Create comparison figure
        extent = [0, box_size, 0, box_size]
        create_comparison_figure(
            density_com, density_phys, extent,
            f'Full Box {desc}',
            SCRIPT_DIR / f'{snap_name}_fullbox_physical_comparison.png',
            scale_factor
        )

    # ==========================================================================
    # Halo Comparisons
    # ==========================================================================
    print("\n--- Halo Comparisons ---")

    halo_width = 10.0  # cMpc/h comoving

    for snap_name, desc in [('snapshot_034', 'a=1'), ('snapshot_074', 'a=100')]:
        print(f"\n{snap_name} Halo ({desc}):")

        positions, particle_ids, box_size, redshift, scale_factor = load_snapshot(snap_name)
        grid_size = int(round(len(positions) ** (1/3)))
        center = HALO_CENTERS[snap_name]

        print(f"  Halo center (comoving): ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Subbox width: {halo_width} cMpc/h (comoving) = {halo_width * scale_factor:.1f} Mpc/h (physical)")

        subbox = {'center': center, 'width': halo_width}

        # Compute in comoving space
        density_com = compute_density_2d(positions, particle_ids, box_size, grid_size,
                                         128, subbox=subbox, scale_factor_compute=1.0)

        # Compute in physical space
        density_phys = compute_density_2d(positions, particle_ids, box_size, grid_size,
                                          128, subbox=subbox, scale_factor_compute=scale_factor)

        print(f"  Comoving density: mean={density_com.mean():.4f}, max={density_com.max():.4f}")
        print(f"  Physical density: mean={density_phys.mean():.6f}, max={density_phys.max():.4f}")

        # Create comparison figure
        extent = [center[0] - halo_width/2, center[0] + halo_width/2,
                  center[1] - halo_width/2, center[1] + halo_width/2]
        create_comparison_figure(
            density_com, density_phys, extent,
            f'Group 0 Halo {desc}',
            SCRIPT_DIR / f'{snap_name}_halo_physical_comparison.png',
            scale_factor
        )

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
