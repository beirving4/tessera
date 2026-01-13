#!/usr/bin/env python3
"""
Example: ORIGAMI Morphology Classification

This script demonstrates how to:
1. Load a GADGET-4 snapshot
2. Compute ORIGAMI morphology classification (void/wall/filament/halo)
3. Compute density field and sample at particle positions
4. Generate conditional PDFs P(1+delta | morphology class)
5. Save results to HDF5 and create visualizations

Reference: Falck, Neyrinck, & Szalay 2012, ApJ 754, 126

Usage:
    python origami_morphology.py snapshot_034.hdf5 -o origami_snap034.h5 --plot origami_snap034.png
"""

import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Try to find the tessera module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

# Import tessera
try:
    import _tessera as ts
    HAS_AT = True
except ImportError:
    try:
        import tessera as ts
        HAS_AT = True
    except ImportError:
        HAS_AT = False
        print("Error: tessera not found. Build the project first.")
        sys.exit(1)


def detect_id_ordering(positions, particle_ids, box_size, grid_size):
    """Detect whether particle IDs use x-major or z-major ordering.

    GADGET-4 typically uses z-major: ID = 1 + z + y*N + x*N^2
    Some codes use x-major: ID = 1 + x + y*N + z*N^2

    We detect by checking which interpretation gives positions that
    vary correctly along each axis.
    """
    spacing = box_size / grid_size
    n_particles = len(positions)

    # Check a few diagnostic particles
    # For x-major: ID=2 should be at x=1, y=0, z=0 (position ~(1,0,0)*spacing)
    # For z-major: ID=2 should be at x=0, y=0, z=1 (position ~(0,0,1)*spacing)

    # Get position of particle with ID=2
    idx_id2 = np.where(particle_ids == 2)[0]
    if len(idx_id2) == 0:
        print("  Warning: Could not find particle ID=2, defaulting to z-major")
        return 'z-major'

    pos_id2 = positions[idx_id2[0]] % box_size

    # For x-major, ID=2 -> (1,0,0), so x should be ~spacing, y,z should be ~0
    x_major_expected_x = spacing
    x_major_score = abs(pos_id2[0] - x_major_expected_x)

    # For z-major, ID=2 -> (0,0,1), so z should be ~spacing, x,y should be ~0
    z_major_expected_z = spacing
    z_major_score = abs(pos_id2[2] - z_major_expected_z)

    # Also check particle ID = N+1 (second row)
    idx_id_np1 = np.where(particle_ids == grid_size + 1)[0]
    if len(idx_id_np1) > 0:
        pos_np1 = positions[idx_id_np1[0]] % box_size
        # x-major: ID=N+1 -> (0,1,0), y should be ~spacing
        x_major_score += abs(pos_np1[1] - spacing)
        # z-major: ID=N+1 -> (0,1,0), y should be ~spacing (same!)
        z_major_score += abs(pos_np1[1] - spacing)

    # Check ID = N^2+1
    idx_id_n2p1 = np.where(particle_ids == grid_size*grid_size + 1)[0]
    if len(idx_id_n2p1) > 0:
        pos_n2p1 = positions[idx_id_n2p1[0]] % box_size
        # x-major: ID=N^2+1 -> (0,0,1), z should be ~spacing
        x_major_score += abs(pos_n2p1[2] - spacing)
        # z-major: ID=N^2+1 -> (1,0,0), x should be ~spacing
        z_major_score += abs(pos_n2p1[0] - spacing)

    if x_major_score < z_major_score:
        return 'x-major'
    else:
        return 'z-major'


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot.

    Handles multi-file snapshots by detecting and loading all chunks.
    For ORIGAMI, we need the complete particle set sorted by Lagrangian ID.
    """
    snapshot_path = Path(snapshot_path)
    print(f"Loading snapshot from {snapshot_path}...")

    # Detect multi-file snapshot
    with h5py.File(snapshot_path, 'r') as f:
        num_files = f['Header'].attrs.get('NumFilesPerSnapshot', 1)
        box_size = f['Header'].attrs['BoxSize']
        redshift = f['Header'].attrs['Redshift']
        time = f['Header'].attrs['Time']
        num_part_total = f['Header'].attrs['NumPart_Total'][particle_type]

        # Get mass
        part_key = f'PartType{particle_type}'
        if part_key in f and 'Masses' in f[part_key]:
            particle_mass = f[f'{part_key}/Masses'][0]
        else:
            particle_mass = f['Header'].attrs['MassTable'][particle_type]

    print(f"  Box size: {box_size}")
    print(f"  Redshift: {redshift:.4f}")
    print(f"  Scale factor: {time:.4f}")
    print(f"  Total particles: {num_part_total:,}")
    print(f"  Number of files: {num_files}")

    if num_files == 1:
        # Single file - load directly
        with h5py.File(snapshot_path, 'r') as f:
            positions = f[f'{part_key}/Coordinates'][:]
            particle_ids = f[f'{part_key}/ParticleIDs'][:]
    else:
        # Multi-file snapshot - load all chunks
        # Determine file naming pattern
        base_name = str(snapshot_path)
        if '.0.hdf5' in base_name:
            base_pattern = base_name.replace('.0.hdf5', '.{}.hdf5')
        elif '_0.hdf5' in base_name:
            base_pattern = base_name.replace('_0.hdf5', '_{}.hdf5')
        else:
            # Assume single file if pattern not detected
            with h5py.File(snapshot_path, 'r') as f:
                positions = f[f'{part_key}/Coordinates'][:]
                particle_ids = f[f'{part_key}/ParticleIDs'][:]
            return positions, particle_ids, box_size, redshift, time, particle_mass

        print(f"  Loading {num_files} chunks...")
        all_positions = []
        all_ids = []

        for i in range(num_files):
            chunk_path = base_pattern.format(i)
            if not Path(chunk_path).exists():
                raise FileNotFoundError(f"Snapshot chunk not found: {chunk_path}")

            with h5py.File(chunk_path, 'r') as f:
                if part_key in f:
                    all_positions.append(f[f'{part_key}/Coordinates'][:])
                    all_ids.append(f[f'{part_key}/ParticleIDs'][:])

            if (i + 1) % 10 == 0 or i == num_files - 1:
                print(f"    Loaded {i + 1}/{num_files} chunks...")

        positions = np.concatenate(all_positions, axis=0)
        particle_ids = np.concatenate(all_ids, axis=0)

        print(f"  Loaded {len(positions):,} particles from {num_files} files")

    # Verify we have the expected number of particles
    if len(positions) != num_part_total:
        print(f"  WARNING: Loaded {len(positions):,} particles, expected {num_part_total:,}")

    return positions, particle_ids, box_size, redshift, time, particle_mass


def compute_origami_morphology(positions, particle_ids, box_size, n_threads=0,
                                id_ordering='auto'):
    """Compute ORIGAMI morphology classification.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions.
    particle_ids : ndarray, shape (N,)
        Particle IDs (1-indexed Lagrangian grid positions).
    box_size : float
        Simulation box size.
    n_threads : int
        Number of OpenMP threads (0 = auto).
    id_ordering : str
        How particle IDs encode Lagrangian positions:
        - 'auto': Automatically detect (default)
        - 'x-major': ID = 1 + x + y*N + z*N^2 (x varies fastest)
        - 'z-major': ID = 1 + z + y*N + x*N^2 (z varies fastest, GADGET-4 default)
    """
    print("\nComputing ORIGAMI morphology...")

    # Determine grid size from particle count
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))
    if grid_size ** 3 != n_particles:
        raise ValueError(f"Particle count {n_particles} is not a perfect cube. "
                         f"Nearest cube root: {grid_size} ({grid_size**3} particles)")

    print(f"  Lagrangian grid: {grid_size}^3")

    # Validate particle IDs
    print("  Validating particle IDs...")
    min_id, max_id = particle_ids.min(), particle_ids.max()
    unique_ids = len(np.unique(particle_ids))
    print(f"    ID range: [{min_id:,}, {max_id:,}]")
    print(f"    Unique IDs: {unique_ids:,}")

    if min_id != 1 or max_id != n_particles:
        print(f"    WARNING: IDs should range from 1 to {n_particles:,}")
        print(f"    This may indicate incomplete snapshot loading or non-Lagrangian IDs")

    if unique_ids != n_particles:
        raise ValueError(f"Found {unique_ids:,} unique IDs for {n_particles:,} particles. "
                         "IDs must be unique for ORIGAMI.")

    # Detect or use specified ID ordering
    if id_ordering == 'auto':
        id_ordering = detect_id_ordering(positions, particle_ids, box_size, grid_size)
        print(f"  Detected ID ordering: {id_ordering}")

    # Sort particles to x-major order (what ORIGAMI expects)
    print(f"  Sorting particles to x-major order (from {id_ordering})...")
    sorted_positions = np.zeros((n_particles, 3), dtype=np.float64)

    if id_ordering == 'x-major':
        # Direct mapping: ID-1 gives x-major index
        sorted_positions[particle_ids - 1] = positions
    elif id_ordering == 'z-major':
        # Convert z-major IDs to x-major indices
        # z-major: ID = 1 + z + y*N + x*N^2
        # x-major: idx = x + y*N + z*N^2
        for i in range(n_particles):
            idx = particle_ids[i] - 1  # 0-indexed z-major
            z = idx % grid_size
            y = (idx // grid_size) % grid_size
            x = idx // (grid_size * grid_size)
            xmajor_idx = x + y * grid_size + z * grid_size * grid_size
            sorted_positions[xmajor_idx] = positions[i]
    else:
        raise ValueError(f"Unknown id_ordering: {id_ordering}")

    sorted_positions = np.ascontiguousarray(sorted_positions)

    # Compute and report displacement statistics
    spacing = box_size / grid_size
    print(f"  Lagrangian spacing: {spacing:.4f} Mpc/h")

    # Create ORIGAMI config
    # Note: Using n_split=1 to avoid OpenMP issues with duplicate libomp on macOS
    # This runs single-threaded but is still reasonably fast
    config = at.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1  # Single domain to avoid OpenMP issues

    # Compute morphology
    print("  Running ORIGAMI algorithm...")
    result = at.origami.compute_morphology(sorted_positions, config)

    print(f"\n  Results:")
    print(f"    Void:     {result.n_void:>10,} particles ({result.f_void:>6.2%})")
    print(f"    Wall:     {result.n_wall:>10,} particles ({result.f_wall:>6.2%})")
    print(f"    Filament: {result.n_filament:>10,} particles ({result.f_filament:>6.2%})")
    print(f"    Halo:     {result.n_halo:>10,} particles ({result.f_halo:>6.2%})")

    return result, sorted_positions, grid_size


def compute_density_and_sample(sorted_positions, box_size, grid_size, result,
                                output_cells=128, n_samples=100, n_threads=0):
    """Compute density field and sample at particle positions."""
    print(f"\nComputing density field ({output_cells}^3)...")

    # Create density config
    density_config = at.density.TetraDensityConfig()
    density_config.lagrangian_grid_size = grid_size
    density_config.output_cells = output_cells
    density_config.box_size = float(box_size)
    density_config.particle_mass = 1.0  # Will normalize
    density_config.n_samples = n_samples
    density_config.n_threads = n_threads

    # Compute 3D density
    density_result = at.density.compute_tetra_density_3d(sorted_positions, density_config)

    # Reshape density to 3D array [z, y, x]
    density_3d = np.array(density_result.density).reshape(
        output_cells, output_cells, output_cells
    )

    print(f"  Density range: [{density_3d.min():.4e}, {density_3d.max():.4e}]")
    print(f"  Mean density: {density_3d.mean():.4e}")

    # Sample density at particle positions
    print("  Sampling density at particle positions...")
    at.origami.sample_density_at_particles(
        density_3d, sorted_positions, box_size, result
    )

    return density_3d, density_result


def deposit_to_grid(sorted_positions, box_size, result, grid_cells=128):
    """Deposit morphology classifications onto a grid."""
    print(f"\nDepositing morphology to {grid_cells}^3 grid...")

    at.origami.deposit_morphology_to_grid(
        sorted_positions, box_size, grid_cells, result
    )

    print("  Grid outputs computed:")
    print(f"    morphology_grid shape: {result.morphology_grid.shape}")
    print(f"    halo_fraction range: [{result.halo_fraction_grid.min():.3f}, {result.halo_fraction_grid.max():.3f}]")


def save_results(output_path, result, density_3d, positions, box_size, redshift,
                 scale_factor, grid_size, particle_mass):
    """Save results to HDF5."""
    print(f"\nSaving results to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Header
        hdr = f.create_group('Header')
        hdr.attrs['box_size'] = box_size
        hdr.attrs['redshift'] = redshift
        hdr.attrs['scale_factor'] = scale_factor
        hdr.attrs['n_particles'] = len(positions)
        hdr.attrs['lagrangian_grid_size'] = grid_size
        hdr.attrs['particle_mass'] = particle_mass

        # Per-particle data
        per_part = f.create_group('PerParticle')
        per_part.create_dataset('morphology', data=np.array(result.morphology),
                                 compression='gzip')
        if len(result.particle_density) > 0:
            per_part.create_dataset('density', data=np.array(result.particle_density),
                                     compression='gzip')

        # Grid data
        if len(result.morphology_grid) > 0:
            grid = f.create_group('Grid')
            grid.attrs['cells'] = result.grid_cells
            grid.create_dataset('morphology', data=np.array(result.morphology_grid),
                                 compression='gzip')
            grid.create_dataset('void_fraction', data=np.array(result.void_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('wall_fraction', data=np.array(result.wall_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('filament_fraction', data=np.array(result.filament_fraction_grid),
                                 compression='gzip')
            grid.create_dataset('halo_fraction', data=np.array(result.halo_fraction_grid),
                                 compression='gzip')

        # Density field
        if density_3d is not None:
            f.create_dataset('density_3d', data=density_3d, compression='gzip')

        # Statistics
        stats = f.create_group('Statistics')
        stats.create_dataset('counts', data=np.array(result.counts))
        stats.create_dataset('mass_fractions', data=np.array(result.fractions))

        # Compute mean density per class
        if len(result.particle_density) > 0:
            morph = np.array(result.morphology)
            density = np.array(result.particle_density)
            mean_densities = np.array([
                density[morph == i].mean() if np.sum(morph == i) > 0 else 0.0
                for i in range(4)
            ])
            stats.create_dataset('mean_density_per_class', data=mean_densities)

    print("  Saved successfully")


def create_visualization(output_path, result, density_3d, box_size):
    """Create visualization of ORIGAMI results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    print(f"\nCreating visualization...")

    fig = plt.figure(figsize=(16, 12))

    # Get middle slice index
    mid = density_3d.shape[0] // 2

    # 1. Density slice
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(density_3d[mid, :, :], origin='lower', cmap='magma',
                      norm=LogNorm(), extent=[0, box_size, 0, box_size])
    ax1.set_title('Density (z-slice)')
    ax1.set_xlabel('x [Mpc/h]')
    ax1.set_ylabel('y [Mpc/h]')
    plt.colorbar(im1, ax=ax1, label='Density')

    # 2. Morphology grid slice
    ax2 = fig.add_subplot(2, 3, 2)
    morph_grid = np.array(result.morphology_grid)
    im2 = ax2.imshow(morph_grid[mid, :, :], origin='lower', cmap='viridis',
                      vmin=0, vmax=3, extent=[0, box_size, 0, box_size])
    ax2.set_title('Dominant Morphology (z-slice)')
    ax2.set_xlabel('x [Mpc/h]')
    ax2.set_ylabel('y [Mpc/h]')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
    cbar2.set_ticklabels(['Void', 'Wall', 'Filament', 'Halo'])

    # 3. Halo fraction slice
    ax3 = fig.add_subplot(2, 3, 3)
    halo_frac = np.array(result.halo_fraction_grid)
    im3 = ax3.imshow(halo_frac[mid, :, :], origin='lower', cmap='Reds',
                      vmin=0, vmax=1, extent=[0, box_size, 0, box_size])
    ax3.set_title('Halo Fraction (z-slice)')
    ax3.set_xlabel('x [Mpc/h]')
    ax3.set_ylabel('y [Mpc/h]')
    plt.colorbar(im3, ax=ax3, label='Halo fraction')

    # 4. Mass fractions bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    classes = ['Void', 'Wall', 'Filament', 'Halo']
    fractions = [result.f_void, result.f_wall, result.f_filament, result.f_halo]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax4.bar(classes, fractions, color=colors)
    ax4.set_ylabel('Mass Fraction')
    ax4.set_title('Morphology Distribution')
    ax4.set_ylim(0, max(fractions) * 1.2)
    for bar, frac in zip(bars, fractions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{frac:.1%}', ha='center', va='bottom', fontsize=10)

    # 5. Conditional density PDFs
    ax5 = fig.add_subplot(2, 3, 5)
    morph = np.array(result.morphology)
    density = np.array(result.particle_density)

    # Normalize by mean
    mean_density = density.mean()
    overdensity = density / mean_density

    bins = np.logspace(-2, 4, 100)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    for i, (name, color) in enumerate(zip(classes, colors)):
        mask = morph == i
        if np.sum(mask) > 0:
            hist, _ = np.histogram(overdensity[mask], bins=bins, density=True)
            ax5.plot(bin_centers, hist, label=name, color=color, linewidth=2)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel(r'$1 + \delta$')
    ax5.set_ylabel('PDF')
    ax5.set_title(r'Conditional PDF: $P(1+\delta | \mathrm{class})$')
    ax5.legend()
    ax5.set_xlim(1e-2, 1e4)
    ax5.grid(True, alpha=0.3)

    # 6. Mean overdensity per class
    ax6 = fig.add_subplot(2, 3, 6)
    mean_overdensities = []
    for i in range(4):
        mask = morph == i
        if np.sum(mask) > 0:
            mean_overdensities.append(overdensity[mask].mean())
        else:
            mean_overdensities.append(0)

    bars = ax6.bar(classes, mean_overdensities, color=colors)
    ax6.axhline(y=1.0, color='black', linestyle='--', label='Mean')
    ax6.set_ylabel(r'Mean $1 + \delta$')
    ax6.set_title('Mean Overdensity by Class')
    ax6.set_yscale('log')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute ORIGAMI morphology classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('snapshot', type=str,
                        help='Path to GADGET-4 HDF5 snapshot')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output plot path (optional)')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Density/morphology grid resolution (default: 128)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Monte Carlo samples for density (default: 100)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0=auto)')
    parser.add_argument('--particle-type', type=int, default=1,
                        help='GADGET particle type (default: 1 for DM)')

    args = parser.parse_args()

    # Load snapshot
    positions, particle_ids, box_size, redshift, scale_factor, particle_mass = \
        load_snapshot(args.snapshot, args.particle_type)

    # Compute ORIGAMI morphology
    result, sorted_positions, grid_size = compute_origami_morphology(
        positions, particle_ids, box_size, args.threads
    )

    # Compute density and sample at particles
    density_3d, density_result = compute_density_and_sample(
        sorted_positions, box_size, grid_size, result,
        args.resolution, args.samples, args.threads
    )

    # Deposit morphology to grid
    deposit_to_grid(sorted_positions, box_size, result, args.resolution)

    # Save results
    save_results(args.output, result, density_3d, sorted_positions, box_size,
                 redshift, scale_factor, grid_size, particle_mass)

    # Create visualization
    if args.plot:
        create_visualization(args.plot, result, density_3d, box_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
