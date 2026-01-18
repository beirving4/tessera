#!/usr/bin/env python3
"""
ORIGAMI Cosmic Web Time-Series Visualization

Creates a Diemer-style time-series image showing the evolution of cosmic web
morphology (voids, walls, filaments, halos) from early times through the far future.

Each morphology type is rendered in a different color:
- Voids: Dark blue/black
- Walls: Cyan/teal
- Filaments: Yellow/gold
- Halos: White/bright

Usage:
    python time_series_origami.py --snapshot-dir /path/to/snapshots --output-dir /path/to/output

Reference: Falck, Neyrinck & Szalay 2012, ApJ 754, 126
"""

import argparse
import os
from pathlib import Path
import time
import sys
import numpy as np

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Try to find the tessera module
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
_build_dir = _repo_root / 'build'
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

try:
    import _tessera as ts
except ImportError:
    try:
        import tessera as ts
    except ImportError:
        print("Error: tessera not found. Build the project first.")
        sys.exit(1)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Morphology type constants
VOID = 0
WALL = 1
FILAMENT = 2
HALO = 3

# Colors for each morphology type (RGB, 0-1 scale)
MORPHOLOGY_COLORS = {
    VOID: np.array([0.0, 0.02, 0.08]),       # Very dark blue (nearly black)
    WALL: np.array([0.0, 0.4, 0.5]),          # Teal/cyan
    FILAMENT: np.array([0.9, 0.7, 0.1]),      # Gold/yellow
    HALO: np.array([1.0, 1.0, 1.0]),          # White
}


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot."""
    if not HAS_H5PY:
        # Use tessera's built-in IO
        snapshot = ts.io.read_gadget4_snapshot(
            str(snapshot_path),
            particle_types=(1 << particle_type),
            read_velocities=False,
            read_ids=True,
        )
        particles = snapshot.particles[particle_type]
        positions = np.array(particles.coordinates, dtype=np.float64)
        particle_ids = np.array(particles.particle_ids, dtype=np.int64)
        box_size = snapshot.header.box_size
        scale_factor = snapshot.header.time
        return positions, particle_ids, box_size, scale_factor

    # Use h5py for more control
    with h5py.File(snapshot_path, 'r') as f:
        part_key = f'PartType{particle_type}'
        positions = np.ascontiguousarray(f[f'{part_key}/Coordinates'][:], dtype=np.float64)
        particle_ids = np.ascontiguousarray(f[f'{part_key}/ParticleIDs'][:], dtype=np.int64)
        box_size = float(f['Header'].attrs['BoxSize'])
        scale_factor = float(f['Header'].attrs['Time'])

    return positions, particle_ids, box_size, scale_factor


def detect_id_ordering(positions, particle_ids, box_size, grid_size):
    """Detect whether particle IDs use x-major or z-major ordering."""
    spacing = box_size / grid_size

    idx_id2 = np.where(particle_ids == 2)[0]
    if len(idx_id2) == 0:
        return 'z-major'

    pos_id2 = positions[idx_id2[0]] % box_size

    x_major_score = abs(pos_id2[0] - spacing)
    z_major_score = abs(pos_id2[2] - spacing)

    idx_id_n2p1 = np.where(particle_ids == grid_size * grid_size + 1)[0]
    if len(idx_id_n2p1) > 0:
        pos_n2p1 = positions[idx_id_n2p1[0]] % box_size
        x_major_score += abs(pos_n2p1[2] - spacing)
        z_major_score += abs(pos_n2p1[0] - spacing)

    return 'x-major' if x_major_score < z_major_score else 'z-major'


def sort_particles_for_origami(positions, particle_ids, box_size):
    """Sort particles to x-major order for ORIGAMI."""
    n_particles = len(positions)
    grid_size = int(round(n_particles ** (1/3)))

    id_ordering = detect_id_ordering(positions, particle_ids, box_size, grid_size)

    sorted_positions = np.zeros((n_particles, 3), dtype=np.float64)

    if id_ordering == 'x-major':
        sorted_positions[particle_ids - 1] = positions
    else:  # z-major (GADGET-4 default)
        for i in range(n_particles):
            idx = particle_ids[i] - 1
            z = idx % grid_size
            y = (idx // grid_size) % grid_size
            x = idx // (grid_size * grid_size)
            xmajor_idx = x + y * grid_size + z * grid_size * grid_size
            sorted_positions[xmajor_idx] = positions[i]

    return np.ascontiguousarray(sorted_positions), grid_size


def compute_origami(positions, particle_ids, box_size):
    """Compute ORIGAMI morphology classification."""
    sorted_positions, grid_size = sort_particles_for_origami(positions, particle_ids, box_size)

    config = ts.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    result = ts.origami.compute_morphology(sorted_positions, config)

    return result, sorted_positions, grid_size


def project_morphology_to_2d(
    sorted_positions,
    morphology,
    box_size,
    output_cells,
    projection_axis=2,
    slab_fraction=0.08,
    slab_center=0.5,
):
    """
    Project morphology classifications to 2D.

    Returns arrays for each morphology type showing the fraction of particles
    of that type in each pixel.
    """
    n_particles = len(sorted_positions)

    # Define slab bounds
    slab_width = box_size * slab_fraction
    slab_min = box_size * slab_center - slab_width / 2
    slab_max = box_size * slab_center + slab_width / 2

    # Select particles in slab
    axis_coords = sorted_positions[:, projection_axis]
    in_slab = (axis_coords >= slab_min) & (axis_coords < slab_max)

    slab_positions = sorted_positions[in_slab]
    slab_morphology = np.array(morphology)[in_slab]

    # Determine which axes to use for 2D projection
    axes_2d = [i for i in range(3) if i != projection_axis]

    # Create grids for each morphology type
    cell_size = box_size / output_cells
    grids = {
        VOID: np.zeros((output_cells, output_cells), dtype=np.float64),
        WALL: np.zeros((output_cells, output_cells), dtype=np.float64),
        FILAMENT: np.zeros((output_cells, output_cells), dtype=np.float64),
        HALO: np.zeros((output_cells, output_cells), dtype=np.float64),
    }
    total_count = np.zeros((output_cells, output_cells), dtype=np.float64)

    # Bin particles
    for i, (pos, morph) in enumerate(zip(slab_positions, slab_morphology)):
        ix = int(pos[axes_2d[0]] / cell_size) % output_cells
        iy = int(pos[axes_2d[1]] / cell_size) % output_cells
        grids[morph][iy, ix] += 1
        total_count[iy, ix] += 1

    # Convert to fractions
    total_count = np.maximum(total_count, 1)  # Avoid division by zero
    for morph in grids:
        grids[morph] /= total_count

    return grids, total_count


def composite_morphology_image(grids, total_count, gamma=0.5):
    """
    Create a composite RGB image from morphology fractions.

    Each pixel's color is a weighted blend of morphology colors,
    with brightness based on total particle count.
    """
    output_cells = grids[VOID].shape[0]
    rgb = np.zeros((output_cells, output_cells, 3), dtype=np.float64)

    # Blend colors based on morphology fractions
    for morph, color in MORPHOLOGY_COLORS.items():
        for c in range(3):
            rgb[:, :, c] += grids[morph] * color[c]

    # Apply brightness based on total count (log scale)
    log_count = np.log10(total_count + 1)
    brightness = log_count / (log_count.max() + 1e-10)
    brightness = np.power(brightness, gamma)  # Gamma correction

    # Apply brightness
    for c in range(3):
        rgb[:, :, c] *= brightness

    return np.clip(rgb, 0, 1)


def generate_origami_projection(
    snapshot_path,
    output_cells=512,
    slab_fraction=0.08,
    projection_axis=2,
):
    """Generate ORIGAMI morphology projection for a single snapshot."""
    # Load snapshot
    positions, particle_ids, box_size, scale_factor = load_snapshot(snapshot_path)

    # Compute ORIGAMI
    result, sorted_positions, grid_size = compute_origami(positions, particle_ids, box_size)

    # Project to 2D
    grids, total_count = project_morphology_to_2d(
        sorted_positions,
        result.morphology,
        box_size,
        output_cells,
        projection_axis=projection_axis,
        slab_fraction=slab_fraction,
    )

    return grids, total_count, scale_factor, box_size


def build_origami_time_series(
    projections,
    a_min=0.02,
    a_max=100.0,
    n_box_replications=4,
    gamma=0.5,
):
    """
    Build a Diemer-style time-series image from ORIGAMI projections.

    Parameters
    ----------
    projections : dict
        Dictionary mapping scale factor to (grids, total_count) tuples
    a_min, a_max : float
        Scale factor range for the image
    n_box_replications : int
        How many times to tile the box in x (= time direction)
    gamma : float
        Gamma correction for brightness

    Returns
    -------
    output : np.ndarray
        RGB time-series image array, shape (height, width, 3)
    """
    a_values = np.array(sorted(projections.keys()))
    first_grids, first_count = projections[a_values[0]]
    L_pix = first_grids[VOID].shape[0]

    width = n_box_replications * L_pix
    height = L_pix
    output = np.zeros((height, width, 3), dtype=np.float64)

    log_a_values = np.log(a_values)

    for x_out in range(width):
        # Map x to scale factor (logarithmic)
        a = a_min * (a_max / a_min) ** (x_out / (width - 1))
        log_a = np.log(a)

        # Position within periodic box
        x_box = x_out % L_pix

        # Find bracketing snapshots
        idx_after = np.searchsorted(log_a_values, log_a)
        idx_after = np.clip(idx_after, 1, len(a_values) - 1)
        idx_before = idx_after - 1

        # Interpolation weight
        denom = log_a_values[idx_after] - log_a_values[idx_before]
        if abs(denom) < 1e-10:
            t = 0.0
        else:
            t = (log_a - log_a_values[idx_before]) / denom
        t = np.clip(t, 0.0, 1.0)

        # Get bracketing projections
        a_before = a_values[idx_before]
        a_after = a_values[idx_after]
        grids_before, count_before = projections[a_before]
        grids_after, count_after = projections[a_after]

        # Interpolate morphology fractions
        interpolated_grids = {}
        for morph in [VOID, WALL, FILAMENT, HALO]:
            col_before = grids_before[morph][:, x_box]
            col_after = grids_after[morph][:, x_box]
            interpolated_grids[morph] = (1 - t) * col_before + t * col_after

        # Interpolate counts (in log space)
        eps = 1e-10
        log_count_before = np.log(count_before[:, x_box] + eps)
        log_count_after = np.log(count_after[:, x_box] + eps)
        interpolated_count = np.exp((1 - t) * log_count_before + t * log_count_after) - eps

        # Compute RGB for this column
        rgb_col = np.zeros((height, 3), dtype=np.float64)
        for morph, color in MORPHOLOGY_COLORS.items():
            for c in range(3):
                rgb_col[:, c] += interpolated_grids[morph] * color[c]

        # Apply brightness
        log_count = np.log10(interpolated_count + 1)
        brightness = log_count / (log_count.max() + 1e-10)
        brightness = np.power(np.maximum(brightness, 0), gamma)

        for c in range(3):
            rgb_col[:, c] *= brightness

        output[:, x_out, :] = np.clip(rgb_col, 0, 1)

    return output


def render_origami_time_series(
    output_image,
    output_path,
    a_min,
    a_max,
    n_box_replications,
    show_axes=True,
    figsize=None,
    dpi=300,
):
    """Render the ORIGAMI time-series image with axes and legend."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if figsize is None:
        aspect = output_image.shape[1] / output_image.shape[0]
        figsize = (14, 14 / aspect)

    if show_axes:
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
    else:
        fig = plt.figure(figsize=figsize, frameon=False, facecolor='black')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    ax.imshow(output_image, origin='lower', aspect='auto')

    if show_axes:
        L_pix = output_image.shape[0]
        width = output_image.shape[1]

        # Scale factor ticks
        key_a_values = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        key_a_values = [a for a in key_a_values if a_min <= a <= a_max]

        tick_positions = []
        tick_labels = []
        for a in key_a_values:
            x = (width - 1) * np.log(a / a_min) / np.log(a_max / a_min)
            tick_positions.append(x)
            if a >= 1:
                tick_labels.append(f'{a:.0f}' if a == int(a) else f'{a:.1f}')
            else:
                tick_labels.append(f'{a:.2g}')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='white', fontsize=10)
        ax.set_xlabel('Scale factor $a$', fontsize=12, color='white')

        ax.set_yticks([])
        ax.set_ylabel('$y$ [comoving]', fontsize=12, color='white')

        # Title
        ax.set_title(
            f'Cosmic Web Evolution : $a = {a_min}$ → $a = {a_max}$',
            fontsize=14, color='white', pad=10
        )

        # Legend
        legend_elements = [
            Patch(facecolor=MORPHOLOGY_COLORS[HALO], edgecolor='white', label='Halos'),
            Patch(facecolor=MORPHOLOGY_COLORS[FILAMENT], edgecolor='white', label='Filaments'),
            Patch(facecolor=MORPHOLOGY_COLORS[WALL], edgecolor='white', label='Walls'),
            Patch(facecolor=MORPHOLOGY_COLORS[VOID], edgecolor='white', label='Voids'),
        ]
        ax.legend(
            handles=legend_elements, loc='upper right',
            facecolor='black', edgecolor='white', labelcolor='white',
            fontsize=10
        )

        ax.tick_params(axis='both', colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    plt.close()

    print(f"Saved ORIGAMI time-series to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ORIGAMI cosmic web time-series visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--snapshot-dir", type=Path, required=True,
                        help="Directory containing snapshot files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--box-size", type=float, default=256.0,
                        help="Box size in Mpc/h")
    parser.add_argument("--output-cells", type=int, default=512,
                        help="Output grid resolution")
    parser.add_argument("--slab-fraction", type=float, default=0.08,
                        help="Fraction of box depth for projection slab")
    parser.add_argument("--n-replications", type=int, default=4,
                        help="Number of box replications in time direction")
    parser.add_argument("--a-min", type=float, default=0.02,
                        help="Minimum scale factor")
    parser.add_argument("--a-max", type=float, default=100.0,
                        help="Maximum scale factor")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Gamma correction for brightness")
    parser.add_argument("--snapshots", type=int, nargs="+",
                        help="Specific snapshot indices to process")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI")
    parser.add_argument("--no-axes", action="store_true",
                        help="Hide axes and legend")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORIGAMI COSMIC WEB TIME-SERIES VISUALIZATION")
    print("=" * 70)
    print()

    # Find snapshots
    if args.snapshots:
        snapshot_files = [
            (idx, args.snapshot_dir / f"snapshot_{idx:03d}.hdf5")
            for idx in args.snapshots
        ]
    else:
        snapshot_files = sorted([
            (int(p.stem.split("_")[-1]), p)
            for p in args.snapshot_dir.glob("snapshot_*.hdf5")
        ])

    print(f"Found {len(snapshot_files)} snapshots")
    print()

    # Generate projections
    print("-" * 70)
    print("STEP 1: Computing ORIGAMI morphology projections")
    print("-" * 70)

    projections = {}
    total_start = time.time()

    for i, (idx, path) in enumerate(snapshot_files):
        print(f"  [{i+1}/{len(snapshot_files)}] Snapshot {idx}...", end=" ", flush=True)
        step_start = time.time()

        grids, total_count, scale_factor, box_size = generate_origami_projection(
            path,
            output_cells=args.output_cells,
            slab_fraction=args.slab_fraction,
        )

        projections[scale_factor] = (grids, total_count)

        print(f"a={scale_factor:.4f}, {time.time() - step_start:.1f}s")

    print(f"\nProjections complete in {time.time() - total_start:.1f}s")
    print()

    # Build time-series image
    print("-" * 70)
    print("STEP 2: Building time-series image")
    print("-" * 70)

    step_start = time.time()
    output_image = build_origami_time_series(
        projections,
        a_min=args.a_min,
        a_max=args.a_max,
        n_box_replications=args.n_replications,
        gamma=args.gamma,
    )
    print(f"Image built in {time.time() - step_start:.1f}s")
    print()

    # Render
    print("-" * 70)
    print("STEP 3: Rendering output")
    print("-" * 70)

    output_path = args.output_dir / "time_series_origami.png"
    render_origami_time_series(
        output_image,
        output_path,
        a_min=args.a_min,
        a_max=args.a_max,
        n_box_replications=args.n_replications,
        show_axes=not args.no_axes,
        dpi=args.dpi,
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
