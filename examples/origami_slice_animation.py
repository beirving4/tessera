#!/usr/bin/env python3
"""
ORIGAMI Morphology Slice Animation

Creates an animation showing the evolution of cosmic web morphology from
early times through the nonlinear regime.

At early times (linear regime), density structure is shown since ORIGAMI
classifies all particles as voids (no shell-crossing). As structure forms,
the visualization transitions to showing ORIGAMI morphology coloring.

Usage:
    python origami_slice_animation.py --snapshot-dir /path/to/snapshots --output animation.mp4
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


# Morphology class constants
VOID = 0
WALL = 1
FILAMENT = 2
HALO = 3

MORPHOLOGY_NAMES = ['Void', 'Wall', 'Filament', 'Halo']

# ORIGAMI morphology colors (RGB, 0-1)
MORPHOLOGY_COLORS = {
    VOID: np.array([0.267, 0.004, 0.329]),    # Dark purple (#440154)
    WALL: np.array([0.193, 0.408, 0.557]),    # Blue (#31688e)
    FILAMENT: np.array([0.208, 0.718, 0.475]), # Green (#35b779)
    HALO: np.array([0.992, 0.906, 0.145]),    # Yellow (#fde725)
}

# Density colormap for linear regime (blue gradient)
DENSITY_COLORS = [
    (0.0, np.array([0.267, 0.004, 0.329])),   # Dark purple (same as void)
    (0.2, np.array([0.282, 0.141, 0.458])),   # Purple
    (0.4, np.array([0.253, 0.265, 0.530])),   # Blue-purple
    (0.6, np.array([0.193, 0.408, 0.557])),   # Blue
    (0.8, np.array([0.143, 0.522, 0.556])),   # Teal
    (1.0, np.array([0.208, 0.718, 0.475])),   # Green (bright regions)
]


def density_colormap(value):
    """Apply density colormap to normalized values."""
    value = np.atleast_1d(value).astype(np.float64)
    result = np.zeros((*value.shape, 3), dtype=np.float64)

    for i in range(len(DENSITY_COLORS) - 1):
        t0, c0 = DENSITY_COLORS[i]
        t1, c1 = DENSITY_COLORS[i + 1]

        mask = (value >= t0) & (value < t1)
        if np.any(mask):
            t = (value[mask] - t0) / (t1 - t0)
            for c in range(3):
                result[mask, c] = c0[c] + t * (c1[c] - c0[c])

    # Handle value >= 1.0
    mask = value >= 1.0
    result[mask] = DENSITY_COLORS[-1][1]

    return result


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot."""
    if not HAS_H5PY:
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


def compute_density_grid(sorted_positions, box_size, grid_size, output_cells):
    """Compute density field on a grid."""
    config = ts.density.TetraDensityConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.output_cells = output_cells
    config.n_samples = 50
    config.n_threads = 1
    config.periodic = True

    result = ts.density.compute_tetra_density_3d(sorted_positions, config)
    density_3d = np.array(result.density).reshape((output_cells, output_cells, output_cells))

    return density_3d


def process_snapshot(snapshot_path, grid_cells=128):
    """Process a single snapshot to get morphology grid and density."""
    # Load
    positions, particle_ids, box_size, scale_factor = load_snapshot(snapshot_path)
    sorted_positions, grid_size = sort_particles_for_origami(positions, particle_ids, box_size)

    # Compute ORIGAMI
    config = ts.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    origami_result = ts.origami.compute_morphology(sorted_positions, config)

    # Deposit morphology to grid
    ts.origami.deposit_morphology_to_grid(sorted_positions, box_size, grid_cells, origami_result)

    # Get morphology fractions per cell
    morphology_grid = np.array(origami_result.morphology_grid)
    void_frac = np.array(origami_result.void_fraction_grid)
    wall_frac = np.array(origami_result.wall_fraction_grid)
    filament_frac = np.array(origami_result.filament_fraction_grid)
    halo_frac = np.array(origami_result.halo_fraction_grid)

    # Compute density for linear regime visualization
    density_3d = compute_density_grid(sorted_positions, box_size, grid_size, grid_cells)

    return {
        'scale_factor': scale_factor,
        'box_size': box_size,
        'morphology_grid': morphology_grid,
        'void_fraction': void_frac,
        'wall_fraction': wall_frac,
        'filament_fraction': filament_frac,
        'halo_fraction': halo_frac,
        'density': density_3d,
        'is_linear_regime': origami_result.is_linear_regime,
        'f_void': origami_result.f_void,
        'f_wall': origami_result.f_wall,
        'f_filament': origami_result.f_filament,
        'f_halo': origami_result.f_halo,
    }


def render_blended_slice(data, slice_axis=0, slice_index=None, gamma=0.7):
    """
    Render a slice blending density and morphology coloring.

    In the linear regime (high void fraction), shows density structure.
    As non-void structures form, blends in ORIGAMI morphology colors.
    """
    grid_cells = data['morphology_grid'].shape[0]
    if slice_index is None:
        slice_index = grid_cells // 2

    # Extract slices
    if slice_axis == 0:
        void_frac = data['void_fraction'][slice_index, :, :]
        wall_frac = data['wall_fraction'][slice_index, :, :]
        filament_frac = data['filament_fraction'][slice_index, :, :]
        halo_frac = data['halo_fraction'][slice_index, :, :]
        density = data['density'][slice_index, :, :]
    elif slice_axis == 1:
        void_frac = data['void_fraction'][:, slice_index, :]
        wall_frac = data['wall_fraction'][:, slice_index, :]
        filament_frac = data['filament_fraction'][:, slice_index, :]
        halo_frac = data['halo_fraction'][:, slice_index, :]
        density = data['density'][:, slice_index, :]
    else:
        void_frac = data['void_fraction'][:, :, slice_index]
        wall_frac = data['wall_fraction'][:, :, slice_index]
        filament_frac = data['filament_fraction'][:, :, slice_index]
        halo_frac = data['halo_fraction'][:, :, slice_index]
        density = data['density'][:, :, slice_index]

    # Normalize density (log scale)
    log_density = np.log10(density + 1e-10)
    log_min, log_max = log_density.min(), log_density.max()
    if log_max > log_min:
        density_norm = (log_density - log_min) / (log_max - log_min)
    else:
        density_norm = np.zeros_like(log_density)

    # Apply gamma correction for better contrast
    density_norm = np.power(density_norm, gamma)

    # Compute density-based colors
    density_colors = density_colormap(density_norm.flatten()).reshape(
        density_norm.shape[0], density_norm.shape[1], 3
    )

    # Compute ORIGAMI morphology colors for non-void regions
    f_nonvoid = wall_frac + filament_frac + halo_frac
    f_nonvoid_safe = np.maximum(f_nonvoid, 1e-10)

    origami_colors = np.zeros((density_norm.shape[0], density_norm.shape[1], 3))
    for morph, frac in [(WALL, wall_frac), (FILAMENT, filament_frac), (HALO, halo_frac)]:
        color = MORPHOLOGY_COLORS[morph]
        weight = frac / f_nonvoid_safe
        for c in range(3):
            origami_colors[:, :, c] += weight * color[c]

    # Blend: voids show density, non-voids show ORIGAMI colors
    rgb = np.zeros((density_norm.shape[0], density_norm.shape[1], 3))
    for c in range(3):
        rgb[:, :, c] = (
            void_frac * density_colors[:, :, c] +
            f_nonvoid * origami_colors[:, :, c]
        )

    return np.clip(rgb, 0, 1)


def create_animation(
    snapshot_dir,
    output_path,
    grid_cells=128,
    slice_axis=0,
    slice_index=None,
    fps=5,
    figsize=(8, 7),
    dpi=150,
    snapshots=None,
):
    """Create animation of ORIGAMI morphology evolution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    snapshot_dir = Path(snapshot_dir)

    # Find snapshots
    if snapshots:
        snapshot_files = [
            (idx, snapshot_dir / f"snapshot_{idx:03d}.hdf5")
            for idx in snapshots
        ]
    else:
        snapshot_files = sorted([
            (int(p.stem.split("_")[-1]), p)
            for p in snapshot_dir.glob("snapshot_*.hdf5")
        ])

    print(f"Found {len(snapshot_files)} snapshots")

    # Process all snapshots
    print("\nProcessing snapshots...")
    all_data = []
    for i, (idx, path) in enumerate(snapshot_files):
        print(f"  [{i+1}/{len(snapshot_files)}] Snapshot {idx}...", end=" ", flush=True)
        start = time.time()
        data = process_snapshot(path, grid_cells)
        status = "[LINEAR]" if data['is_linear_regime'] else ""
        print(f"a={data['scale_factor']:.4f} {status} ({time.time()-start:.1f}s)")
        all_data.append(data)

    box_size = all_data[0]['box_size']

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initial frame
    rgb = render_blended_slice(all_data[0], slice_axis, slice_index)
    im = ax.imshow(rgb, origin='lower', extent=[0, box_size, 0, box_size],
                   interpolation='nearest')

    axis_names = ['z', 'y', 'x']
    slice_name = axis_names[slice_axis]
    if slice_axis == 0:
        xlabel, ylabel = 'x [Mpc/h]', 'y [Mpc/h]'
    elif slice_axis == 1:
        xlabel, ylabel = 'x [Mpc/h]', 'z [Mpc/h]'
    else:
        xlabel, ylabel = 'y [Mpc/h]', 'z [Mpc/h]'

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Title (will be updated)
    title = ax.set_title('', fontsize=12, fontweight='medium')

    # Legend
    legend_elements = [
        Patch(facecolor=MORPHOLOGY_COLORS[HALO], edgecolor='white', label='Halo'),
        Patch(facecolor=MORPHOLOGY_COLORS[FILAMENT], edgecolor='white', label='Filament'),
        Patch(facecolor=MORPHOLOGY_COLORS[WALL], edgecolor='white', label='Wall'),
        Patch(facecolor=DENSITY_COLORS[2][1], edgecolor='white', label='Void (density)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout()

    def update(frame):
        data = all_data[frame]
        rgb = render_blended_slice(data, slice_axis, slice_index)
        im.set_array(rgb)

        a = data['scale_factor']
        z = 1.0 / a - 1.0

        if data['is_linear_regime']:
            regime_str = " [LINEAR REGIME]"
        else:
            regime_str = ""

        if z < 0.01:
            title.set_text(f'Cosmic Web Evolution ({slice_name}-slice): a = {a:.2f}{regime_str}')
        elif z < 10:
            title.set_text(f'Cosmic Web Evolution ({slice_name}-slice): z = {z:.2f} (a = {a:.3f}){regime_str}')
        else:
            title.set_text(f'Cosmic Web Evolution ({slice_name}-slice): a = {a:.2f}{regime_str}')

        return [im, title]

    print(f"\nCreating animation ({len(all_data)} frames at {fps} fps)...")
    anim = FuncAnimation(fig, update, frames=len(all_data), blit=False, interval=1000/fps)

    # Save
    output_path = Path(output_path)
    if output_path.suffix.lower() in ['.mp4', '.mov']:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'ORIGAMI Morphology Evolution'})
        anim.save(output_path, writer=writer, dpi=dpi)
    elif output_path.suffix.lower() == '.gif':
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    else:
        # Default to mp4
        output_path = output_path.with_suffix('.mp4')
        writer = FFMpegWriter(fps=fps, metadata={'title': 'ORIGAMI Morphology Evolution'})
        anim.save(output_path, writer=writer, dpi=dpi)

    plt.close()
    print(f"Saved animation to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create ORIGAMI morphology slice animation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--snapshot-dir", type=Path, required=True,
                        help="Directory containing snapshot files")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output animation path (.mp4, .gif)")

    # Snapshot selection
    parser.add_argument("--snapshots", type=int, nargs="+",
                        help="Specific snapshot indices to process")

    # Slice options
    parser.add_argument("--slice-axis", type=int, default=0, choices=[0, 1, 2],
                        help="Axis perpendicular to slice (0=z, 1=y, 2=x)")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="Index along slice axis (default: middle)")

    # Grid options
    parser.add_argument("--grid-cells", type=int, default=128,
                        help="Grid cells for morphology/density deposition")

    # Animation options
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 7],
                        help="Figure size (width height) in inches")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI")

    args = parser.parse_args()

    print("=" * 70)
    print("ORIGAMI MORPHOLOGY SLICE ANIMATION")
    print("=" * 70)

    create_animation(
        args.snapshot_dir,
        args.output,
        grid_cells=args.grid_cells,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        fps=args.fps,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        snapshots=args.snapshots,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
