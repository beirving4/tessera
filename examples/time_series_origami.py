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

# Import shared utilities
from tessera.utils import load_snapshot_h5py, infer_grid_size, check_h5py_available

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
    HAS_H5PY = check_h5py_available()


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

# Density colormap colors (similar to original time series - blue gradient)
DENSITY_COLORS = [
    (0.0, np.array([0.0, 0.02, 0.04])),    # Nearly black
    (0.2, np.array([0.0, 0.1, 0.2])),      # Dark blue
    (0.4, np.array([0.0, 0.25, 0.4])),     # Medium blue
    (0.6, np.array([0.0, 0.4, 0.55])),     # Blue
    (0.8, np.array([0.0, 0.65, 0.8])),     # Cyan
    (1.0, np.array([0.8, 0.95, 1.0])),     # Light cyan/white
]

# Linear regime threshold - if void fraction > this, field is considered linear
LINEAR_REGIME_VOID_THRESHOLD = 0.99


def is_linear_regime(origami_result, void_threshold=LINEAR_REGIME_VOID_THRESHOLD):
    """
    Detect if the density field is still in the linear regime.

    In the linear regime, no shell-crossing has occurred, so ORIGAMI
    classifies all particles as voids (0 phase-space crossings).

    Parameters
    ----------
    origami_result : OrigamiResult
        Result from ts.origami.compute_morphology()
    void_threshold : float
        Void fraction above which the field is considered linear (default: 0.99)

    Returns
    -------
    bool
        True if field is in linear regime, False otherwise
    """
    return origami_result.f_void >= void_threshold


def get_linear_regime_info(origami_result):
    """
    Get detailed information about the linear regime status.

    Returns
    -------
    dict
        Dictionary with keys:
        - is_linear: bool, True if in linear regime
        - void_fraction: float, fraction of particles classified as void
        - nonlinear_fraction: float, fraction of particles with shell-crossing
        - recommendation: str, recommendation for analysis
    """
    f_nonlinear = 1.0 - origami_result.f_void
    is_linear = origami_result.f_void >= LINEAR_REGIME_VOID_THRESHOLD

    if is_linear:
        recommendation = (
            "Field is in linear regime (no shell-crossing). "
            "ORIGAMI classification not meaningful - use density-based analysis instead."
        )
    elif origami_result.f_void > 0.9:
        recommendation = (
            "Field is in quasi-linear regime (limited shell-crossing). "
            "ORIGAMI classification may be incomplete."
        )
    else:
        recommendation = (
            "Field is in nonlinear regime. "
            "ORIGAMI classification is meaningful."
        )

    return {
        'is_linear': is_linear,
        'void_fraction': origami_result.f_void,
        'nonlinear_fraction': f_nonlinear,
        'recommendation': recommendation,
    }


def density_colormap(value):
    """
    Apply density colormap to normalized values.

    Parameters
    ----------
    value : float or np.ndarray
        Normalized value(s) in [0, 1]

    Returns
    -------
    np.ndarray
        RGB color(s)
    """
    value = np.atleast_1d(value)
    result = np.zeros((*value.shape, 3))

    for i in range(len(DENSITY_COLORS) - 1):
        t0, c0 = DENSITY_COLORS[i]
        t1, c1 = DENSITY_COLORS[i + 1]

        mask = (value >= t0) & (value < t1)
        if np.any(mask):
            t = (value[mask] - t0) / (t1 - t0)
            for c in range(3):
                result[mask, c] = c0[c] + t * (c1[c] - c0[c])

    # Handle value == 1.0
    mask = value >= 1.0
    result[mask] = DENSITY_COLORS[-1][1]

    return result.squeeze() if result.shape[0] == 1 else result


def load_snapshot(snapshot_path, particle_type=1):
    """Load particle data from a GADGET-4 HDF5 snapshot."""
    # Use utils function for loading (handles both h5py and tessera fallback)
    positions, particle_ids, box_size, scale_factor = load_snapshot_h5py(
        snapshot_path, particle_type=particle_type, read_ids=True
    )
    return positions, particle_ids, box_size, scale_factor


def compute_origami(positions, particle_ids, box_size):
    """Compute ORIGAMI morphology classification using the unified pipeline."""
    # Infer grid size using utils function
    grid_size = infer_grid_size(len(positions))

    # Use unified pipeline - handles ID ordering detection and sorting internally
    config = ts.origami.PipelineConfig(grid_size, float(box_size))
    config.n_threads = 1
    config.n_split = 1

    # Run pipeline (positions are sorted in-place)
    result = ts.origami.run_pipeline(positions, particle_ids, config)

    # Return compatible interface - positions are now sorted
    return result, positions, grid_size


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
    of that type in each pixel, plus a normalized density field.
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

    # Save raw counts before normalization for density field
    raw_count = total_count.copy()

    # Convert to fractions
    total_count = np.maximum(total_count, 1)  # Avoid division by zero
    for morph in grids:
        grids[morph] /= total_count

    # Compute normalized density field (log-scaled, 0-1)
    log_density = np.log10(raw_count + 1)
    density_normalized = log_density / (log_density.max() + 1e-10)

    return grids, total_count, density_normalized


def composite_morphology_image(grids, total_count, density_normalized, gamma=0.5):
    """
    Create a composite RGB image blending density and morphology.

    In the linear regime (mostly voids), density structure is shown.
    As non-void structures form, ORIGAMI morphology coloring is blended in.

    Parameters
    ----------
    grids : dict
        Morphology fraction grids {VOID: array, WALL: array, ...}
    total_count : np.ndarray
        Particle count per pixel
    density_normalized : np.ndarray
        Normalized density field (0-1)
    gamma : float
        Gamma correction for brightness
    """
    output_cells = grids[VOID].shape[0]
    rgb = np.zeros((output_cells, output_cells, 3), dtype=np.float64)

    # Compute density-based colors (for void regions / linear regime)
    density_colors = density_colormap(density_normalized.flatten()).reshape(
        output_cells, output_cells, 3
    )

    # Compute ORIGAMI morphology colors for non-void particles
    # Normalize non-void fractions to sum to 1 where non-void particles exist
    f_nonvoid = grids[WALL] + grids[FILAMENT] + grids[HALO]
    f_nonvoid_safe = np.maximum(f_nonvoid, 1e-10)

    origami_colors = np.zeros((output_cells, output_cells, 3), dtype=np.float64)
    for morph in [WALL, FILAMENT, HALO]:
        color = MORPHOLOGY_COLORS[morph]
        weight = grids[morph] / f_nonvoid_safe
        for c in range(3):
            origami_colors[:, :, c] += weight * color[c]

    # Blend: voids show density, non-voids show ORIGAMI colors
    # For each pixel: final = f_void * density_color + f_nonvoid * origami_color
    for c in range(3):
        rgb[:, :, c] = (
            grids[VOID] * density_colors[:, :, c] +
            f_nonvoid * origami_colors[:, :, c]
        )

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

    # Check linear regime status
    linear_info = get_linear_regime_info(result)
    if linear_info['is_linear']:
        print(f"[LINEAR REGIME: {linear_info['void_fraction']*100:.1f}% void]", end=" ")

    # Project to 2D
    grids, total_count, density_normalized = project_morphology_to_2d(
        sorted_positions,
        result.morphology,
        box_size,
        output_cells,
        projection_axis=projection_axis,
        slab_fraction=slab_fraction,
    )

    return grids, total_count, density_normalized, scale_factor, box_size


def build_origami_time_series(
    projections,
    a_min=0.02,
    a_max=100.0,
    n_box_replications=4,
    gamma=0.5,
):
    """
    Build a Diemer-style time-series image from ORIGAMI projections.

    Blends density-based coloring (for voids/linear regime) with ORIGAMI
    morphology coloring (for non-void structures).

    Parameters
    ----------
    projections : dict
        Dictionary mapping scale factor to (grids, total_count, density_normalized) tuples
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
    first_grids, first_count, first_density = projections[a_values[0]]
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
        grids_before, count_before, density_before = projections[a_before]
        grids_after, count_after, density_after = projections[a_after]

        # Interpolate morphology fractions
        interpolated_grids = {}
        for morph in [VOID, WALL, FILAMENT, HALO]:
            col_before = grids_before[morph][:, x_box]
            col_after = grids_after[morph][:, x_box]
            interpolated_grids[morph] = (1 - t) * col_before + t * col_after

        # Interpolate density
        density_col = (1 - t) * density_before[:, x_box] + t * density_after[:, x_box]

        # Interpolate counts (in log space)
        eps = 1e-10
        log_count_before = np.log(count_before[:, x_box] + eps)
        log_count_after = np.log(count_after[:, x_box] + eps)
        interpolated_count = np.exp((1 - t) * log_count_before + t * log_count_after) - eps

        # Compute density-based colors (for void regions / linear regime)
        density_colors = density_colormap(density_col)

        # Compute non-void fraction and ORIGAMI colors
        f_void = interpolated_grids[VOID]
        f_nonvoid = interpolated_grids[WALL] + interpolated_grids[FILAMENT] + interpolated_grids[HALO]
        f_nonvoid_safe = np.maximum(f_nonvoid, 1e-10)

        # ORIGAMI morphology colors for non-void particles
        origami_colors = np.zeros((height, 3), dtype=np.float64)
        for morph in [WALL, FILAMENT, HALO]:
            color = MORPHOLOGY_COLORS[morph]
            weight = interpolated_grids[morph] / f_nonvoid_safe
            for c in range(3):
                origami_colors[:, c] += weight * color[c]

        # Blend: voids show density, non-voids show ORIGAMI colors
        rgb_col = np.zeros((height, 3), dtype=np.float64)
        for c in range(3):
            rgb_col[:, c] = (
                f_void * density_colors[:, c] +
                f_nonvoid * origami_colors[:, c]
            )

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

        # Legend - note: voids show density structure, not solid color
        legend_elements = [
            Patch(facecolor=MORPHOLOGY_COLORS[HALO], edgecolor='white', label='Halos'),
            Patch(facecolor=MORPHOLOGY_COLORS[FILAMENT], edgecolor='white', label='Filaments'),
            Patch(facecolor=MORPHOLOGY_COLORS[WALL], edgecolor='white', label='Walls'),
            Patch(facecolor=DENSITY_COLORS[3][1], edgecolor='white', label='Voids (density)'),
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

        grids, total_count, density_normalized, scale_factor, box_size = generate_origami_projection(
            path,
            output_cells=args.output_cells,
            slab_fraction=args.slab_fraction,
        )

        projections[scale_factor] = (grids, total_count, density_normalized)

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
