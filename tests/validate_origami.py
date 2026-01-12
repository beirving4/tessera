#!/usr/bin/env python3
"""
Validation script to compare AsymptoticTetra ORIGAMI implementation
against the original ORIGAMI code from Falck et al.

This script:
1. Generates synthetic test data with known structure
2. Writes it in binary format for original code
3. Runs both implementations
4. Compares the morphology classifications
"""

import numpy as np
import struct
import subprocess
import tempfile
import os
import sys
from pathlib import Path

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

def write_pos_file(positions, filepath):
    """Write positions to binary format expected by original ORIGAMI code.

    Format:
    - 4-byte int: number of particles
    - N floats: x coordinates
    - N floats: y coordinates
    - N floats: z coordinates
    """
    n = len(positions)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('i', n))
        f.write(positions[:, 0].astype(np.float32).tobytes())
        f.write(positions[:, 1].astype(np.float32).tobytes())
        f.write(positions[:, 2].astype(np.float32).tobytes())

def read_tag_file(filepath):
    """Read morphology tags from original ORIGAMI output."""
    with open(filepath, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        tags = np.frombuffer(f.read(n), dtype=np.uint8)
    return tags

def write_parameter_file(params_path, pos_file, out_dir, tag_label, box_size, np1d, nsplit=1):
    """Write parameter file for original ORIGAMI code."""
    with open(params_path, 'w') as f:
        # First line is a comment (skipped by read_parameters)
        f.write(f"# ORIGAMI validation test\n")
        f.write(f"posfile\t\t{pos_file}\n")
        f.write(f"outdir\t\t{out_dir}\n")
        f.write(f"taglabel\t{tag_label}\n")
        f.write(f"boxsize\t\t{box_size}\n")
        f.write(f"np1d\t\t{np1d}\n")
        f.write(f"nsplit\t\t{nsplit}\n")
        f.write(f"numfiles\t0\n")
        # Blank line before buffer section
        f.write(f"\n")
        f.write(f"buffer\t\t0.1\n")
        f.write(f"ndiv\t\t2\n")
        # Blank line before halo section
        f.write(f"\n")
        f.write(f"volcut\t\t.01\n")
        f.write(f"npmin\t\t20\n")
        f.write(f"halolabel\ttest\n")

def run_original_origami(origami_path, params_path):
    """Run original ORIGAMI code."""
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    result = subprocess.run(
        [origami_path, params_path],
        capture_output=True,
        text=True,
        env=env
    )
    # Note: original code returns 1 on success (return(1) in main)
    return result

def run_asymptotic_origami(positions, box_size, grid_size):
    """Run AsymptoticTetra ORIGAMI implementation."""
    import _asymptotic_tetra as at

    config = at.origami.OrigamiConfig()
    config.lagrangian_grid_size = grid_size
    config.box_size = float(box_size)
    config.n_threads = 1
    config.n_split = 1

    result = at.origami.compute_morphology(positions.astype(np.float64), config)
    return np.array(result.morphology)

def generate_test_data(grid_size, box_size, displacement_type='uniform'):
    """Generate test positions with different displacement patterns."""
    n_particles = grid_size ** 3
    spacing = box_size / grid_size

    # Create Lagrangian grid positions
    positions = np.zeros((n_particles, 3), dtype=np.float64)
    for idx in range(n_particles):
        x = idx % grid_size
        y = (idx // grid_size) % grid_size
        z = idx // (grid_size * grid_size)
        positions[idx] = [(x + 0.5) * spacing, (y + 0.5) * spacing, (z + 0.5) * spacing]

    if displacement_type == 'uniform':
        # No displacement - uniform grid
        pass
    elif displacement_type == 'small':
        # Small random displacement (< spacing/4)
        np.random.seed(42)
        positions += np.random.uniform(-spacing/4, spacing/4, positions.shape)
    elif displacement_type == 'medium':
        # Medium random displacement (~ spacing/2)
        np.random.seed(42)
        positions += np.random.uniform(-spacing/2, spacing/2, positions.shape)
    elif displacement_type == 'large':
        # Large random displacement (~ spacing)
        np.random.seed(42)
        positions += np.random.uniform(-spacing, spacing, positions.shape)
    elif displacement_type == 'very_large':
        # Very large random displacement (~ 2*spacing)
        np.random.seed(42)
        positions += np.random.uniform(-2*spacing, 2*spacing, positions.shape)
    elif displacement_type == 'collapse':
        # Simulate collapse toward a few centers
        np.random.seed(42)
        centers = np.array([
            [box_size/4, box_size/4, box_size/4],
            [3*box_size/4, 3*box_size/4, 3*box_size/4],
            [box_size/4, 3*box_size/4, box_size/4],
            [3*box_size/4, box_size/4, 3*box_size/4],
        ])
        for i in range(n_particles):
            # Find nearest center
            min_dist = np.inf
            nearest = None
            for c in centers:
                diff = positions[i] - c
                diff = np.where(diff > box_size/2, diff - box_size, diff)
                diff = np.where(diff < -box_size/2, diff + box_size, diff)
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    min_dist = dist
                    nearest = c
            # Move toward center
            diff = nearest - positions[i]
            diff = np.where(diff > box_size/2, diff - box_size, diff)
            diff = np.where(diff < -box_size/2, diff + box_size, diff)
            positions[i] += 0.4 * diff

    # Apply periodic boundary conditions
    positions = positions % box_size

    return positions

def compare_results(original_tags, asymptotic_tags):
    """Compare morphology classifications from both implementations."""
    n = len(original_tags)

    if len(asymptotic_tags) != n:
        print(f"  ERROR: Length mismatch! Original: {n}, AsymptoticTetra: {len(asymptotic_tags)}")
        return False, {}

    # Count matches
    matches = np.sum(original_tags == asymptotic_tags)
    match_rate = matches / n

    # Per-class statistics
    stats = {}
    for cls in range(4):
        orig_count = np.sum(original_tags == cls)
        asym_count = np.sum(asymptotic_tags == cls)
        both = np.sum((original_tags == cls) & (asymptotic_tags == cls))
        stats[cls] = {
            'original': orig_count,
            'asymptotic': asym_count,
            'agreement': both,
            'orig_frac': orig_count / n,
            'asym_frac': asym_count / n,
        }

    return match_rate == 1.0, {
        'match_rate': match_rate,
        'total': n,
        'matches': matches,
        'per_class': stats
    }

def main():
    # Paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    origami_path = Path("/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/origami/code/origamitag")

    if not origami_path.exists():
        print(f"ERROR: Original ORIGAMI binary not found at {origami_path}")
        print("Please build it first with: cd code && make origamitag")
        return 1

    # Test configurations
    test_cases = [
        ('uniform', 'Uniform grid (no displacement)'),
        ('small', 'Small displacement'),
        ('medium', 'Medium displacement'),
        ('large', 'Large displacement'),
        ('very_large', 'Very large displacement'),
        ('collapse', 'Collapse toward centers'),
    ]

    grid_size = 32
    box_size = 32.0

    print("=" * 70)
    print("ORIGAMI Implementation Validation")
    print("=" * 70)
    print(f"Grid size: {grid_size}^3 = {grid_size**3:,} particles")
    print(f"Box size: {box_size}")
    print()

    all_passed = True

    for disp_type, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 50)

        # Use a fresh temp directory for each test
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Generate test data
            positions = generate_test_data(grid_size, box_size, disp_type)

            # Write files for original code
            pos_file = tmpdir / f"test.pos"
            params_file = tmpdir / f"params.txt"
            out_dir = str(tmpdir) + "/"
            tag_label = "test"

            write_pos_file(positions, pos_file)
            write_parameter_file(params_file, str(pos_file), out_dir, tag_label, box_size, grid_size)

            # Run original ORIGAMI
            result = run_original_origami(str(origami_path), str(params_file))
            # Original code returns 1 on success (return(1) in main())
            if result.returncode not in [0, 1]:
                print(f"  ERROR running original ORIGAMI (code {result.returncode}): {result.stderr}")
                all_passed = False
                continue

            # Read original results
            tag_file = tmpdir / f"{tag_label}tag.dat"
            if not tag_file.exists():
                print(f"  ERROR: Output tag file not found: {tag_file}")
                all_passed = False
                continue

            original_tags = read_tag_file(tag_file)

            # Run AsymptoticTetra ORIGAMI
            try:
                asymptotic_tags = run_asymptotic_origami(positions, box_size, grid_size)
            except Exception as e:
                print(f"  ERROR running AsymptoticTetra: {e}")
                all_passed = False
                continue

            # Compare results
            perfect_match, stats = compare_results(original_tags, asymptotic_tags)

            # Print results
            class_names = ['Void', 'Wall', 'Filament', 'Halo']
            print(f"  Match rate: {stats['match_rate']:.4%} ({stats['matches']:,}/{stats['total']:,})")
            print()
            print(f"  {'Class':<10} {'Original':>12} {'AsymptoticTetra':>16} {'Agreement':>12}")
            print(f"  {'-'*10} {'-'*12} {'-'*16} {'-'*12}")
            for cls in range(4):
                s = stats['per_class'][cls]
                print(f"  {class_names[cls]:<10} {s['original']:>12,} {s['asymptotic']:>16,} {s['agreement']:>12,}")

            if perfect_match:
                print(f"\n  PASSED: Perfect match!")
            else:
                print(f"\n  FAILED: Mismatch detected")
                all_passed = False

                # Show disagreement details
                disagreements = np.where(original_tags != asymptotic_tags)[0]
                if len(disagreements) > 0:
                    print(f"\n  Sample disagreements (first 10):")
                    for idx in disagreements[:10]:
                        x = idx % grid_size
                        y = (idx // grid_size) % grid_size
                        z = idx // (grid_size * grid_size)
                        print(f"    Particle {idx} (grid {x},{y},{z}): "
                              f"Original={class_names[original_tags[idx]]}, "
                              f"Ours={class_names[asymptotic_tags[idx]]}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    sys.exit(main())
