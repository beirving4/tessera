#!/usr/bin/env python3
"""
Test jackknife histogram implementation against non-jackknife version.

The non-jackknife histogram functions have been validated against the original
ORIGAMI code, so they serve as ground truth. The jackknife implementation should
produce identical global (full-box) histograms and PDFs.

Usage:
    python test_jackknife_histogram.py
    pytest test_jackknife_histogram.py -v
"""

import numpy as np
import sys

# Import tessera
try:
    import tessera as ts
except ImportError:
    # Try development build
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))
    import _tessera as ts


def test_jackknife_histogram_matches_standard():
    """Test that jackknife global histogram matches standard histogram."""
    np.random.seed(42)

    # Create test data
    n_particles = 100000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    overdensity = np.random.lognormal(mean=0, sigma=0.8, size=n_particles).astype(np.float64)

    # Parameters
    n_bins = 50
    range_min = max(overdensity.min(), 1e-3)
    range_max = overdensity.max() * 1.001

    # Method 1: Non-jackknife
    hist_result = ts.stats.histogram_log(overdensity, n_bins, range_min, range_max)
    pdf_nojk = ts.stats.histogram_to_pdf(hist_result)
    counts_nojk = np.array(hist_result.counts)
    edges_nojk = np.array(hist_result.bin_edges)

    # Method 2: Jackknife
    jk_result = ts.stats.compute_jackknife_histogram(
        positions, overdensity, box_size, n_bins, range_min, range_max,
        log_bins=True, n_subboxes_per_dim=2, n_threads=0
    )
    counts_jk = np.array(jk_result.global_counts)
    pdf_jk = np.array(jk_result.global_pdf)
    edges_jk = np.array(jk_result.bin_edges)

    # Assertions
    assert np.allclose(edges_nojk, edges_jk, rtol=1e-10), "Bin edges should match"
    assert np.array_equal(counts_nojk, counts_jk), "Histogram counts should match exactly"
    assert np.allclose(pdf_nojk, pdf_jk, rtol=1e-10, atol=1e-15), "PDF values should match"


def test_jackknife_conditional_matches_standard():
    """Test that jackknife conditional 'all' matches standard histogram."""
    np.random.seed(42)

    # Create test data
    n_particles = 100000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    overdensity = np.random.lognormal(mean=0, sigma=0.8, size=n_particles).astype(np.float64)
    morphology = np.random.randint(0, 4, n_particles, dtype=np.uint8)

    # Parameters
    n_bins = 50
    range_min = max(overdensity.min(), 1e-3)
    range_max = overdensity.max() * 1.001

    # Method 1: Non-jackknife
    hist_result = ts.stats.histogram_log(overdensity, n_bins, range_min, range_max)
    counts_nojk = np.array(hist_result.counts)
    pdf_nojk = ts.stats.histogram_to_pdf(hist_result)

    # Method 2: Jackknife conditional
    jk_cond = ts.stats.compute_jackknife_conditional_histogram(
        positions, overdensity, morphology, box_size, n_bins, range_min, range_max,
        log_bins=True, n_subboxes_per_dim=2, n_threads=0
    )
    counts_cond_all = np.array(jk_cond.all.global_counts)
    pdf_cond_all = np.array(jk_cond.all.global_pdf)

    # Assertions
    assert np.array_equal(counts_nojk, counts_cond_all), "Conditional 'all' counts should match"
    assert np.allclose(pdf_nojk, pdf_cond_all, rtol=1e-10, atol=1e-15), "Conditional 'all' PDF should match"


def test_jackknife_class_counts_sum_to_total():
    """Test that per-class counts sum to total counts."""
    np.random.seed(42)

    # Create test data
    n_particles = 100000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    overdensity = np.random.lognormal(mean=0, sigma=0.8, size=n_particles).astype(np.float64)
    morphology = np.random.randint(0, 4, n_particles, dtype=np.uint8)

    # Parameters
    n_bins = 50
    range_min = max(overdensity.min(), 1e-3)
    range_max = overdensity.max() * 1.001

    # Compute jackknife conditional histogram
    jk_cond = ts.stats.compute_jackknife_conditional_histogram(
        positions, overdensity, morphology, box_size, n_bins, range_min, range_max,
        log_bins=True, n_subboxes_per_dim=2, n_threads=0
    )

    # Sum per-class counts
    class_counts_sum = (
        np.array(jk_cond.void_class.global_counts) +
        np.array(jk_cond.wall_class.global_counts) +
        np.array(jk_cond.filament_class.global_counts) +
        np.array(jk_cond.halo_class.global_counts)
    )
    total_counts = np.array(jk_cond.all.global_counts)

    assert np.array_equal(total_counts, class_counts_sum), "Per-class counts should sum to total"


def test_jackknife_errors_are_nonnegative():
    """Test that jackknife errors are non-negative."""
    np.random.seed(42)

    # Create test data
    n_particles = 100000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    overdensity = np.random.lognormal(mean=0, sigma=0.8, size=n_particles).astype(np.float64)

    # Parameters
    n_bins = 50
    range_min = max(overdensity.min(), 1e-3)
    range_max = overdensity.max() * 1.001

    # Compute jackknife histogram
    jk_result = ts.stats.compute_jackknife_histogram(
        positions, overdensity, box_size, n_bins, range_min, range_max,
        log_bins=True, n_subboxes_per_dim=2, n_threads=0
    )

    counts_error = np.array(jk_result.counts_error)
    pdf_error = np.array(jk_result.pdf_error)

    assert np.all(counts_error >= 0), "Counts error should be non-negative"
    assert np.all(pdf_error >= 0), "PDF error should be non-negative"


def test_jackknife_subbox_counts():
    """Test that jackknife correctly tracks subbox data."""
    np.random.seed(42)

    # Create test data
    n_particles = 10000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    overdensity = np.random.lognormal(mean=0, sigma=0.8, size=n_particles).astype(np.float64)

    # Parameters
    n_bins = 20
    range_min = max(overdensity.min(), 1e-3)
    range_max = overdensity.max() * 1.001
    n_subboxes_per_dim = 2
    n_subboxes = n_subboxes_per_dim ** 3

    # Compute jackknife histogram
    jk_result = ts.stats.compute_jackknife_histogram(
        positions, overdensity, box_size, n_bins, range_min, range_max,
        log_bins=True, n_subboxes_per_dim=n_subboxes_per_dim, n_threads=0
    )

    assert jk_result.n_subboxes == n_subboxes, f"Expected {n_subboxes} subboxes"
    assert len(jk_result.subbox_counts) == n_subboxes, "Should have counts for each subbox"
    assert len(jk_result.subbox_n_total) == n_subboxes, "Should have n_total for each subbox"

    # Sum of subbox counts should equal global counts
    subbox_sum = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_subboxes):
        subbox_sum += np.array(jk_result.subbox_counts[i])

    global_counts = np.array(jk_result.global_counts)
    assert np.array_equal(global_counts, subbox_sum), "Subbox counts should sum to global"


def test_linear_bins():
    """Test jackknife with linear binning."""
    np.random.seed(42)

    # Create test data
    n_particles = 10000
    box_size = 100.0
    positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
    values = np.random.normal(0, 1, size=n_particles).astype(np.float64)

    # Parameters
    n_bins = 30
    range_min = values.min()
    range_max = values.max() * 1.001

    # Compute jackknife histogram with linear bins
    jk_result = ts.stats.compute_jackknife_histogram(
        positions, values, box_size, n_bins, range_min, range_max,
        log_bins=False,  # Linear bins
        n_subboxes_per_dim=2, n_threads=0
    )

    # Basic checks
    assert jk_result.n_bins == n_bins
    assert len(jk_result.bin_edges) == n_bins + 1
    assert len(jk_result.global_counts) == n_bins

    # Check that bin edges are linearly spaced
    edges = np.array(jk_result.bin_edges)
    diffs = np.diff(edges)
    assert np.allclose(diffs, diffs[0], rtol=1e-10), "Linear bins should have equal width"


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Jackknife histogram matches standard", test_jackknife_histogram_matches_standard),
        ("Jackknife conditional 'all' matches standard", test_jackknife_conditional_matches_standard),
        ("Per-class counts sum to total", test_jackknife_class_counts_sum_to_total),
        ("Jackknife errors are non-negative", test_jackknife_errors_are_nonnegative),
        ("Jackknife subbox counts", test_jackknife_subbox_counts),
        ("Linear bins", test_linear_bins),
    ]

    print("=" * 70)
    print("Jackknife Histogram Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}")
            print(f"         {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
