"""
Tests for the tessera stats module.
"""

import numpy as np
import pytest


class TestHistogram:
    """Tests for histogram computation."""

    def test_histogram_basic(self, ts):
        """Test basic histogram computation."""
        np.random.seed(42)
        values = np.random.uniform(0, 10, 1000).astype(np.float64)

        hist = ts.stats.histogram(values, n_bins=20, range_min=0.0, range_max=10.0)

        assert len(hist.counts) == 20
        assert sum(hist.counts) == 1000  # all values should be binned

    def test_histogram_log(self, ts):
        """Test log-spaced histogram computation."""
        np.random.seed(42)
        values = np.random.lognormal(0, 0.5, 1000).astype(np.float64)

        hist = ts.stats.histogram_log(values, n_bins=20)

        assert len(hist.counts) == 20
        # Not all values may be binned if outside range
        assert sum(hist.counts) <= 1000

    def test_histogram_edges(self, ts):
        """Test that histogram edges are correct."""
        np.random.seed(42)
        values = np.random.uniform(0, 10, 1000).astype(np.float64)

        hist = ts.stats.histogram(values, n_bins=10, range_min=0.0, range_max=10.0)

        # Check bin edges
        assert len(hist.bin_edges) == 11  # n_bins + 1
        assert hist.bin_edges[0] == pytest.approx(0.0)
        assert hist.bin_edges[-1] == pytest.approx(10.0)


class TestHistogramToPdf:
    """Tests for histogram to PDF conversion."""

    def test_pdf_normalization(self, ts):
        """Test that PDF integrates to approximately 1."""
        np.random.seed(42)
        values = np.random.uniform(0, 10, 10000).astype(np.float64)

        hist = ts.stats.histogram(values, n_bins=50, range_min=0.0, range_max=10.0)
        pdf = ts.stats.histogram_to_pdf(hist)

        # PDF should integrate to 1
        bin_widths = np.diff(np.array(hist.bin_edges))
        integral = sum(p * w for p, w in zip(pdf, bin_widths))

        assert integral == pytest.approx(1.0, rel=0.05)


class TestJackknifeHistogram:
    """Tests for jackknife histogram computation."""

    def test_jackknife_basic(self, ts):
        """Test basic jackknife histogram computation."""
        np.random.seed(42)
        positions = np.random.uniform(0, 10, (1000, 3)).astype(np.float64)
        values = np.random.lognormal(0, 0.5, 1000).astype(np.float64)

        jk = ts.stats.compute_jackknife_histogram(
            positions, values, 10.0, 20, 0.1, 10.0,
            log_bins=True, n_subboxes_per_dim=2, n_threads=0
        )

        assert jk.n_bins == 20
        assert jk.n_subboxes == 8  # 2^3
        assert len(jk.global_counts) == 20
        assert len(jk.counts_error) == 20

    def test_jackknife_subbox_count(self, ts):
        """Test jackknife subbox partitioning."""
        np.random.seed(42)
        positions = np.random.uniform(0, 10, (1000, 3)).astype(np.float64)
        values = np.random.uniform(1, 10, 1000).astype(np.float64)

        for n_sub in [2, 3, 4]:
            jk = ts.stats.compute_jackknife_histogram(
                positions, values, 10.0, 10, 1.0, 10.0,
                log_bins=False, n_subboxes_per_dim=n_sub, n_threads=0
            )
            assert jk.n_subboxes == n_sub ** 3


class TestBinCenters:
    """Tests for bin center computation."""

    def test_bin_centers_linear(self, ts):
        """Test bin centers for linear bins."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        centers = ts.stats.bin_centers(edges, log_space=False)

        expected = [0.5, 1.5, 2.5, 3.5]
        assert len(centers) == len(expected)
        for c, e in zip(centers, expected):
            assert c == pytest.approx(e)

    def test_bin_centers_log(self, ts):
        """Test bin centers for log bins (geometric mean)."""
        edges = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
        centers = ts.stats.bin_centers(edges, log_space=True)

        # Geometric mean of [1, 10] = sqrt(10) ≈ 3.16
        expected = [np.sqrt(10), np.sqrt(1000), np.sqrt(100000)]
        assert len(centers) == len(expected)
        for c, e in zip(centers, expected):
            assert c == pytest.approx(e, rel=0.01)
