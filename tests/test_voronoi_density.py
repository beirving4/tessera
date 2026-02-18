"""
Tests for Voronoi (VTFE) density computation.

Validates:
1. Volume sum equals box volume (mass conservation)
2. Mean overdensity ≈ 1.0
3. Volumes match freud-analysis (if installed) within floating-point tolerance
4. Uniform grid produces equal volumes
"""

import os
import sys
import pytest
import numpy as np

# Handle macOS OpenMP conflict
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Add build directory to path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_dir = os.path.join(_repo_root, 'build')
if os.path.exists(_build_dir):
    sys.path.insert(0, _build_dir)


def _get_ts():
    """Import tessera module."""
    try:
        import tessera as ts
        return ts
    except ImportError:
        import _tessera as ts
        return ts


def _parallel_threads():
    """Return thread count for parallel tests.

    Uses 4 threads on all platforms.  On macOS, the dual-libomp crash is
    avoided by building tessera against conda's libomp (same copy that
    numpy/scipy use).  If the old Homebrew-linked build is still installed,
    __init__.py forces OMP_NUM_THREADS=1 and the OpenMP runtime will
    silently clamp the thread count.
    """
    return 4


def _has_voronoi():
    """Check if Voronoi density is available."""
    ts = _get_ts()
    try:
        # Check if the function exists and is not a stub
        func = ts.density.compute_voronoi_density
        # Check the docstring for the "requires BUILD_WITH_VORO" stub marker
        if func.__doc__ and "requires BUILD_WITH_VORO" in func.__doc__:
            return False
        return True
    except AttributeError:
        return False


requires_voro = pytest.mark.skipif(
    not _has_voronoi(),
    reason="tessera built without voro++ support (BUILD_WITH_VORO=OFF)"
)


@requires_voro
class TestVoronoiDensityBasic:
    """Basic Voronoi density tests."""

    def test_uniform_grid_equal_volumes(self):
        """Uniform grid particles should all have equal Voronoi volumes."""
        ts = _get_ts()
        grid_size = 8
        box_size = 10.0
        cell_size = box_size / grid_size

        # Create uniform grid positions
        positions = []
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    positions.append([
                        (x + 0.5) * cell_size,
                        (y + 0.5) * cell_size,
                        (z + 0.5) * cell_size
                    ])
        positions = np.array(positions, dtype=np.float64)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)

        n_particles = grid_size ** 3
        expected_volume = box_size ** 3 / n_particles

        volumes = np.array(result.volumes)
        assert len(volumes) == n_particles
        # rtol ~1e-7 accounts for the small random jitter applied to break
        # voro++ duplicate detection on co-located particles
        np.testing.assert_allclose(volumes, expected_volume, rtol=1e-7)

    def test_volume_sum_equals_box_volume(self):
        """Sum of all Voronoi volumes should equal total box volume."""
        ts = _get_ts()
        n_particles = 32 ** 3
        box_size = 100.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)

        volumes = np.array(result.volumes)
        total_volume = volumes.sum()
        expected_volume = box_size ** 3

        np.testing.assert_allclose(total_volume, expected_volume, rtol=1e-8)

    def test_volume_weighted_overdensity_unity(self):
        """Volume-weighted mean overdensity should be exactly 1.0.

        The arithmetic mean of V_mean/V_i != 1 (Jensen's inequality),
        but the volume-weighted mean sum(V_i * V_mean/V_i) / V_box = 1.
        """
        ts = _get_ts()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(123)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)

        density = np.array(result.density)
        volumes = np.array(result.volumes)
        overdensity = density / result.mean_density

        # Volume-weighted mean overdensity = sum(V_i * od_i) / box_volume
        vol_weighted_mean = np.sum(volumes * overdensity) / box_size ** 3
        np.testing.assert_allclose(vol_weighted_mean, 1.0, rtol=1e-8)

    def test_result_fields(self):
        """Check all result fields are populated correctly."""
        ts = _get_ts()
        n_particles = 8 ** 3
        box_size = 10.0

        np.random.seed(999)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)

        assert result.n_particles == n_particles
        assert result.mean_density == pytest.approx(n_particles / box_size ** 3)
        assert result.mean_volume == pytest.approx(box_size ** 3 / n_particles)
        assert result.total_time_ms > 0
        assert len(np.array(result.density)) == n_particles
        assert len(np.array(result.volumes)) == n_particles

    def test_no_volumes_option(self):
        """compute_volumes=False should return empty volumes array."""
        ts = _get_ts()
        n_particles = 8 ** 3
        box_size = 10.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1, compute_volumes=False)

        density = np.array(result.density)
        volumes = np.array(result.volumes)

        assert len(density) == n_particles
        assert len(volumes) == 0

    def test_density_positive(self):
        """All densities should be positive."""
        ts = _get_ts()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(7)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)

        density = np.array(result.density)
        assert np.all(density > 0)

    def test_parallel_matches_serial(self):
        """Parallel should give identical volumes to serial (n_threads=1).

        On macOS ARM64, OpenMP is limited to 1 thread; the per-thread
        voro_compute code path is still exercised.  On Linux, n_threads=4
        tests actual thread-safety.
        """
        ts = _get_ts()
        n_threads = _parallel_threads()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(314)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result_serial = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)
        result_parallel = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=n_threads)

        vol_serial = np.array(result_serial.volumes)
        vol_parallel = np.array(result_parallel.volumes)
        np.testing.assert_allclose(vol_parallel, vol_serial, rtol=1e-12)

    def test_parallel_volume_sum(self):
        """Volume sum with parallel threads still equals box volume."""
        ts = _get_ts()
        n_threads = _parallel_threads()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(271)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=n_threads)

        volumes = np.array(result.volumes)
        np.testing.assert_allclose(volumes.sum(), box_size ** 3, rtol=1e-8)

    def test_parallel_uniform_grid(self):
        """Uniform grid with parallel threads gives equal volumes."""
        ts = _get_ts()
        n_threads = _parallel_threads()
        grid_size = 8
        box_size = 10.0
        cell_size = box_size / grid_size

        positions = []
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    positions.append([
                        (x + 0.5) * cell_size,
                        (y + 0.5) * cell_size,
                        (z + 0.5) * cell_size
                    ])
        positions = np.array(positions, dtype=np.float64)

        result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=n_threads)

        n_particles = grid_size ** 3
        expected_volume = box_size ** 3 / n_particles
        volumes = np.array(result.volumes)
        np.testing.assert_allclose(volumes, expected_volume, rtol=1e-7)

    def test_input_validation(self):
        """Should raise on invalid inputs."""
        ts = _get_ts()

        # Wrong shape
        with pytest.raises(RuntimeError, match="shape"):
            ts.density.compute_voronoi_density(
                np.zeros((10,), dtype=np.float64), box_size=1.0)

        # 2D but wrong second dim
        with pytest.raises(RuntimeError, match="shape"):
            ts.density.compute_voronoi_density(
                np.zeros((10, 2), dtype=np.float64), box_size=1.0)


@requires_voro
class TestVoronoiDensityFreudValidation:
    """Validate against freud-analysis Voronoi computation."""

    @pytest.fixture(autouse=True)
    def check_freud(self):
        """Skip if freud is not installed."""
        pytest.importorskip("freud", reason="freud-analysis not installed")

    def test_volumes_match_freud(self):
        """Voronoi volumes should match freud within floating-point tolerance."""
        import freud
        ts = _get_ts()

        n_per_dim = 16
        n_particles = n_per_dim ** 3
        box_size = 50.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        # Compute with tessera
        ts_result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)
        ts_volumes = np.array(ts_result.volumes)

        # Compute with freud
        box = freud.box.Box.cube(box_size)
        # freud expects positions centered at origin
        centered_pos = positions - box_size / 2.0
        voro = freud.locality.Voronoi()
        voro.compute((box, centered_pos.astype(np.float32)))
        freud_volumes = voro.volumes

        # Sort both by volume for comparison (particle ordering may differ)
        # Actually, both should be indexed by the same particle, so compare directly
        np.testing.assert_allclose(ts_volumes, freud_volumes, rtol=1e-6,
                                   err_msg="Voronoi volumes differ from freud")

    def test_volume_sum_matches_freud(self):
        """Both implementations should agree on total volume."""
        import freud
        ts = _get_ts()

        n_particles = 32 ** 3
        box_size = 100.0

        np.random.seed(99)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        # tessera
        ts_result = ts.density.compute_voronoi_density(
            positions, box_size=box_size, n_threads=1)
        ts_total = np.array(ts_result.volumes).sum()

        # freud
        box = freud.box.Box.cube(box_size)
        centered_pos = positions - box_size / 2.0
        voro = freud.locality.Voronoi()
        voro.compute((box, centered_pos.astype(np.float32)))
        freud_total = voro.volumes.sum()

        np.testing.assert_allclose(ts_total, freud_total, rtol=1e-4)
        np.testing.assert_allclose(ts_total, box_size ** 3, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
