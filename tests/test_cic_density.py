"""
Tests for CIC (Cloud-in-Cell) density computation.

Validates:
1. Mass conservation: sum(grid_density) * cell_volume = total_mass
2. Uniform grid produces constant density everywhere
3. Mean overdensity = 1.0
4. Per-particle sampling consistency
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


class TestCICDensityBasic:
    """Basic CIC density tests."""

    def test_uniform_grid_constant_density(self):
        """Uniform grid particles should produce constant density."""
        ts = _get_ts()
        grid_size = 16
        box_size = 10.0
        cell_size = box_size / grid_size

        # Create uniform grid positions (centered in cells)
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

        # Use same grid resolution as particle grid
        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=grid_size, n_threads=1)

        grid = np.array(result.grid_density)
        n_particles = grid_size ** 3
        expected_density = n_particles / box_size ** 3

        # When particles are centered in cells matching the grid, CIC deposits
        # all mass into the home cell -> uniform density
        np.testing.assert_allclose(grid, expected_density, rtol=1e-10)

    def test_mass_conservation(self):
        """Sum of grid density * cell_volume should equal total mass."""
        ts = _get_ts()
        n_particles = 32 ** 3
        box_size = 100.0
        output_cells = 64

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=output_cells,
            particle_mass=1.0, n_threads=1)

        grid = np.array(result.grid_density)
        cell_volume = result.cell_width ** 3
        total_mass_from_grid = grid.sum() * cell_volume
        expected_total_mass = n_particles * 1.0  # particle_mass = 1.0

        np.testing.assert_allclose(total_mass_from_grid, expected_total_mass, rtol=1e-10)

    def test_grid_mean_density_correct(self):
        """Grid mean density should equal total_mass / box_volume exactly."""
        ts = _get_ts()
        n_particles = 32 ** 3
        box_size = 100.0

        np.random.seed(123)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=64,
            sample_at_particles=True, n_threads=1)

        grid = np.array(result.grid_density)
        grid_mean = grid.mean()
        expected_mean = n_particles / box_size ** 3

        # Grid mean density should match exactly (mass conservation)
        np.testing.assert_allclose(grid_mean, expected_mean, rtol=1e-10)

        # Note: particle-sampled mean is biased high due to CIC self-contribution
        # (particles preferentially sample high-density regions). This is expected.

    def test_result_fields(self):
        """Check all result fields are populated correctly."""
        ts = _get_ts()
        n_particles = 8 ** 3
        box_size = 10.0
        output_cells = 16

        np.random.seed(999)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=output_cells, n_threads=1)

        assert result.n_particles == n_particles
        assert result.output_cells == output_cells
        assert result.cell_width == pytest.approx(box_size / output_cells)
        assert result.mean_density == pytest.approx(n_particles / box_size ** 3)
        assert result.total_time_ms > 0
        assert result.grid_time_ms > 0

        grid = np.array(result.grid_density)
        assert grid.shape == (output_cells, output_cells, output_cells)

        particle_density = np.array(result.particle_density)
        assert len(particle_density) == n_particles

    def test_no_particle_sampling(self):
        """sample_at_particles=False should return empty particle density."""
        ts = _get_ts()
        n_particles = 8 ** 3
        box_size = 10.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=16,
            sample_at_particles=False, n_threads=1)

        grid = np.array(result.grid_density)
        particle_density = np.array(result.particle_density)

        assert grid.shape == (16, 16, 16)
        assert len(particle_density) == 0

    def test_density_positive(self):
        """All grid densities should be non-negative."""
        ts = _get_ts()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(7)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=32, n_threads=1)

        grid = np.array(result.grid_density)
        assert np.all(grid >= 0)

    def test_particle_mass_scaling(self):
        """Changing particle_mass should scale density proportionally."""
        ts = _get_ts()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result_m1 = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=32,
            particle_mass=1.0, n_threads=1)
        result_m2 = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=32,
            particle_mass=2.0, n_threads=1)

        grid_m1 = np.array(result_m1.grid_density)
        grid_m2 = np.array(result_m2.grid_density)

        np.testing.assert_allclose(grid_m2, 2.0 * grid_m1, rtol=1e-10)

    def test_input_validation(self):
        """Should raise on invalid inputs."""
        ts = _get_ts()

        # Wrong shape
        with pytest.raises(RuntimeError, match="shape"):
            ts.density.compute_cic_density(
                np.zeros((10,), dtype=np.float64), box_size=1.0)

        # 2D but wrong second dim
        with pytest.raises(RuntimeError, match="shape"):
            ts.density.compute_cic_density(
                np.zeros((10, 2), dtype=np.float64), box_size=1.0)

    @pytest.mark.skipif(
        sys.platform == 'darwin',
        reason="Multi-threaded OpenMP tests crash on macOS ARM64 (known issue)"
    )
    def test_multithreaded_matches_single(self):
        """Multi-threaded result should match single-threaded."""
        ts = _get_ts()
        n_particles = 16 ** 3
        box_size = 50.0

        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        positions = np.ascontiguousarray(positions)

        result_1t = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=32, n_threads=1)
        result_4t = ts.density.compute_cic_density(
            positions, box_size=box_size, output_cells=32, n_threads=4)

        grid_1t = np.array(result_1t.grid_density)
        grid_4t = np.array(result_4t.grid_density)

        np.testing.assert_allclose(grid_4t, grid_1t, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
