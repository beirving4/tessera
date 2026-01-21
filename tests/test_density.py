"""
Tests for the tessera density module.
"""

import numpy as np
import pytest


class TestTetraDensityConfig:
    """Tests for TetraDensityConfig."""

    def test_default_config(self, ts):
        """Test default configuration values."""
        config = ts.density.TetraDensityConfig()
        assert config.n_threads == 0  # auto
        assert config.subbox_enabled is False

    def test_config_attributes(self, ts, density_config):
        """Test setting configuration attributes."""
        config = density_config(grid_size=16, box_size=100.0, output_cells=32)
        assert config.lagrangian_grid_size == 16
        assert config.box_size == 100.0
        assert config.output_cells == 32


class TestDensity2DProjection:
    """Tests for 2D density projection."""

    def test_basic_projection(self, ts, random_positions, density_config):
        """Test basic 2D projection computation."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = density_config(grid_size=8, box_size=10.0, output_cells=16)

        result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)
        density = np.array(result.density)

        # Result is a 2D array
        assert density.shape == (16, 16)
        assert np.all(density >= 0)  # density should be non-negative

    def test_projection_axes(self, ts, random_positions, density_config):
        """Test projection along different axes."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = density_config(grid_size=8, box_size=10.0, output_cells=16)

        for axis in [0, 1, 2]:
            result = ts.density.compute_tetra_density_2d_projection(positions, config, axis)
            density = np.array(result.density)
            assert density.shape == (16, 16)

    def test_mass_conservation(self, ts, random_positions, density_config):
        """Test that total mass is approximately conserved."""
        n_particles = 8**3
        positions = random_positions(n_particles=n_particles, box_size=10.0)
        config = density_config(grid_size=8, box_size=10.0, output_cells=16, n_samples=100)
        config.particle_mass = 1.0

        result = ts.density.compute_tetra_density_2d_projection(positions, config, 2)
        density = np.array(result.density)

        # Total density should be close to total mass (within Monte Carlo error)
        total_density = density.sum()
        cell_area = (config.box_size / config.output_cells) ** 2
        projected_mass = total_density * cell_area

        # Allow 20% tolerance for Monte Carlo sampling
        assert abs(projected_mass - n_particles) / n_particles < 0.2


class TestDensity3D:
    """Tests for 3D density computation."""

    def test_basic_3d(self, ts, random_positions, density_config):
        """Test basic 3D density computation."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = density_config(grid_size=8, box_size=10.0, output_cells=16)

        result = ts.density.compute_tetra_density_3d(positions, config)
        density = np.array(result.density)

        # Result can be reshaped to 3D
        assert density.size == 16 * 16 * 16
        assert np.all(density >= 0)


class TestSortByLagrangianId:
    """Tests for Lagrangian ID sorting."""

    def test_sort_by_lagrangian_id(self, ts, random_positions):
        """Test sorting particles by Lagrangian ID."""
        grid_size = 8
        n_particles = grid_size ** 3
        positions = random_positions(n_particles=n_particles, box_size=10.0)

        # Create Lagrangian IDs (1-indexed, z-major ordering)
        particle_ids = np.arange(1, n_particles + 1, dtype=np.int64)

        # Shuffle
        np.random.shuffle(particle_ids)

        sorted_positions = ts.density.sort_by_lagrangian_id(
            positions, particle_ids, grid_size, id_offset=1
        )

        assert sorted_positions.shape == positions.shape
        assert sorted_positions.dtype == np.float64

    def test_sort_preserves_shape(self, ts):
        """Test that sorting preserves array shape."""
        grid_size = 4
        n_particles = grid_size ** 3
        positions = np.random.uniform(0, 10, (n_particles, 3)).astype(np.float64)
        particle_ids = np.arange(1, n_particles + 1, dtype=np.int64)

        sorted_positions = ts.density.sort_by_lagrangian_id(
            positions, particle_ids, grid_size
        )

        assert sorted_positions.shape == (n_particles, 3)
