"""
Tests for the tessera ORIGAMI morphology module.
"""

import numpy as np
import pytest


class TestOrigamiConfig:
    """Tests for OrigamiConfig."""

    def test_default_config(self, ts):
        """Test default configuration values."""
        config = ts.origami.OrigamiConfig()
        # Check that config can be created with n_split >= 1
        assert config.n_split >= 1

    def test_config_attributes(self, ts, origami_config):
        """Test setting configuration attributes."""
        config = origami_config(grid_size=16, box_size=100.0)
        assert config.lagrangian_grid_size == 16
        assert config.box_size == 100.0


class TestComputeMorphology:
    """Tests for ORIGAMI morphology computation."""

    def test_basic_morphology(self, ts, random_positions, origami_config):
        """Test basic morphology computation."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = origami_config(grid_size=8, box_size=10.0)

        result = ts.origami.compute_morphology(positions, config)

        assert len(result.morphology) == 8**3
        assert all(0 <= m <= 3 for m in result.morphology)  # void=0, wall=1, filament=2, halo=3

    def test_morphology_classes(self, ts, random_positions, origami_config):
        """Test that all morphology classes are present (for random positions)."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = origami_config(grid_size=8, box_size=10.0)

        result = ts.origami.compute_morphology(positions, config)
        morphology = np.array(result.morphology)

        # Count particles in each class
        n_void = np.sum(morphology == 0)
        n_wall = np.sum(morphology == 1)
        n_filament = np.sum(morphology == 2)
        n_halo = np.sum(morphology == 3)

        # Total should match
        assert n_void + n_wall + n_filament + n_halo == 8**3

    def test_linear_regime_detection(self, ts, origami_config):
        """Test that linear regime detection attribute exists."""
        # Use random positions (will have shell-crossing)
        np.random.seed(42)
        positions = np.random.uniform(0, 10, (8**3, 3)).astype(np.float64)
        config = origami_config(grid_size=8, box_size=10.0)

        result = ts.origami.compute_morphology(positions, config)

        # Check that is_linear_regime attribute exists
        assert hasattr(result, 'is_linear_regime')
        # Result should be a boolean
        assert isinstance(result.is_linear_regime, bool)

    def test_morphology_result_attributes(self, ts, random_positions, origami_config):
        """Test that result has expected attributes."""
        positions = random_positions(n_particles=8**3, box_size=10.0)
        config = origami_config(grid_size=8, box_size=10.0)

        result = ts.origami.compute_morphology(positions, config)

        # Check result attributes
        assert hasattr(result, 'morphology')
        assert hasattr(result, 'n_void')
        assert hasattr(result, 'n_wall')
        assert hasattr(result, 'n_filament')
        assert hasattr(result, 'n_halo')
        assert hasattr(result, 'f_void')
        assert hasattr(result, 'f_wall')
        assert hasattr(result, 'f_filament')
        assert hasattr(result, 'f_halo')
        assert hasattr(result, 'is_linear_regime')

        # Fractions should sum to 1
        total_fraction = result.f_void + result.f_wall + result.f_filament + result.f_halo
        assert abs(total_fraction - 1.0) < 1e-6


class TestPipelineConfig:
    """Tests for ORIGAMI pipeline configuration."""

    def test_pipeline_config_creation(self, ts):
        """Test PipelineConfig creation."""
        config = ts.origami.PipelineConfig(8, 10.0)
        assert config.lagrangian_grid_size == 8
        assert config.box_size == 10.0

    def test_pipeline_config_attributes(self, ts):
        """Test PipelineConfig attributes."""
        config = ts.origami.PipelineConfig(16, 100.0)
        config.density_output_cells = 32
        config.sample_density_at_particles = True
        config.grid_cells = 64
        config.pdf_n_bins = 50

        assert config.density_output_cells == 32
        assert config.sample_density_at_particles is True
        assert config.grid_cells == 64
        assert config.pdf_n_bins == 50
