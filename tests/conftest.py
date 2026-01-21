"""
Pytest configuration and shared fixtures for tessera tests.
"""

import os
import sys
import pytest
import numpy as np

# Handle macOS OpenMP conflict
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Add build directory to path for development testing
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_dir = os.path.join(_repo_root, 'build')
if os.path.exists(_build_dir):
    sys.path.insert(0, _build_dir)


@pytest.fixture(scope="session")
def ts():
    """Import tessera module."""
    try:
        import tessera as ts
        return ts
    except ImportError:
        # Fall back to _tessera for development builds
        import _tessera as ts
        return ts


@pytest.fixture
def random_positions():
    """Generate random particle positions for testing."""
    def _generate(n_particles=512, box_size=10.0, seed=42):
        np.random.seed(seed)
        positions = np.random.uniform(0, box_size, (n_particles, 3)).astype(np.float64)
        return np.ascontiguousarray(positions)
    return _generate


@pytest.fixture
def grid_positions():
    """Generate grid-based particle positions (8^3 = 512 particles)."""
    def _generate(grid_size=8, box_size=10.0):
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
        return np.array(positions, dtype=np.float64)
    return _generate


@pytest.fixture
def density_config(ts):
    """Create a basic density configuration."""
    def _create(grid_size=8, box_size=10.0, output_cells=16, n_samples=10):
        config = ts.density.TetraDensityConfig()
        config.lagrangian_grid_size = grid_size
        config.box_size = box_size
        config.output_cells = output_cells
        config.n_samples = n_samples
        config.n_threads = 1
        config.periodic = True
        config.particle_mass = 1.0
        return config
    return _create


@pytest.fixture
def origami_config(ts):
    """Create a basic ORIGAMI configuration."""
    def _create(grid_size=8, box_size=10.0):
        config = ts.origami.OrigamiConfig()
        config.lagrangian_grid_size = grid_size
        config.box_size = box_size
        config.n_threads = 1
        config.n_split = 1
        return config
    return _create


# Path to test data (merger tree file)
@pytest.fixture(scope="session")
def merger_tree_path():
    """Path to test merger tree file (if available)."""
    # Check for environment variable first
    env_path = os.environ.get('TESSERA_TEST_TREE_FILE')
    if env_path and os.path.exists(env_path):
        return env_path

    # Check common locations
    test_paths = [
        "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/trees.hdf5",
    ]

    for path in test_paths:
        if os.path.exists(path):
            return path

    return None


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_hdf5: mark test as requiring HDF5 support"
    )
    config.addinivalue_line(
        "markers", "requires_tree_file: mark test as requiring a merger tree file"
    )
