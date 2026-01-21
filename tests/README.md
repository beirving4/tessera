# tessera Test Suite

This directory contains the pytest test suite for the tessera library.

## Quick Start

```bash
# Run all tests (macOS requires KMP_DUPLICATE_LIB_OK for OpenMP)
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_density.py -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "histogram"

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=tessera --cov-report=html
```

## Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Package marker
├── conftest.py            # Shared fixtures and pytest configuration
├── test_density.py        # Density module tests
├── test_io.py             # I/O module tests (including MergerTree)
├── test_origami.py        # ORIGAMI morphology tests
├── test_stats.py          # Statistics module tests
└── test_jackknife_histogram.py  # Jackknife validation tests
```

## Test Modules

### test_density.py

Tests for the tetrahedron-based density computation module.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestTetraDensityConfig` | 2 | Configuration object creation and attributes |
| `TestDensity2DProjection` | 3 | 2D density projections along different axes |
| `TestDensity3D` | 1 | Full 3D density field computation |
| `TestSortByLagrangianId` | 2 | Particle sorting by Lagrangian ID |

### test_origami.py

Tests for the ORIGAMI morphological classification module.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestOrigamiConfig` | 2 | Configuration object creation and attributes |
| `TestComputeMorphology` | 4 | Morphology computation and linear regime detection |
| `TestPipelineConfig` | 2 | Unified pipeline configuration |

### test_stats.py

Tests for the statistics and histogram module.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestHistogram` | 3 | Linear and log-spaced histograms |
| `TestHistogramToPdf` | 1 | PDF normalization |
| `TestJackknifeHistogram` | 2 | Jackknife resampling histograms |
| `TestBinCenters` | 2 | Bin center computation (linear and geometric) |

### test_io.py

Tests for the I/O module, including the high-performance MergerTree reader.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestIOModule` | 2 | Basic I/O module availability |
| `TestMergerTreeHeader` | 2 | MergerTreeHeader struct |
| `TestHaloInfo` | 2 | HaloInfo struct |
| `TestTreeTableEntry` | 1 | TreeTableEntry struct |
| `TestMergerTree` | 10 | MergerTree reader (staged loading, search, cache) |
| `TestHaloTracker` | 6 | Branch tracing and coordinate unwrapping |
| `TestReadMergerTreeHeader` | 1 | Standalone header reading function |

### test_jackknife_histogram.py

Validation tests comparing jackknife histograms against standard (validated) histograms.

| Test | Description |
|------|-------------|
| `test_jackknife_histogram_matches_standard` | Global histogram matches non-jackknife version |
| `test_jackknife_conditional_matches_standard` | Conditional 'all' matches standard histogram |
| `test_jackknife_class_counts_sum_to_total` | Per-class counts sum to total |
| `test_jackknife_errors_are_nonnegative` | Error estimates are non-negative |
| `test_jackknife_subbox_counts` | Subbox partitioning is correct |
| `test_linear_bins` | Linear binning works correctly |

## Fixtures

Shared fixtures are defined in `conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `ts` | session | Import tessera module |
| `random_positions` | function | Generate random particle positions |
| `grid_positions` | function | Generate grid-based particle positions |
| `density_config` | function | Create TetraDensityConfig |
| `origami_config` | function | Create OrigamiConfig |
| `merger_tree_path` | session | Path to test merger tree file |

## Markers

Custom pytest markers for conditional test execution:

```python
@pytest.mark.requires_hdf5        # Test requires HDF5 support
@pytest.mark.requires_tree_file   # Test requires a merger tree file
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` on macOS to handle OpenMP library conflicts |
| `TESSERA_TEST_TREE_FILE` | Path to merger tree HDF5 file for I/O tests |

## Running Tests with Merger Tree Data

The MergerTree tests require an actual GADGET-4 merger tree file. Set the environment variable:

```bash
export TESSERA_TEST_TREE_FILE="/path/to/trees.hdf5"
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_io.py -v
```

Without this file, MergerTree tests will be skipped.

## Running Tests in CI

The CI workflow runs pytest after the inline module tests:

```yaml
- name: Run pytest test suite
  env:
    KMP_DUPLICATE_LIB_OK: TRUE
  run: |
    python -m pytest tests/ -v --ignore=tests/test_io.py -k "not requires_tree_file"
```

Note: IO tests requiring tree files are skipped in CI since test data is not available.

## Adding New Tests

### 1. Create a new test file

```python
# tests/test_new_module.py
"""Tests for the new_module."""

import numpy as np
import pytest


class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self, ts):
        """Test basic functionality."""
        # Use the ts fixture to get the tessera module
        result = ts.new_module.some_function()
        assert result is not None
```

### 2. Add fixtures if needed

Add shared fixtures to `conftest.py`:

```python
@pytest.fixture
def new_config(ts):
    """Create a new module configuration."""
    def _create(**kwargs):
        config = ts.new_module.Config()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config
    return _create
```

### 3. Use markers for conditional tests

```python
@pytest.mark.requires_hdf5
def test_hdf5_feature(self, ts):
    """Test that requires HDF5."""
    pass

@pytest.mark.requires_tree_file
def test_tree_feature(self, ts, merger_tree_path):
    """Test that requires merger tree data."""
    if merger_tree_path is None:
        pytest.skip("No merger tree file available")
    # ... test code
```

## Test Coverage

To generate a coverage report:

```bash
pip install pytest-cov
python -m pytest tests/ --cov=tessera --cov-report=html
open htmlcov/index.html  # View report in browser
```

## Troubleshooting

### OpenMP Errors on macOS

If you see "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized":

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python -m pytest tests/ -v
```

### Import Errors

If you get "generic_type: type already registered" errors, ensure you're importing `tessera` (the package) rather than `_tessera` (the C++ module) directly. The test fixtures handle this correctly.

### Missing Test Data

MergerTree tests will be skipped if no tree file is available. This is expected behavior in CI environments.

## Test History

| Date | Tests | Description |
|------|-------|-------------|
| 2025-01-20 | 54 | Initial test suite with density, origami, stats, io modules |
