# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Install (builds C++ extension via scikit-build-core + CMake)
pip install .

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import tessera; print(tessera.available_modules())"
```

### Development build (manual CMake)

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_HDF5=ON
make -j$(sysctl -n hw.ncpu)   # macOS
make -j$(nproc)                # Linux
```

Then use `PYTHONPATH=/path/to/tessera/build:$PYTHONPATH` to import.

### macOS prerequisites

```bash
brew install hdf5 libomp cmake
```

## Testing

```bash
# Run all tests (KMP_DUPLICATE_LIB_OK needed on macOS for OpenMP conflicts)
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -v

# Run a single test file
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_density.py -v

# Run a specific test
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_density.py::test_2d_projection -v

# Skip tests requiring external data
python -m pytest tests/ -v --ignore=tests/test_io.py -k "not requires_tree_file"
```

**Markers:** `@pytest.mark.requires_hdf5`, `@pytest.mark.requires_tree_file`
**Environment:** `TESSERA_TEST_TREE_FILE` — path to a merger tree HDF5 file for I/O tests.

## Architecture

### Three-layer design

```
C++ Core (include/, src/)  →  pybind11 Bindings (python/bindings/)  →  Python Package (python/tessera/)
```

1. **C++ core** — All computation lives under `tessera::` namespace in `include/` (headers) and `src/` (implementation), organized by module: `density/`, `origami/`, `io/`, `geom/`, `math/`, `stats/`, `cosmo/`, `halo/`, `render/`.

2. **pybind11 bindings** — Each C++ module has a `*_bindings.cpp` in `python/bindings/`. The entry point `module.cpp` creates submodules (`ts.density`, `ts.origami`, etc.). The compiled module is named `_tessera`.

3. **Python package** (`python/tessera/`) — Wraps `_tessera` with a Pythonic API, adds `utils/` (snapshot loading, smoothing, hybrid density) and `visualize/` (matplotlib plotting). The `__init__.py` handles platform detection, OpenMP workarounds on macOS ARM64, and re-exports C++ submodules.

### Import convention

```python
import tessera as ts        # Installed package (wraps _tessera)
import _tessera as ts       # Direct C++ module (development builds)
```

Both expose the same submodules: `ts.density`, `ts.origami`, `ts.stats`, `ts.io`, `ts.geom`, `ts.math`, `ts.cosmo`, `ts.halo`, `ts.render`.

### Key design patterns

- **Unified pipeline**: `ts.origami.run_pipeline()` handles sorting → ORIGAMI → density → PDFs in one call, avoiding redundant data copies.
- **Thread-local buffers**: Density grids use per-thread accumulation (no atomics) for OpenMP parallelism.
- **Streaming indices**: Tetrahedra indices generated on-the-fly, not pre-allocated.
- **Periodic boundaries**: `Vec3f::sub(other, width)` handles simulation box wrapping throughout.
- **In-place operations**: `*_self()` methods on geometry types for zero-allocation paths.
- **Lagrangian sorting**: Particles must be sorted by Lagrangian ID before tessellation (`ts.density.sort_by_lagrangian_id()`). The pipeline does this automatically.

### Adding a new C++ module

1. Add headers in `include/<module>/` and sources in `src/<module>/`
2. Create `python/bindings/<module>_bindings.cpp` with `init_<module>(py::module_& m)`
3. Register in `python/bindings/module.cpp`
4. Add source globs in `CMakeLists.txt`

### NumPy ↔ C++ data flow

Positions are `np.ndarray` of shape `(N, 3)` with dtype `float64` and C-contiguous layout (`np.ascontiguousarray`). Particle IDs are `int64`. Results come back as flat arrays that need reshaping (e.g., `result.density` → `.reshape(256, 256, 256)` for 3D).

## Platform notes

- **macOS ARM64**: `__init__.py` auto-sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` to work around OpenMP crashes. Tests set this in `conftest.py`.
- **HDF5 conflicts**: On macOS, h5py and tessera's HighFive can conflict. The package provides `tessera.check_h5py_compatibility()` for diagnosis.
- **CMake on macOS**: OpenMP detection requires Homebrew libomp; `CMakeLists.txt` handles this with `brew --prefix libomp`.
- **Build option `BUILD_WITH_HDF5=OFF`**: Disables GADGET-4 I/O and merger tree code for environments without HDF5.
