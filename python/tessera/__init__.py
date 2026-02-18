"""
tessera - High-performance phase-space tessellation for cosmological simulations.

This package provides Python bindings to a C++ library for computing density fields
from N-body simulation data using tetrahedron-based Monte Carlo integration.

Example usage:
    >>> import tessera as ts
    >>>
    >>> # Create a random number generator
    >>> gen = ts.math.Generator.new_time_seed()
    >>> print(gen.uniform(0, 1))
    >>>
    >>> # Work with geometry primitives
    >>> v1 = ts.geom.Vec3f(1.0, 2.0, 3.0)
    >>> v2 = ts.geom.Vec3f(4.0, 5.0, 6.0)
    >>> print(v1.dot(v2))
    >>>
    >>> # Create tetrahedra
    >>> tet = ts.geom.Tetra(
    ...     ts.geom.Vec3f(0, 0, 0),
    ...     ts.geom.Vec3f(1, 0, 0),
    ...     ts.geom.Vec3f(0, 1, 0),
    ...     ts.geom.Vec3f(0, 0, 1)
    ... )
    >>> print(tet.volume())

Modules:
    geom - Geometry primitives (Vec3f, Tetra, Grid, CellBounds)
    math - Random number generation (Generator, TauswortheGenerator)
    density - Density field computation (Quantity, Buffer)
    io - File I/O for simulation formats (GadgetHeader, SheetHeader)
    render - Rendering interface
    cosmo - Cosmological calculations (rho_critical, rho_average)
    halo - Halo finding and analysis (SubhaloFinder, RadiusType)
    origami - ORIGAMI morphology classification
    stats - Statistical functions
"""

# =============================================================================
# Platform-specific configuration (MUST run before importing native module)
# =============================================================================
import os as _os
import platform as _platform
import sys as _sys


def _has_libomp_conflict():
    """Check if tessera's native module links a different libomp than the process.

    On macOS, loading two libomp copies (e.g. Homebrew + conda) causes crashes.
    If tessera was built against conda's libomp (the same one numpy/scipy use),
    multi-threading is safe and we should NOT force OMP_NUM_THREADS=1.
    """
    try:
        import subprocess as _sp
        _so_path = _os.path.join(_os.path.dirname(__file__), "_tessera" +
                                  _os.path.splitext(__import__('importlib').util
                                  .find_spec("_tessera").origin)[1]
                                  if __import__('importlib').util.find_spec("_tessera")
                                  else "")
        if not _os.path.isfile(_so_path):
            # Installed as top-level _tessera, check site-packages
            import importlib.util
            spec = importlib.util.find_spec("_tessera")
            _so_path = spec.origin if spec else ""
        if not _so_path or not _os.path.isfile(_so_path):
            return True  # Can't check, assume conflict for safety
        result = _sp.run(["otool", "-L", _so_path], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "libomp" in line:
                # Extract the dylib path
                path = line.strip().split()[0]
                conda_prefix = _os.environ.get("CONDA_PREFIX", "")
                if conda_prefix and path.startswith(conda_prefix):
                    return False  # Same libomp as conda — no conflict
                if "/homebrew/" in path.lower() or "/opt/homebrew/" in path:
                    return True   # Homebrew libomp — likely conflicts with conda
                return True  # Unknown libomp — assume conflict
        return False  # No libomp linked (OpenMP disabled) — no conflict
    except Exception:
        return True  # Can't determine — assume conflict for safety


def _configure_platform():
    """
    Configure environment for platform-specific issues.

    On macOS with Apple Silicon, crashes can occur when multiple libomp copies
    are loaded (e.g. Homebrew's for tessera + conda's for numpy). If tessera
    was built against conda's libomp (matching numpy/scipy), multi-threading
    is safe and OMP_NUM_THREADS is left unrestricted.

    Only when a libomp conflict is detected does this force OMP_NUM_THREADS=1.
    KMP_DUPLICATE_LIB_OK=TRUE is always set on macOS ARM64 as a safety net.
    """
    system = _platform.system()
    machine = _platform.machine()

    # Detect macOS with Apple Silicon (ARM64)
    is_macos_arm64 = (system == "Darwin" and machine == "arm64")

    if is_macos_arm64:
        # Fix duplicate OpenMP library conflicts
        if "KMP_DUPLICATE_LIB_OK" not in _os.environ:
            _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Only force single-thread if a libomp conflict is detected
        if "OMP_NUM_THREADS" not in _os.environ:
            if _has_libomp_conflict():
                _os.environ["OMP_NUM_THREADS"] = "1"

        return True  # Platform workarounds applied

    return False  # No workarounds needed


# Apply platform configuration before any native imports
_platform_workarounds_applied = _configure_platform()


def get_platform_info():
    """
    Get information about the current platform and any workarounds applied.

    Returns
    -------
    dict
        Dictionary with keys:
        - system: Operating system (e.g., 'Darwin', 'Linux', 'Windows')
        - machine: CPU architecture (e.g., 'arm64', 'x86_64')
        - is_macos_arm64: True if running on macOS with Apple Silicon
        - workarounds_applied: True if platform workarounds were applied
        - omp_num_threads: Current OMP_NUM_THREADS setting
        - kmp_duplicate_lib_ok: Current KMP_DUPLICATE_LIB_OK setting
    """
    return {
        "system": _platform.system(),
        "machine": _platform.machine(),
        "is_macos_arm64": (_platform.system() == "Darwin" and _platform.machine() == "arm64"),
        "workarounds_applied": _platform_workarounds_applied,
        "omp_num_threads": _os.environ.get("OMP_NUM_THREADS"),
        "kmp_duplicate_lib_ok": _os.environ.get("KMP_DUPLICATE_LIB_OK"),
    }


def check_h5py_compatibility(warn=True):
    """
    Check for potential HDF5 library conflicts with h5py.

    On macOS, there can be conflicts between h5py's HDF5 library and tessera's
    HighFive library, causing crashes when both are used in the same process.

    Parameters
    ----------
    warn : bool
        If True, print a warning if h5py is loaded and we're on a potentially
        problematic platform.

    Returns
    -------
    dict
        Dictionary with keys:
        - h5py_loaded: True if h5py is currently imported
        - potential_conflict: True if there's a potential HDF5 conflict
        - recommendation: String with recommended action
    """
    h5py_loaded = "h5py" in _sys.modules
    is_macos = _platform.system() == "Darwin"
    potential_conflict = h5py_loaded and is_macos

    if potential_conflict:
        recommendation = (
            "h5py and tessera may have conflicting HDF5 libraries on macOS. "
            "If you experience crashes, try running h5py operations in a separate "
            "process, or use the two-stage workflow in halo_evolution examples."
        )
    else:
        recommendation = "No known conflicts detected."

    if warn and potential_conflict:
        import warnings
        warnings.warn(recommendation, RuntimeWarning)

    return {
        "h5py_loaded": h5py_loaded,
        "potential_conflict": potential_conflict,
        "recommendation": recommendation,
    }


# =============================================================================
# Package metadata
# =============================================================================

__version__ = "1.0.0"

# Try to import the native module
_native_available = False

try:
    # Try relative import first (for pip installed package)
    from ._tessera import geom, math, density, io, render, cosmo, halo
    _native_available = True
except ImportError:
    try:
        # Fall back to absolute import (for development with PYTHONPATH)
        from _tessera import geom, math, density, io, render, cosmo, halo
        _native_available = True
    except ImportError:
        pass

# Import origami and stats (they may be in different submodules)
origami = None
stats = None

if _native_available:
    try:
        from ._tessera import origami, stats
    except ImportError:
        try:
            from _tessera import origami, stats
        except ImportError:
            pass

if _native_available:
    # Convenience imports at top level from submodules
    Vec3f = geom.Vec3f
    Tetra = geom.Tetra
    Grid = geom.Grid
    GridLocation = geom.GridLocation
    CellBounds = geom.CellBounds
    TetraIdxs = geom.TetraIdxs
    Generator = math.Generator
    GeneratorType = math.GeneratorType
    Quantity = density.Quantity
    CosmologyHeader = io.CosmologyHeader
    CatalogHeader = io.CatalogHeader
    SheetHeader = io.SheetHeader
    GadgetHeader = io.GadgetHeader
    ByteOrder = io.ByteOrder
    # Halo module
    RadiusType = halo.RadiusType
    HaloGrid = halo.HaloGrid
    SubhaloFinder = halo.SubhaloFinder
    HaloData = halo.HaloData
    Val = halo.Val
else:
    import warnings
    warnings.warn("Native module not available. Some functionality will be limited.")
    geom = None
    math = None
    density = None
    io = None
    render = None
    cosmo = None
    halo = None

# =============================================================================
# Python submodules (utils and visualize)
# =============================================================================

# Import Python submodules - these are always available
from . import utils
from . import visualize

__all__ = [
    # C++ Submodules
    "geom",
    "math",
    "density",
    "io",
    "render",
    "cosmo",
    "halo",
    "origami",
    "stats",
    # Python Submodules
    "utils",
    "visualize",
    # Geometry
    "Vec3f",
    "Tetra",
    "Grid",
    "GridLocation",
    "CellBounds",
    "TetraIdxs",
    # Math
    "Generator",
    "GeneratorType",
    # Density
    "Quantity",
    # I/O
    "CosmologyHeader",
    "CatalogHeader",
    "SheetHeader",
    "GadgetHeader",
    "ByteOrder",
    # Halo
    "RadiusType",
    "HaloGrid",
    "SubhaloFinder",
    "HaloData",
    "Val",
    # Platform utilities
    "get_platform_info",
    "check_h5py_compatibility",
    # Meta
    "__version__",
]
