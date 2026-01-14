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


def _configure_platform():
    """
    Configure environment for platform-specific issues.

    On macOS with Apple Silicon, there are known issues with OpenMP that can
    cause crashes during multi-threaded density computations. This function
    automatically sets environment variables to work around these issues.

    Issues addressed:
    1. OpenMP thread crashes on Apple Silicon: Set OMP_NUM_THREADS=1
    2. Duplicate OpenMP library conflicts: Set KMP_DUPLICATE_LIB_OK=TRUE

    These settings are only applied on macOS ARM64 and only if the user hasn't
    already set them explicitly.
    """
    system = _platform.system()
    machine = _platform.machine()

    # Detect macOS with Apple Silicon (ARM64)
    is_macos_arm64 = (system == "Darwin" and machine == "arm64")

    if is_macos_arm64:
        # Fix OpenMP crashes on Apple Silicon
        # Only set if not already configured by user
        if "OMP_NUM_THREADS" not in _os.environ:
            _os.environ["OMP_NUM_THREADS"] = "1"

        # Fix duplicate OpenMP library conflicts
        # This occurs when multiple libraries (e.g., numpy, tessera) link different
        # versions of libomp
        if "KMP_DUPLICATE_LIB_OK" not in _os.environ:
            _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

__all__ = [
    # Submodules
    "geom",
    "math",
    "density",
    "io",
    "render",
    "cosmo",
    "halo",
    "origami",
    "stats",
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
