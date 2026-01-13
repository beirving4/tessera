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
    # Meta
    "__version__",
]
