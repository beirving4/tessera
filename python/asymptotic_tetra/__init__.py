"""
AsymptoticTetra - High-performance phase-space tessellation for cosmological simulations.

This package provides Python bindings to a C++ library for computing density fields
from N-body simulation data using tetrahedron-based Monte Carlo integration.

Example usage:
    >>> import asymptotic_tetra as at
    >>> 
    >>> # Create a random number generator
    >>> gen = at.math.Generator.new_time_seed()
    >>> print(gen.uniform(0, 1))
    >>> 
    >>> # Work with geometry primitives
    >>> v1 = at.geom.Vec3f(1.0, 2.0, 3.0)
    >>> v2 = at.geom.Vec3f(4.0, 5.0, 6.0)
    >>> print(v1.dot(v2))
    >>> 
    >>> # Create tetrahedra
    >>> tet = at.geom.Tetra(
    ...     at.geom.Vec3f(0, 0, 0),
    ...     at.geom.Vec3f(1, 0, 0),
    ...     at.geom.Vec3f(0, 1, 0),
    ...     at.geom.Vec3f(0, 0, 1)
    ... )
    >>> print(tet.volume())

Modules:
    geom - Geometry primitives (Vec3f, Tetra, Grid, CellBounds)
    math - Random number generation (Generator, TauswortheGenerator)
    density - Density field computation (Quantity, Buffer)
    io - File I/O for simulation formats (GadgetHeader, SheetHeader)
    render - Rendering interface (coming soon)
"""

try:
    from ._asymptotic_tetra import (
        geom,
        math,
        density,
        io,
        render,
    )
    __version__ = "1.0.0"

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
except ImportError as e:
    # Fallback if native module not available
    import warnings
    warnings.warn(f"Native module not available: {e}. Some functionality will be limited.")
    __version__ = "1.0.0"
    geom = None
    math = None
    density = None
    io = None
    render = None

__all__ = [
    # Submodules
    "geom",
    "math", 
    "density",
    "io",
    "render",
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
    "create_buffer",
    # I/O
    "CosmologyHeader",
    "CatalogHeader",
    "SheetHeader",
    "GadgetHeader",
    "ByteOrder",
    # Meta
    "__version__",
]
