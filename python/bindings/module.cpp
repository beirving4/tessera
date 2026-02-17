#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for binding functions
void bind_geom(py::module& m);
void bind_math(py::module& m);
void bind_density(py::module& m);
void bind_tetra_density(py::module& m);
void bind_sph(py::module& m);
void bind_io(py::module& m);
void bind_render(py::module& m);
void bind_cosmo(py::module& m);
void bind_halo(py::module& m);
void bind_stats(py::module_& m);
void bind_origami(py::module& m);
void bind_voronoi_density(py::module& m);
void bind_cic_density(py::module& m);

PYBIND11_MODULE(_tessera, m) {
    m.doc() = R"pbdoc(
        tessera - C++ library for phase-space tessellation

        This module provides high-performance routines for computing
        density fields from cosmological N-body simulations using
        tetrahedron-based Monte Carlo integration.

        Submodules:
            geom - Geometry primitives (Vec3f, Tetra, Grid, CellBounds)
            math - Random number generation
            density - Density field computation and buffers
            io - File I/O for Gadget and sheet formats
            render - High-level rendering interface
            cosmo - Cosmological calculations
            halo - Halo finding and analysis
            stats - Statistical functions (histograms, PDFs)
            origami - ORIGAMI morphology classification (cosmic web)
    )pbdoc";

    // Create submodules
    auto geom_m = m.def_submodule("geom", "Geometry primitives");
    auto math_m = m.def_submodule("math", "Random number generation");
    auto density_m = m.def_submodule("density", "Density field computation");
    auto io_m = m.def_submodule("io", "File I/O");
    auto render_m = m.def_submodule("render", "Rendering interface");
    auto cosmo_m = m.def_submodule("cosmo", "Cosmological calculations");
    auto halo_m = m.def_submodule("halo", "Halo finding and analysis");

    // Bind all submodules
    bind_geom(geom_m);
    bind_math(math_m);
    bind_density(density_m);
    bind_tetra_density(density_m);  // Add tessellation functions to density module
    bind_sph(density_m);  // Add SPH rendering to density module
    bind_voronoi_density(density_m);  // Add Voronoi (VTFE) density to density module
    bind_cic_density(density_m);  // Add CIC (Cloud-in-Cell) density to density module
    bind_io(io_m);
    bind_render(render_m);
    bind_cosmo(cosmo_m);
    bind_halo(halo_m);
    bind_stats(m);  // Creates its own submodule
    bind_origami(m);  // Creates origami submodule

    // Version info
    m.attr("__version__") = "1.0.0";
}
