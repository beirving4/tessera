#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "render/box.h"
#include "render/manager.h"
#include "render/hist.h"

namespace py = pybind11;
using namespace tessera::render;
using namespace tessera;

void bind_render(py::module& m) {
    // BoxConfig
    py::class_<BoxConfig>(m, "BoxConfig", "Configuration for a rendering box")
        .def(py::init<>())
        .def_readwrite("x", &BoxConfig::x, "X origin")
        .def_readwrite("y", &BoxConfig::y, "Y origin")
        .def_readwrite("z", &BoxConfig::z, "Z origin")
        .def_readwrite("x_width", &BoxConfig::x_width, "X width")
        .def_readwrite("y_width", &BoxConfig::y_width, "Y width")
        .def_readwrite("z_width", &BoxConfig::z_width, "Z width")
        .def_readwrite("projection_axis", &BoxConfig::projection_axis,
                       "Projection axis ('X', 'Y', 'Z', or empty for 3D)")
        .def("is_projection", &BoxConfig::is_projection,
             "Check if this is a 2D projection");

    // Box interface
    py::class_<Box, std::unique_ptr<Box>>(m, "Box", "Abstract rendering box")
        .def("cell_span", &Box::cell_span, "Get cell span in each dimension")
        .def("cell_origin", &Box::cell_origin, "Get cell origin in each dimension")
        .def("cell_width", &Box::cell_width, "Get physical width of each cell")
        .def("num_cells", &Box::num_cells, "Get number of cells per dimension")
        .def("points", &Box::points, "Get number of sample points per tetrahedron")
        .def("projection_axis", &Box::projection_axis,
             "Get projection axis (-1, false if 3D)");

    // Box3D
    py::class_<Box3D, Box, std::unique_ptr<Box3D>>(m, "Box3D", "3D rendering box")
        .def(py::init<double, int, int, density::Quantity, const BoxConfig&>(),
             py::arg("box_width"), py::arg("pts"), py::arg("cells"),
             py::arg("quantity"), py::arg("config"),
             "Create a 3D rendering box");

    // Box2D
    py::class_<Box2D, Box, std::unique_ptr<Box2D>>(m, "Box2D", "2D projected rendering box")
        .def(py::init<double, int, int, density::Quantity, const BoxConfig&>(),
             py::arg("box_width"), py::arg("pts"), py::arg("cells"),
             py::arg("quantity"), py::arg("config"),
             "Create a 2D projected rendering box");

    // Factory function
    m.def("create_box", &create_box,
          py::arg("box_width"), py::arg("pts"), py::arg("cells"),
          py::arg("quantity"), py::arg("config"),
          "Create a rendering box (2D or 3D based on config)");

    // Utility functions
    m.def("int_floor", &int_floor, py::arg("x"),
          "Floor function that handles negative values correctly");
    m.def("bound", &bound, py::arg("x"), py::arg("cells"),
          "Apply periodic boundary to an index");

    // =========================================================================
    // Histogram module
    // =========================================================================

    // GridLocation
    py::class_<GridLocation>(m, "GridLocation", "Spatial information about a grid")
        .def(py::init<>())
        .def_readwrite("origin", &GridLocation::origin, "Origin position")
        .def_readwrite("span", &GridLocation::span, "Span in each dimension")
        .def_readwrite("pixel_origin", &GridLocation::pixel_origin, "Pixel origin")
        .def_readwrite("pixel_span", &GridLocation::pixel_span, "Pixel span")
        .def_readwrite("pixel_width", &GridLocation::pixel_width, "Width of each pixel");

    // CosmoInfo
    py::class_<CosmoInfo>(m, "CosmoInfo", "Cosmological parameters")
        .def(py::init<>())
        .def_readwrite("redshift", &CosmoInfo::redshift)
        .def_readwrite("scale_factor", &CosmoInfo::scale_factor)
        .def_readwrite("omega_m", &CosmoInfo::omega_m)
        .def_readwrite("omega_l", &CosmoInfo::omega_l)
        .def_readwrite("hubble", &CosmoInfo::hubble)
        .def_readwrite("rho_mean", &CosmoInfo::rho_mean)
        .def_readwrite("rho_critical", &CosmoInfo::rho_critical)
        .def_readwrite("box_width", &CosmoInfo::box_width);

    // RenderInfo
    py::class_<RenderInfo>(m, "RenderInfo", "Rendering parameters")
        .def(py::init<>())
        .def_readwrite("particles", &RenderInfo::particles)
        .def_readwrite("total_pixels", &RenderInfo::total_pixels)
        .def_readwrite("subsample_length", &RenderInfo::subsample_length)
        .def_readwrite("min_projection_depth", &RenderInfo::min_projection_depth)
        .def_readwrite("projection_axis", &RenderInfo::projection_axis);

    // TypeInfo
    py::class_<TypeInfo>(m, "TypeInfo", "Grid type information")
        .def(py::init<>())
        .def_readwrite("header_size", &TypeInfo::header_size)
        .def_readwrite("grid_type", &TypeInfo::grid_type)
        .def_readwrite("is_vector_grid", &TypeInfo::is_vector_grid);

    // GridHeader
    py::class_<GridHeader>(m, "GridHeader", "Metadata for a density grid file")
        .def(py::init<>())
        .def_readwrite("endianness_version", &GridHeader::endianness_version)
        .def_readwrite("type", &GridHeader::type)
        .def_readwrite("cosmo", &GridHeader::cosmo)
        .def_readwrite("render", &GridHeader::render)
        .def_readwrite("loc", &GridHeader::loc)
        .def_readwrite("vel", &GridHeader::vel);

    // Grid I/O functions
    m.def("read_grid_header", &read_grid_header, py::arg("filename"),
          "Read a GridHeader from a binary file");
    m.def("read_grid", &read_grid, py::arg("filename"),
          "Read a scalar grid from a binary file");

    // HistInfo
    py::class_<HistInfo>(m, "HistInfo", "Histogram configuration parameters")
        .def(py::init<>())
        .def_readwrite("min", &HistInfo::min, "Minimum value for histogram range")
        .def_readwrite("max", &HistInfo::max, "Maximum value for histogram range")
        .def_readwrite("bins", &HistInfo::bins, "Number of histogram bins")
        .def_readwrite("scale", &HistInfo::scale, "Scale type: 'linear' or 'log'")
        .def("is_log", &HistInfo::is_log, "Check if using logarithmic scale");

    // HistBox
    py::class_<HistBox>(m, "HistBox", "Spatial region with associated histogram data")
        .def(py::init<>())
        .def_readwrite("origin", &HistBox::origin, "Origin position")
        .def_readwrite("span", &HistBox::span, "Span in each dimension")
        .def_readwrite("centers", &HistBox::centers, "Bin centers")
        .def_readwrite("counts", &HistBox::counts, "Histogram counts")
        .def_static("from_config", &HistBox::from_config,
                    py::arg("x"), py::arg("y"), py::arg("z"),
                    py::arg("x_width"), py::arg("y_width"), py::arg("z_width"),
                    py::arg("bins"),
                    "Create a HistBox from configuration parameters")
        .def("contains", &HistBox::contains, py::arg("v"), py::arg("L"),
             "Check if point is contained within box (with periodic boundaries)");

    // HistManager
    py::class_<HistManager>(m, "HistManager",
        "Manager for parallel histogram computation from phase sheet data")
        .def(py::init<const std::vector<std::string>&, std::vector<HistBox>&,
                      int, const std::string&, const std::string&>(),
             py::arg("files"), py::arg("boxes"), py::arg("points"),
             py::arg("quantity"), py::arg("grid_file"),
             "Create a histogram manager")
        .def("subsample", &HistManager::subsample, py::arg("skip"),
             "Set subsampling factor (must be power of 2)")
        .def("hist", &HistManager::hist, py::arg("info"),
             "Compute histograms for all boxes")
        .def("hist_from_file", &HistManager::hist_from_file,
             py::arg("file"), py::arg("info"),
             "Process a single file and update histograms")
        .def("num_workers", &HistManager::num_workers,
             "Get number of worker threads")
        .def("boxes", py::overload_cast<>(&HistManager::boxes),
             py::return_value_policy::reference_internal,
             "Get the histogram boxes (with computed results)");

    m.doc() = R"pbdoc(
        Rendering module for density field computation.
        
        This module provides the high-level interface for:
        - Configuring rendering boxes (BoxConfig)
        - Creating 2D and 3D rendering domains (Box2D, Box3D)
        - Managing the rendering pipeline (Manager)
        - Computing density fields from particle data
        
        Example usage::
        
            from tessera import render, density
            
            # Create a box configuration
            config = render.BoxConfig()
            config.x = 0
            config.y = 0
            config.z = 0
            config.x_width = 100
            config.y_width = 100
            config.z_width = 100
            
            # Create a 3D box
            box = render.create_box(
                box_width=500.0,
                pts=100,
                cells=256,
                quantity=density.Quantity.Density,
                config=config
            )
    )pbdoc";
}
