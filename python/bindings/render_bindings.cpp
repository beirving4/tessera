#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "render/box.h"
#include "render/manager.h"

namespace py = pybind11;
using namespace asymptotic_tetra::render;
using namespace asymptotic_tetra;

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

    m.doc() = R"pbdoc(
        Rendering module for density field computation.
        
        This module provides the high-level interface for:
        - Configuring rendering boxes (BoxConfig)
        - Creating 2D and 3D rendering domains (Box2D, Box3D)
        - Managing the rendering pipeline (Manager)
        - Computing density fields from particle data
        
        Example usage::
        
            from asymptotic_tetra import render, density
            
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
