#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "io/headers.h"

namespace py = pybind11;
using namespace asymptotic_tetra::io;

void bind_io(py::module& m) {
    // ByteOrder enum
    py::enum_<ByteOrder>(m, "ByteOrder", "Byte order for binary I/O")
        .value("Little", ByteOrder::Little)
        .value("Big", ByteOrder::Big);

    // CosmologyHeader
    py::class_<CosmologyHeader>(m, "CosmologyHeader", "Cosmological parameters")
        .def(py::init<>())
        .def_readwrite("z", &CosmologyHeader::z, "Redshift")
        .def_readwrite("omega_m", &CosmologyHeader::omega_m, "Matter density")
        .def_readwrite("omega_l", &CosmologyHeader::omega_l, "Dark energy density")
        .def_readwrite("h100", &CosmologyHeader::h100, "Hubble parameter (H0/100)")
        .def("__repr__", [](const CosmologyHeader& h) {
            return "CosmologyHeader(z=" + std::to_string(h.z) + 
                   ", omega_m=" + std::to_string(h.omega_m) +
                   ", omega_l=" + std::to_string(h.omega_l) + ")";
        });

    // CatalogHeader
    py::class_<CatalogHeader>(m, "CatalogHeader", "Particle catalog metadata")
        .def(py::init<>())
        .def_readwrite("cosmo", &CatalogHeader::cosmo)
        .def_readwrite("mass", &CatalogHeader::mass)
        .def_readwrite("count", &CatalogHeader::count)
        .def_readwrite("total_count", &CatalogHeader::total_count)
        .def_readwrite("count_width", &CatalogHeader::count_width)
        .def_readwrite("total_width", &CatalogHeader::total_width);

    // SheetHeader
    py::class_<SheetHeader>(m, "SheetHeader", "Phase sheet file header")
        .def(py::init<>())
        .def_readwrite("cosmo", &SheetHeader::cosmo)
        .def_readwrite("count", &SheetHeader::count)
        .def_readwrite("count_width", &SheetHeader::count_width)
        .def_readwrite("segment_width", &SheetHeader::segment_width)
        .def_readwrite("grid_width", &SheetHeader::grid_width)
        .def_readwrite("grid_count", &SheetHeader::grid_count)
        .def_readwrite("idx", &SheetHeader::idx)
        .def_readwrite("cells", &SheetHeader::cells)
        .def_readwrite("mass", &SheetHeader::mass)
        .def_readwrite("total_width", &SheetHeader::total_width)
        .def_readwrite("origin", &SheetHeader::origin)
        .def_readwrite("width", &SheetHeader::width)
        .def("cell_bounds", &SheetHeader::cell_bounds, py::arg("cells"));

    // GadgetHeader
    py::class_<GadgetHeader>(m, "GadgetHeader", "Gadget-2 file header")
        .def(py::init<>())
        .def_readonly("box_size", &GadgetHeader::box_size)
        .def_readonly("redshift", &GadgetHeader::redshift)
        .def_readonly("omega0", &GadgetHeader::omega0)
        .def_readonly("omega_lambda", &GadgetHeader::omega_lambda)
        .def_readonly("hubble_param", &GadgetHeader::hubble_param)
        .def("standardize", &GadgetHeader::standardize)
        .def("wrap_distance", &GadgetHeader::wrap_distance, py::arg("x"));

    // Utility functions
    m.def("is_system_little_endian", &is_system_little_endian,
          "Check if system uses little-endian byte order");
    m.def("system_byte_order", &system_byte_order,
          "Get system's native byte order");
}
