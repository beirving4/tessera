#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "density/quantity.h"
#include "density/buffer.h"

namespace py = pybind11;
using namespace asymptotic_tetra::density;

void bind_density(py::module& m) {
    // Quantity enum
    py::enum_<Quantity>(m, "Quantity", "Physical quantity types")
        .value("Density", Quantity::Density)
        .value("DensityGradient", Quantity::DensityGradient)
        .value("Velocity", Quantity::Velocity)
        .value("VelocityDivergence", Quantity::VelocityDivergence)
        .value("VelocityCurl", Quantity::VelocityCurl);

    // Quantity utility functions
    m.def("quantity_to_string", &quantity_to_string, py::arg("q"));
    m.def("quantity_from_string", &quantity_from_string, py::arg("str"));
    m.def("requires_velocity", &requires_velocity, py::arg("q"));
    m.def("can_project", &can_project, py::arg("q"));
    m.def("is_vector", &is_vector, py::arg("q"));

    // Buffer abstract interface (exposed for type hints)
    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer", "Density field buffer")
        .def("length", &Buffer::length)
        .def("clear", &Buffer::clear)
        .def("quantity", &Buffer::quantity);

    // DensityBuffer
    py::class_<DensityBuffer, Buffer, std::shared_ptr<DensityBuffer>>(m, "DensityBuffer")
        .def(py::init<size_t>(), py::arg("len"))
        .def("finalized_scalar_buffer", [](DensityBuffer& buf) {
            auto data = buf.finalized_scalar_buffer();
            return py::array_t<float>(data.size(), data.data());
        });

    // Factory function
    m.def("create_buffer", [](Quantity q, size_t len, size_t wlen) {
        return create_buffer(q, len, wlen, nullptr);
    }, py::arg("quantity"), py::arg("len"), py::arg("wlen") = 0,
       "Create a buffer for the given quantity type");
}
