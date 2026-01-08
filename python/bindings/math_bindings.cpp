#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "math/rand.h"

namespace py = pybind11;
using namespace asymptotic_tetra::math;

void bind_math(py::module& m) {
    // GeneratorType enum
    py::enum_<GeneratorType>(m, "GeneratorType", "Random number generator types")
        .value("Xorshift", GeneratorType::Xorshift)
        .value("Golang", GeneratorType::Golang)
        .value("Tausworthe", GeneratorType::Tausworthe)
        .value("Default", GeneratorType::Default);

    // Generator class
    py::class_<Generator>(m, "Generator", "Random number generator")
        .def(py::init<GeneratorType, uint64_t>(), 
             py::arg("type") = GeneratorType::Default,
             py::arg("seed") = 42)
        .def_static("new_time_seed", &Generator::new_time_seed,
                    py::arg("type") = GeneratorType::Default,
                    "Create generator with time-based seed")
        .def("uniform_int", &Generator::uniform_int, 
             py::arg("low"), py::arg("high"),
             "Generate uniform random integer in [low, high)")
        .def("uniform", &Generator::uniform, 
             py::arg("low") = 0.0, py::arg("high") = 1.0,
             "Generate uniform random double in [low, high)")
        .def("uniform_array", [](Generator& gen, double low, double high, size_t n) {
            std::vector<double> result(n);
            gen.uniform_at(low, high, result);
            return py::array_t<double>(n, result.data());
        }, py::arg("low") = 0.0, py::arg("high") = 1.0, py::arg("n") = 1,
           "Generate array of uniform random doubles")
        .def("gaussian", &Generator::gaussian,
             "Generate standard normal random variable");
}
