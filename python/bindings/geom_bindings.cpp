#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "geom/vec.h"
#include "geom/grid.h"
#include "geom/tetra.h"

namespace py = pybind11;
using namespace tessera::geom;

void bind_geom(py::module& m) {
    // Vec3f binding
    py::class_<Vec3f>(m, "Vec3f", "3D vector with single-precision floats")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property("x", 
            [](const Vec3f& v) { return v[0]; },
            [](Vec3f& v, float val) { v[0] = val; })
        .def_property("y",
            [](const Vec3f& v) { return v[1]; },
            [](Vec3f& v, float val) { v[1] = val; })
        .def_property("z",
            [](const Vec3f& v) { return v[2]; },
            [](Vec3f& v, float val) { v[2] = val; })
        .def("__getitem__", [](const Vec3f& v, size_t i) { return v[i]; })
        .def("__setitem__", [](Vec3f& v, size_t i, float val) { v[i] = val; })
        .def("scale", &Vec3f::scale, py::arg("k"))
        .def("mod", &Vec3f::mod, py::arg("width"))
        .def("add", &Vec3f::add, py::arg("other"))
        .def("sub", &Vec3f::sub, py::arg("other"), py::arg("width"))
        .def("dot", &Vec3f::dot, py::arg("other"))
        .def("cross", &Vec3f::cross, py::arg("other"))
        .def("norm", &Vec3f::norm)
        .def("__add__", &Vec3f::add)
        .def("__sub__", [](const Vec3f& a, const Vec3f& b) { return a - b; })
        .def("__mul__", [](const Vec3f& v, float s) { return v * s; })
        .def("__rmul__", [](const Vec3f& v, float s) { return s * v; })
        .def("__repr__", [](const Vec3f& v) {
            return "Vec3f(" + std::to_string(v[0]) + ", " + 
                   std::to_string(v[1]) + ", " + std::to_string(v[2]) + ")";
        })
        .def("to_numpy", [](const Vec3f& v) {
            return py::array_t<float>({3}, {sizeof(float)}, v.data.data());
        });

    // CellBounds binding
    py::class_<CellBounds>(m, "CellBounds", "Axis-aligned cell bounds")
        .def(py::init<>())
        .def(py::init<const std::array<int, 3>&, const std::array<int, 3>&>())
        .def_readwrite("origin", &CellBounds::origin)
        .def_readwrite("width", &CellBounds::width)
        .def("intersect", &CellBounds::intersect, py::arg("other"), py::arg("cells"))
        .def("intersect_unbounded", &CellBounds::intersect_unbounded, py::arg("other"))
        .def("__eq__", &CellBounds::operator==)
        .def("__repr__", [](const CellBounds& cb) {
            return "CellBounds(origin=[" + 
                   std::to_string(cb.origin[0]) + ", " +
                   std::to_string(cb.origin[1]) + ", " +
                   std::to_string(cb.origin[2]) + "], width=[" +
                   std::to_string(cb.width[0]) + ", " +
                   std::to_string(cb.width[1]) + ", " +
                   std::to_string(cb.width[2]) + "])";
        });

    // Grid binding
    py::class_<Grid>(m, "Grid", "3D grid indexing")
        .def(py::init<>())
        .def(py::init<const std::array<int, 3>&, const std::array<int, 3>&>())
        .def_readonly("bounds", &Grid::bounds)
        .def_readonly("length", &Grid::length)
        .def_readonly("area", &Grid::area)
        .def_readonly("volume", &Grid::volume)
        .def("idx", &Grid::idx, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("coords", &Grid::coords, py::arg("idx"))
        .def("bounds_check", &Grid::bounds_check, py::arg("x"), py::arg("y"), py::arg("z"));

    // GridLocation binding
    py::class_<GridLocation, Grid>(m, "GridLocation", "Grid with physical location")
        .def(py::init<>())
        .def(py::init<const std::array<int, 3>&, const std::array<int, 3>&, double, int>())
        .def_readwrite("cells", &GridLocation::cells)
        .def_readwrite("box_width", &GridLocation::box_width);

    // TetraIdxs binding
    py::class_<TetraIdxs>(m, "TetraIdxs", "Tetrahedron corner indices")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int>())
        .def("__getitem__", [](const TetraIdxs& t, size_t i) { return t[i]; })
        .def("__setitem__", [](TetraIdxs& t, size_t i, int64_t val) { t[i] = val; })
        .def("init", &TetraIdxs::init, py::arg("idx"), py::arg("count_width"),
             py::arg("skip"), py::arg("dir"), py::return_value_policy::reference)
        .def("init_cartesian", &TetraIdxs::init_cartesian);

    // Tetra binding
    py::class_<Tetra>(m, "Tetra", "Tetrahedron with volume and sampling")
        .def(py::init<>())
        .def(py::init<const Vec3f&, const Vec3f&, const Vec3f&, const Vec3f&>())
        .def_readwrite("corners", &Tetra::corners)
        .def("init", &Tetra::init, py::return_value_policy::reference)
        .def("volume", &Tetra::volume)
        .def("contains", &Tetra::contains, py::arg("v"))
        .def("barycenter", &Tetra::barycenter, py::return_value_policy::reference)
        .def("cell_bounds", &Tetra::cell_bounds, py::arg("cell_width"))
        .def("distribute_tetra", &Tetra::distribute_tetra, py::arg("pts"), py::arg("out"));

    // Free functions
    m.def("distribute_unit", &distribute_unit, py::arg("vec_buf"),
          "Distribute points from unit cube to unit tetrahedron");
    
    // Constants
    m.attr("TETRA_DIR_COUNT") = TETRA_DIR_COUNT;
    m.attr("TETRA_CENTERED_COUNT") = TETRA_CENTERED_COUNT;
}
