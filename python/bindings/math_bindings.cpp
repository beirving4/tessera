#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "math/rand.h"
#include "math/sobol.h"
#include "math/mat.h"
#include "math/interpolate.h"
#include "math/sort.h"
#include "math/calc.h"

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
    
    // ========================================================================
    // Sobol Sequences
    // ========================================================================
    py::class_<SobolSequence>(m, "SobolSequence",
        R"pbdoc(
        Sobol quasi-random sequence generator.
        
        Sobol sequences provide better coverage than pseudo-random numbers,
        making them ideal for Monte Carlo integration.
        
        Example:
            >>> seq = at.math.SobolSequence()
            >>> points = [seq.next(3) for _ in range(100)]  # 100 3D points
        )pbdoc")
        .def(py::init<>())
        .def("init", &SobolSequence::init, "Initialize/reset the sequence")
        .def("reset", &SobolSequence::reset, "Reset sequence to beginning")
        .def("next", &SobolSequence::next, py::arg("dim"),
             "Get next point in sequence with given dimensionality")
        .def("next_at", [](SobolSequence& seq, std::vector<double>& target) {
            seq.next_at(target);
        }, py::arg("target"), "Get next point, storing in-place")
        .def("sequence_number", &SobolSequence::sequence_number,
             "Get current sequence number")
        .def_static("max_sequence", &SobolSequence::max_sequence,
                    "Get maximum sequence length")
        .def_static("max_dim", &SobolSequence::max_dim,
                    "Get maximum supported dimensions");
    
    // ========================================================================
    // Matrix Operations
    // ========================================================================
    py::class_<Matrix>(m, "Matrix",
        "Matrix class for linear algebra operations")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def(py::init<const std::vector<double>&, int, int>(),
             py::arg("vals"), py::arg("width"), py::arg("height"))
        .def_readwrite("vals", &Matrix::vals)
        .def_readwrite("width", &Matrix::width)
        .def_readwrite("height", &Matrix::height)
        .def("at", py::overload_cast<int, int>(&Matrix::at),
             py::arg("row"), py::arg("col"), "Access element")
        .def("mult", &Matrix::mult, py::arg("other"), "Multiply matrices")
        .def("transpose", &Matrix::transpose, "Transpose matrix")
        .def_static("identity", &Matrix::identity, py::arg("n"),
                    "Create identity matrix")
        .def_static("mult_vec", [](const Matrix& m, const std::vector<double>& v) {
            std::vector<double> out(m.height);
            Matrix::mult_vec(m, v, out);
            return out;
        }, py::arg("m"), py::arg("v"),
           "Multiply matrix by vector: return M * v")
        .def_static("vec_mult", [](const std::vector<double>& v, const Matrix& m) {
            std::vector<double> out(m.width);
            Matrix::vec_mult(v, m, out);
            return out;
        }, py::arg("v"), py::arg("m"),
           "Multiply vector by matrix: return v * M");
    
    py::class_<LUFactors>(m, "LUFactors",
        "LU decomposition for matrix operations")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("n"))
        .def("factorize", &LUFactors::factorize, py::arg("m"),
             "Compute LU decomposition")
        .def("solve_vector", &LUFactors::solve_vector, py::arg("b"),
             "Solve M * x = b for x")
        .def("invert", &LUFactors::invert, "Compute inverse matrix")
        .def("determinant", &LUFactors::determinant, "Compute determinant");
    
    m.def("lu_decompose", &lu_decompose, py::arg("m"),
          "Compute LU decomposition of matrix");
    m.def("invert", &invert, py::arg("m"), "Invert a matrix");
    m.def("determinant", &determinant, py::arg("m"), "Compute determinant");
    m.def("solve", &solve, py::arg("m"), py::arg("b"),
          "Solve M * x = b for x");
    
    // ========================================================================
    // Interpolation
    // ========================================================================
    m.def("search", &search, py::arg("xs"), py::arg("x"),
          "Binary search for interval containing x");
    
    // 1D Interpolators
    py::class_<Linear>(m, "Linear", "1D linear interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"))
        .def_readwrite("xs", &Linear::xs)
        .def_readwrite("ys", &Linear::ys)
        .def("eval", &Linear::eval, py::arg("x"), "Evaluate at x")
        .def("eval_all", &Linear::eval_all, py::arg("xs"),
             "Evaluate at multiple points");
    
    py::class_<Spline>(m, "Spline", "Cubic spline interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"))
        .def_readwrite("xs", &Spline::xs)
        .def_readwrite("ys", &Spline::ys)
        .def("eval", &Spline::eval, py::arg("x"), "Evaluate at x")
        .def("eval_all", &Spline::eval_all, py::arg("xs"),
             "Evaluate at multiple points");
    
    // 2D Interpolators
    py::class_<BiLinear>(m, "BiLinear", "2D bilinear interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"), py::arg("vals"))
        .def("eval", &BiLinear::eval, py::arg("x"), py::arg("y"),
             "Evaluate at (x, y)")
        .def("eval_all", &BiLinear::eval_all, py::arg("xs"), py::arg("ys"),
             "Evaluate at multiple points");
    
    py::class_<BiCubic>(m, "BiCubic", "2D bicubic interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"), py::arg("vals"))
        .def("eval", &BiCubic::eval, py::arg("x"), py::arg("y"),
             "Evaluate at (x, y)")
        .def("eval_all", &BiCubic::eval_all, py::arg("xs"), py::arg("ys"),
             "Evaluate at multiple points");
    
    // 3D Interpolators
    py::class_<TriLinear>(m, "TriLinear", "3D trilinear interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&, const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("vals"))
        .def("eval", &TriLinear::eval, py::arg("x"), py::arg("y"), py::arg("z"),
             "Evaluate at (x, y, z)")
        .def("eval_all", &TriLinear::eval_all,
             py::arg("xs"), py::arg("ys"), py::arg("zs"),
             "Evaluate at multiple points");
    
    py::class_<TriCubic>(m, "TriCubic", "3D tricubic interpolation")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&, const std::vector<double>&>(),
             py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("vals"))
        .def("eval", &TriCubic::eval, py::arg("x"), py::arg("y"), py::arg("z"),
             "Evaluate at (x, y, z)")
        .def("eval_all", &TriCubic::eval_all,
             py::arg("xs"), py::arg("ys"), py::arg("zs"),
             "Evaluate at multiple points");
    
    // ========================================================================
    // Sort Utilities
    // ========================================================================
    m.def("reverse", [](std::vector<double> xs) {
        reverse(xs);
        return xs;
    }, py::arg("xs"), "Reverse a vector");
    
    m.def("shell_sort", [](std::vector<double> xs) {
        shell_sort(xs);
        return xs;
    }, py::arg("xs"), "Shell sort a vector");
    
    m.def("quickselect", [](std::vector<double> xs, int k) {
        return quickselect(xs, k);
    }, py::arg("xs"), py::arg("k"),
       "Find k-th smallest element (modifies copy)");
    
    m.def("median", [](std::vector<double> xs) {
        return median(xs);
    }, py::arg("xs"), "Find median (modifies copy)");
    
    m.def("median_copy", &median_copy, py::arg("xs"),
          "Find median (non-modifying)");
    
    m.def("argsort", &argsort, py::arg("xs"),
          "Get indices that would sort the array");
    
    m.def("argsort_descending", &argsort_descending, py::arg("xs"),
          "Get indices for descending sort");
    
    m.def("percentile", [](std::vector<double> xs, double p) {
        return percentile(xs, p);
    }, py::arg("xs"), py::arg("p"),
       "Calculate percentile (0-1 range)");
    
    // ========================================================================
    // Calculus (Derivatives)
    // ========================================================================
    m.def("deriv", &deriv, py::arg("xs"), py::arg("ys"), py::arg("order"),
          R"pbdoc(
          Compute numerical derivative.
          
          Parameters
          ----------
          xs : list[float]
              X coordinates
          ys : list[float]
              Y values
          order : int
              Accuracy order (2 or 4)
              
          Returns
          -------
          list[float]
              Derivative dy/dx at each point
          )pbdoc");
    
    m.def("deriv2", &deriv2, py::arg("xs"), py::arg("ys"), py::arg("order") = 2,
          "Compute second derivative");
    
    m.def("gradient", &gradient, py::arg("ys"), py::arg("dx"),
          "Compute gradient of uniformly spaced data");
}
