/**
 * @file cic_density_bindings.cpp
 * @brief Python bindings for CIC (Cloud-in-Cell) density computation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "density/cic_density.h"

namespace py = pybind11;

void bind_cic_density(py::module& m) {
    using namespace tessera::density;

    // CICDensityConfig
    py::class_<CICDensityConfig>(m, "CICDensityConfig",
        R"pbdoc(
        Configuration for CIC (Cloud-in-Cell) density computation.

        CIC deposits particle mass onto a regular grid using trilinear weights,
        then optionally samples the grid back at particle positions. The grid
        resolution sets the smoothing scale (cell_width = box_size / output_cells).

        Reference: Klypin et al. 2018, MNRAS 478, 4602

        Attributes
        ----------
        box_size : float
            Physical size of the periodic simulation box.
        output_cells : int
            Grid resolution (cells per dimension). Default: 256.
        particle_mass : float
            Mass per particle. Default: 1.0.
        n_threads : int
            Number of OpenMP threads. 0 = auto-detect (capped at 8). Default: 0.
        periodic : bool
            Use periodic boundary conditions. Default: True.
        sample_at_particles : bool
            Sample grid density at particle positions via trilinear interpolation.
            Default: True.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](double box_size, int output_cells, double particle_mass,
                         int n_threads, bool periodic, bool sample_at_particles) {
            CICDensityConfig cfg;
            cfg.box_size = box_size;
            cfg.output_cells = output_cells;
            cfg.particle_mass = particle_mass;
            cfg.n_threads = n_threads;
            cfg.periodic = periodic;
            cfg.sample_at_particles = sample_at_particles;
            return cfg;
        }),
             py::arg("box_size"),
             py::arg("output_cells") = 256,
             py::arg("particle_mass") = 1.0,
             py::arg("n_threads") = 0,
             py::arg("periodic") = true,
             py::arg("sample_at_particles") = true)
        .def_readwrite("box_size", &CICDensityConfig::box_size)
        .def_readwrite("output_cells", &CICDensityConfig::output_cells)
        .def_readwrite("particle_mass", &CICDensityConfig::particle_mass)
        .def_readwrite("n_threads", &CICDensityConfig::n_threads)
        .def_readwrite("periodic", &CICDensityConfig::periodic)
        .def_readwrite("sample_at_particles", &CICDensityConfig::sample_at_particles);

    // CICDensityResult
    py::class_<CICDensityResult>(m, "CICDensityResult",
        R"pbdoc(
        Result of CIC density computation.

        Attributes
        ----------
        grid_density : numpy.ndarray
            3D density grid, shape (output_cells, output_cells, output_cells).
        particle_density : numpy.ndarray
            Per-particle density from grid interpolation, shape (N,).
            Empty if sample_at_particles was False.
        mean_density : float
            Mean density = total_mass / box_volume.
        cell_width : float
            Physical cell width = box_size / output_cells.
        output_cells : int
            Grid resolution used.
        total_time_ms : float
            Total computation wall time in milliseconds.
        grid_time_ms : float
            Grid deposition time in milliseconds.
        sample_time_ms : float
            Particle sampling time in milliseconds.
        n_particles : int
            Number of particles processed.
        )pbdoc")
        .def_readonly("mean_density", &CICDensityResult::mean_density)
        .def_readonly("cell_width", &CICDensityResult::cell_width)
        .def_readonly("output_cells", &CICDensityResult::output_cells)
        .def_readonly("total_time_ms", &CICDensityResult::total_time_ms)
        .def_readonly("grid_time_ms", &CICDensityResult::grid_time_ms)
        .def_readonly("sample_time_ms", &CICDensityResult::sample_time_ms)
        .def_readonly("n_particles", &CICDensityResult::n_particles)
        .def_property_readonly("grid_density", [](const CICDensityResult& r) {
            if (r.grid_density.empty()) {
                return py::array_t<double>();
            }
            int c = r.output_cells;
            return py::array_t<double>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(double) * c * c),
                 static_cast<py::ssize_t>(sizeof(double) * c),
                 static_cast<py::ssize_t>(sizeof(double))},
                r.grid_density.data()
            );
        }, "3D density grid, shape (cells, cells, cells)")
        .def_property_readonly("particle_density", [](const CICDensityResult& r) {
            if (r.particle_density.empty()) {
                return py::array_t<double>();
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.particle_density.size())},
                {sizeof(double)},
                r.particle_density.data()
            );
        }, "Per-particle density as numpy array (N,)");

    // compute_cic_density
    m.def("compute_cic_density",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           double box_size,
           int output_cells,
           double particle_mass,
           int n_threads,
           bool periodic,
           bool sample_at_particles) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            int64_t n_particles = buf.shape[0];

            CICDensityConfig cfg;
            cfg.box_size = box_size;
            cfg.output_cells = output_cells;
            cfg.particle_mass = particle_mass;
            cfg.n_threads = n_threads;
            cfg.periodic = periodic;
            cfg.sample_at_particles = sample_at_particles;

            const double* data = static_cast<const double*>(buf.ptr);
            return compute_cic_density(data, n_particles, cfg);
        },
        py::arg("positions"),
        py::arg("box_size"),
        py::arg("output_cells") = 256,
        py::arg("particle_mass") = 1.0,
        py::arg("n_threads") = 0,
        py::arg("periodic") = true,
        py::arg("sample_at_particles") = true,
        R"pbdoc(
        Compute CIC (Cloud-in-Cell) density on a regular grid.

        Deposits particle mass onto a grid using trilinear interpolation weights,
        then optionally samples the grid density at particle positions. This is
        the standard Eulerian density estimator used in Klypin et al. (2018).

        The grid resolution sets the smoothing scale: cell_width = box_size / output_cells.
        CIC is the fastest density method (O(N) with constant factor ~8 per particle).

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Eulerian space (any order, no sorting needed).
        box_size : float
            Physical size of the periodic simulation box.
        output_cells : int, optional
            Grid resolution (cells per dimension). Default: 256.
        particle_mass : float, optional
            Mass per particle. Default: 1.0.
        n_threads : int, optional
            Number of OpenMP threads. 0 = auto (capped at 8). Default: 0.
        periodic : bool, optional
            Use periodic boundary conditions. Default: True.
        sample_at_particles : bool, optional
            Sample grid density at particle positions. Default: True.

        Returns
        -------
        CICDensityResult
            Result with 3D density grid and optional per-particle density.

        Examples
        --------
        >>> result = ts.density.compute_cic_density(positions, box_size=256.0)
        >>> overdensity = result.particle_density / result.mean_density  # 1 + delta
        >>> print(f"Grid shape: {result.grid_density.shape}")
        )pbdoc");
}
