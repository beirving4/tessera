/**
 * @file voronoi_density_bindings.cpp
 * @brief Python bindings for Voronoi (VTFE) density computation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef TESSERA_HAS_VORO
#include "density/voronoi_density.h"
#endif

namespace py = pybind11;

void bind_voronoi_density(py::module& m) {
#ifdef TESSERA_HAS_VORO
    using namespace tessera::density;

    // VoronoiDensityConfig
    py::class_<VoronoiDensityConfig>(m, "VoronoiDensityConfig",
        R"pbdoc(
        Configuration for Voronoi (VTFE) density computation.

        The Voronoi Tessellation Field Estimator computes per-particle density
        from Voronoi cell volumes: density_i = mean_density * V_mean / V_i.
        This resolves densities up to 10^4-10^6, matching Falck et al. (2012).

        Attributes
        ----------
        box_size : float
            Physical size of the periodic simulation box.
        n_threads : int
            Number of OpenMP threads. 0 = auto-detect (capped at 8).
        compute_volumes : bool
            If True, return raw Voronoi cell volumes in result. Default: True.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](double box_size, int n_threads, bool compute_volumes) {
            VoronoiDensityConfig cfg;
            cfg.box_size = box_size;
            cfg.n_threads = n_threads;
            cfg.compute_volumes = compute_volumes;
            return cfg;
        }),
             py::arg("box_size"),
             py::arg("n_threads") = 0,
             py::arg("compute_volumes") = true)
        .def_readwrite("box_size", &VoronoiDensityConfig::box_size)
        .def_readwrite("n_threads", &VoronoiDensityConfig::n_threads)
        .def_readwrite("compute_volumes", &VoronoiDensityConfig::compute_volumes);

    // VoronoiDensityResult
    py::class_<VoronoiDensityResult>(m, "VoronoiDensityResult",
        R"pbdoc(
        Result of Voronoi (VTFE) density computation.

        Attributes
        ----------
        density : numpy.ndarray
            Per-particle density, shape (N,).
            density_i = mean_density * V_mean / V_i.
        volumes : numpy.ndarray
            Raw Voronoi cell volumes V_i, shape (N,).
            Empty if compute_volumes was False.
        mean_density : float
            N / box_volume (assuming unit particle mass).
        mean_volume : float
            V_mean = box_volume / N.
        total_time_ms : float
            Computation wall time in milliseconds.
        n_particles : int
            Number of particles processed.
        )pbdoc")
        .def_readonly("mean_density", &VoronoiDensityResult::mean_density)
        .def_readonly("mean_volume", &VoronoiDensityResult::mean_volume)
        .def_readonly("total_time_ms", &VoronoiDensityResult::total_time_ms)
        .def_readonly("n_particles", &VoronoiDensityResult::n_particles)
        .def_property_readonly("density", [](const VoronoiDensityResult& r) {
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.density.size())},
                {sizeof(double)},
                r.density.data()
            );
        }, "Per-particle density as numpy array (N,)")
        .def_property_readonly("volumes", [](const VoronoiDensityResult& r) {
            if (r.volumes.empty()) {
                return py::array_t<double>();
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.volumes.size())},
                {sizeof(double)},
                r.volumes.data()
            );
        }, "Voronoi cell volumes as numpy array (N,)");

    // compute_voronoi_density
    m.def("compute_voronoi_density",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           double box_size,
           int n_threads,
           bool compute_volumes) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            int64_t n_particles = buf.shape[0];

            VoronoiDensityConfig cfg;
            cfg.box_size = box_size;
            cfg.n_threads = n_threads;
            cfg.compute_volumes = compute_volumes;

            const double* data = static_cast<const double*>(buf.ptr);
            return compute_voronoi_density(data, n_particles, cfg);
        },
        py::arg("positions"),
        py::arg("box_size"),
        py::arg("n_threads") = 0,
        py::arg("compute_volumes") = true,
        R"pbdoc(
        Compute per-particle density using Voronoi cell volumes (VTFE).

        Uses voro++ to build a periodic Voronoi tessellation in Eulerian space,
        then computes density from cell volumes:
            density_i = mean_density * V_mean / V_i

        This is the Voronoi Tessellation Field Estimator (VTFE) from Falck et al.
        (2012), resolving overdensities up to 10^4-10^6 for N=256^3.

        Unlike tessellation-based density (compute_particle_density), this does
        NOT require Lagrangian sorting — it works directly on Eulerian positions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Eulerian space (any order).
        box_size : float
            Physical size of the periodic simulation box.
        n_threads : int, optional
            Number of OpenMP threads. 0 = auto (capped at 8). Default: 0.
        compute_volumes : bool, optional
            If True, return raw Voronoi cell volumes. Default: True.

        Returns
        -------
        VoronoiDensityResult
            Result with per-particle densities and optional volumes.

        Examples
        --------
        >>> result = ts.density.compute_voronoi_density(positions, box_size=256.0)
        >>> overdensity = result.density / result.mean_density  # 1 + delta
        >>> print(f"Max overdensity: {overdensity.max():.0f}")
        )pbdoc");

#else
    // Stub when voro++ is not available
    m.def("compute_voronoi_density",
        [](py::object, double, int, bool) -> py::object {
            throw std::runtime_error(
                "Voronoi density not available: tessera was built without voro++ support. "
                "Rebuild with -DBUILD_WITH_VORO=ON.");
        },
        py::arg("positions"),
        py::arg("box_size"),
        py::arg("n_threads") = 0,
        py::arg("compute_volumes") = true,
        "Compute Voronoi density (requires BUILD_WITH_VORO=ON)");
#endif
}
