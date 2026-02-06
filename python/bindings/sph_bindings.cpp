/**
 * @file sph_bindings.cpp
 * @brief Python bindings for SPH density renderer
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "density/sph_renderer.h"

namespace py = pybind11;
using namespace tessera::density;

void bind_sph(py::module& m) {
    // SPHKernel enum
    py::enum_<SPHKernel>(m, "SPHKernel",
        R"pbdoc(
        SPH kernel function types.

        Attributes
        ----------
        CUBIC_SPLINE : SPHKernel
            M4 cubic spline kernel. Most common choice, compact support at 2h.
        WENDLAND_C2 : SPHKernel
            Wendland C2 kernel. Smoother than cubic spline, compact support at 2h.
        WENDLAND_C4 : SPHKernel
            Wendland C4 kernel. Very smooth, compact support at 2h.
        QUINTIC_SPLINE : SPHKernel
            M6 quintic spline kernel. High accuracy, compact support at 3h.
        )pbdoc")
        .value("CUBIC_SPLINE", SPHKernel::CUBIC_SPLINE)
        .value("WENDLAND_C2", SPHKernel::WENDLAND_C2)
        .value("WENDLAND_C4", SPHKernel::WENDLAND_C4)
        .value("QUINTIC_SPLINE", SPHKernel::QUINTIC_SPLINE);

    // Kernel string conversion
    m.def("sph_kernel_to_string", &sph_kernel_to_string,
        py::arg("kernel"),
        "Convert SPH kernel enum to string");

    m.def("sph_kernel_from_string", &sph_kernel_from_string,
        py::arg("name"),
        "Parse SPH kernel from string (e.g., 'cubic_spline', 'wendland_c2')");

    // SPHConfig
    py::class_<SPHConfig>(m, "SPHConfig",
        R"pbdoc(
        Configuration for SPH density rendering.

        Attributes
        ----------
        output_cells : int
            Output grid cells per dimension. Default: 256.
        box_size : float
            Physical size of output region.
        center : tuple
            Center of rendering region (x, y, z).
        render_width : float
            Width of rendering region. Default: use box_size.
        kernel : SPHKernel
            SPH kernel function. Default: CUBIC_SPLINE.
        smoothing_length : float
            Fixed smoothing length. 0 = adaptive (recommended). Default: 0.
        neighbors_target : int
            Target neighbors for adaptive smoothing. Default: 32.
        h_min : float
            Minimum smoothing length. 0 = no limit.
        h_max : float
            Maximum smoothing length. 0 = no limit.
        sim_box_size : float
            Full simulation box size (for periodic boundaries).
        periodic : bool
            Use periodic boundary conditions. Default: True.
        particle_mass : float
            Mass per particle. Default: 1.0.
        n_threads : int
            Number of OpenMP threads. 0 = auto. Default: 0.
        projection_axis : int
            Axis to project along (0=x, 1=y, 2=z). Default: 2 (z).
        slice_thickness : float
            Slice thickness (0 = full projection). Default: 0.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](int output_cells, double box_size,
                        std::tuple<double, double, double> center,
                        SPHKernel kernel, double smoothing_length,
                        double sim_box_size, bool periodic,
                        double particle_mass, int projection_axis,
                        int n_threads) {
            SPHConfig cfg;
            cfg.output_cells = output_cells;
            cfg.box_size = box_size;
            cfg.center = {std::get<0>(center), std::get<1>(center), std::get<2>(center)};
            cfg.render_width = box_size;
            cfg.kernel = kernel;
            cfg.smoothing_length = smoothing_length;
            cfg.sim_box_size = sim_box_size;
            cfg.periodic = periodic;
            cfg.particle_mass = particle_mass;
            cfg.projection_axis = projection_axis;
            cfg.n_threads = n_threads;
            return cfg;
        }),
             py::arg("output_cells") = 256,
             py::arg("box_size") = 0.0,
             py::arg("center") = std::make_tuple(0.0, 0.0, 0.0),
             py::arg("kernel") = SPHKernel::CUBIC_SPLINE,
             py::arg("smoothing_length") = 0.0,
             py::arg("sim_box_size") = 0.0,
             py::arg("periodic") = true,
             py::arg("particle_mass") = 1.0,
             py::arg("projection_axis") = 2,
             py::arg("n_threads") = 0)
        .def_readwrite("output_cells", &SPHConfig::output_cells)
        .def_readwrite("box_size", &SPHConfig::box_size)
        .def_property("center",
            [](const SPHConfig& cfg) {
                return std::make_tuple(cfg.center[0], cfg.center[1], cfg.center[2]);
            },
            [](SPHConfig& cfg, std::tuple<double, double, double> c) {
                cfg.center = {std::get<0>(c), std::get<1>(c), std::get<2>(c)};
            })
        .def_readwrite("render_width", &SPHConfig::render_width)
        .def_readwrite("kernel", &SPHConfig::kernel)
        .def_readwrite("smoothing_length", &SPHConfig::smoothing_length)
        .def_readwrite("neighbors_target", &SPHConfig::neighbors_target)
        .def_readwrite("h_min", &SPHConfig::h_min)
        .def_readwrite("h_max", &SPHConfig::h_max)
        .def_readwrite("sim_box_size", &SPHConfig::sim_box_size)
        .def_readwrite("periodic", &SPHConfig::periodic)
        .def_readwrite("particle_mass", &SPHConfig::particle_mass)
        .def_readwrite("n_threads", &SPHConfig::n_threads)
        .def_readwrite("projection_axis", &SPHConfig::projection_axis)
        .def_readwrite("slice_thickness", &SPHConfig::slice_thickness)
        .def_static("for_halo", &SPHConfig::for_halo,
            py::arg("halo_center"),
            py::arg("box_width"),
            py::arg("sim_box"),
            py::arg("resolution") = 256,
            R"pbdoc(
            Create configuration for halo-centered rendering.

            Parameters
            ----------
            halo_center : tuple
                Halo center position (x, y, z).
            box_width : float
                Physical width of rendering region.
            sim_box : float
                Full simulation box size.
            resolution : int
                Output grid resolution. Default: 256.

            Returns
            -------
            SPHConfig
                Configuration for halo rendering.
            )pbdoc");

    // SPHResult2D
    py::class_<SPHResult2D>(m, "SPHResult2D",
        R"pbdoc(
        Result of 2D SPH density projection.

        Attributes
        ----------
        density : numpy.ndarray
            2D surface density field, shape (cells, cells).
        counts : numpy.ndarray
            Particles contributing to each cell.
        cells : int
            Number of cells per dimension.
        cell_width : float
            Physical cell width.
        projection_axis : int
            Axis projected along.
        mean_density : float
            Mean surface density.
        total_mass : float
            Total mass in projection.
        render_time_ms : float
            Rendering time in milliseconds.
        n_particles : int
            Number of particles processed.
        kernel_evals : int
            Total kernel evaluations.
        extent_x : tuple
            X extent (min, max).
        extent_y : tuple
            Y extent (min, max).
        )pbdoc")
        .def_readonly("cells", &SPHResult2D::cells)
        .def_readonly("cell_width", &SPHResult2D::cell_width)
        .def_readonly("projection_axis", &SPHResult2D::projection_axis)
        .def_readonly("mean_density", &SPHResult2D::mean_density)
        .def_readonly("total_mass", &SPHResult2D::total_mass)
        .def_readonly("render_time_ms", &SPHResult2D::render_time_ms)
        .def_readonly("n_particles", &SPHResult2D::n_particles)
        .def_readonly("kernel_evals", &SPHResult2D::kernel_evals)
        .def_property_readonly("density", [](const SPHResult2D& r) {
            int c = r.cells;
            return py::array_t<double>(
                {c, c},
                {sizeof(double) * c, sizeof(double)},
                r.density.data()
            );
        }, "2D surface density as numpy array (cells, cells)")
        .def_property_readonly("counts", [](const SPHResult2D& r) {
            int c = r.cells;
            return py::array_t<int64_t>(
                {c, c},
                {sizeof(int64_t) * c, sizeof(int64_t)},
                r.counts.data()
            );
        }, "Particle count per cell as numpy array (cells, cells)")
        .def_property_readonly("extent_x", [](const SPHResult2D& r) {
            return std::make_tuple(r.extent_x[0], r.extent_x[1]);
        })
        .def_property_readonly("extent_y", [](const SPHResult2D& r) {
            return std::make_tuple(r.extent_y[0], r.extent_y[1]);
        })
        .def_property_readonly("extent", [](const SPHResult2D& r) {
            return std::make_tuple(r.extent_x[0], r.extent_x[1],
                                   r.extent_y[0], r.extent_y[1]);
        }, "Full extent as (x_min, x_max, y_min, y_max)");

    // SPHResult3D
    py::class_<SPHResult3D>(m, "SPHResult3D",
        R"pbdoc(
        Result of 3D SPH density field.

        Attributes
        ----------
        density : numpy.ndarray
            3D density field, shape (cells, cells, cells).
        counts : numpy.ndarray
            Particles contributing to each voxel.
        cells : int
            Number of cells per dimension.
        cell_width : float
            Physical cell width.
        mean_density : float
            Mean density.
        total_mass : float
            Total mass in volume.
        render_time_ms : float
            Rendering time in milliseconds.
        )pbdoc")
        .def_readonly("cells", &SPHResult3D::cells)
        .def_readonly("cell_width", &SPHResult3D::cell_width)
        .def_readonly("mean_density", &SPHResult3D::mean_density)
        .def_readonly("total_mass", &SPHResult3D::total_mass)
        .def_readonly("render_time_ms", &SPHResult3D::render_time_ms)
        .def_readonly("n_particles", &SPHResult3D::n_particles)
        .def_readonly("kernel_evals", &SPHResult3D::kernel_evals)
        .def_property_readonly("density", [](const SPHResult3D& r) {
            int c = r.cells;
            return py::array_t<double>(
                {c, c, c},
                {sizeof(double) * c * c, sizeof(double) * c, sizeof(double)},
                r.density.data()
            );
        }, "3D density as numpy array (cells, cells, cells)")
        .def_property_readonly("counts", [](const SPHResult3D& r) {
            int c = r.cells;
            return py::array_t<int64_t>(
                {c, c, c},
                {sizeof(int64_t) * c * c, sizeof(int64_t) * c, sizeof(int64_t)},
                r.counts.data()
            );
        }, "Particle count per voxel as numpy array");

    // SPHParticles
    py::class_<SPHParticles>(m, "SPHParticles",
        R"pbdoc(
        Particle data for SPH rendering.

        Structure-of-Arrays format for efficient SIMD access.

        Attributes
        ----------
        x, y, z : list
            Particle positions.
        mass : list
            Particle masses (optional).
        h : list
            Smoothing lengths (optional, computed if empty).
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](py::array_t<float> x, py::array_t<float> y, py::array_t<float> z,
                        py::array_t<float> mass, py::array_t<float> h) {
            SPHParticles p;
            auto xb = x.request();
            auto yb = y.request();
            auto zb = z.request();

            int64_t n = xb.shape[0];
            p.x.assign(static_cast<float*>(xb.ptr), static_cast<float*>(xb.ptr) + n);
            p.y.assign(static_cast<float*>(yb.ptr), static_cast<float*>(yb.ptr) + n);
            p.z.assign(static_cast<float*>(zb.ptr), static_cast<float*>(zb.ptr) + n);

            if (mass.size() > 0) {
                auto mb = mass.request();
                p.mass.assign(static_cast<float*>(mb.ptr), static_cast<float*>(mb.ptr) + n);
            }
            if (h.size() > 0) {
                auto hb = h.request();
                p.h.assign(static_cast<float*>(hb.ptr), static_cast<float*>(hb.ptr) + n);
            }
            return p;
        }),
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("mass") = py::array_t<float>(),
             py::arg("h") = py::array_t<float>())
        .def("__len__", &SPHParticles::size)
        .def_property_readonly("size", &SPHParticles::size)
        .def("empty", &SPHParticles::empty)
        .def("clear", &SPHParticles::clear)
        .def("add", &SPHParticles::add,
            py::arg("x"), py::arg("y"), py::arg("z"),
            py::arg("mass") = 1.0f, py::arg("h") = 0.0f)
        .def_property_readonly("x", [](const SPHParticles& p) {
            return py::array_t<float>(p.x.size(), p.x.data());
        })
        .def_property_readonly("y", [](const SPHParticles& p) {
            return py::array_t<float>(p.y.size(), p.y.data());
        })
        .def_property_readonly("z", [](const SPHParticles& p) {
            return py::array_t<float>(p.z.size(), p.z.data());
        })
        .def_property_readonly("mass", [](const SPHParticles& p) {
            return py::array_t<float>(p.mass.size(), p.mass.data());
        })
        .def_property_readonly("h", [](const SPHParticles& p) {
            return py::array_t<float>(p.h.size(), p.h.data());
        });

    // SPHRenderer
    py::class_<SPHRenderer>(m, "SPHRenderer",
        R"pbdoc(
        SPH density renderer.

        High-performance SPH rendering with SIMD optimization and OpenMP
        parallelization.

        Parameters
        ----------
        config : SPHConfig
            Rendering configuration.

        Examples
        --------
        >>> config = ts.density.SPHConfig.for_halo(
        ...     halo_center=(50.0, 50.0, 50.0),
        ...     box_width=10.0,
        ...     sim_box=100.0,
        ...     resolution=256
        ... )
        >>> renderer = ts.density.SPHRenderer(config)
        >>> result = renderer.render_2d(particles)
        )pbdoc")
        .def(py::init<const SPHConfig&>(), py::arg("config"))
        .def("render_2d",
            py::overload_cast<const SPHParticles&>(&SPHRenderer::render_2d),
            py::arg("particles"),
            R"pbdoc(
            Render 2D density projection.

            Parameters
            ----------
            particles : SPHParticles
                Particle data.

            Returns
            -------
            SPHResult2D
                2D density projection result.
            )pbdoc")
        .def("render_3d", &SPHRenderer::render_3d,
            py::arg("particles"),
            "Render 3D density field.")
        .def("compute_smoothing_lengths", &SPHRenderer::compute_smoothing_lengths,
            py::arg("particles"),
            "Compute adaptive smoothing lengths for particles.")
        .def_property_readonly("config", &SPHRenderer::config);

    // Convenience function: render_sph_density_2d
    m.def("render_sph_density_2d",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           std::tuple<double, double, double> center,
           double box_width,
           int output_cells,
           double sim_box_size,
           int projection_axis,
           SPHKernel kernel,
           double smoothing_length,
           int n_threads) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            int64_t n = buf.shape[0];
            std::array<double, 3> c = {std::get<0>(center), std::get<1>(center), std::get<2>(center)};

            return render_sph_density_2d(
                static_cast<const double*>(buf.ptr), n, c,
                box_width, output_cells, sim_box_size,
                projection_axis, kernel, smoothing_length, n_threads
            );
        },
        py::arg("positions"),
        py::arg("center"),
        py::arg("box_width"),
        py::arg("output_cells") = 256,
        py::arg("sim_box_size") = 0.0,
        py::arg("projection_axis") = 2,
        py::arg("kernel") = SPHKernel::CUBIC_SPLINE,
        py::arg("smoothing_length") = 0.0,
        py::arg("n_threads") = 0,
        R"pbdoc(
        Render 2D SPH density projection.

        High-level function for rendering density around a halo or region.

        Parameters
        ----------
        positions : numpy.ndarray
            Particle positions, shape (N, 3).
        center : tuple
            Center of rendering region (x, y, z).
        box_width : float
            Width of rendering region.
        output_cells : int
            Output grid resolution. Default: 256.
        sim_box_size : float
            Full simulation box size (for periodic boundaries). Default: 0 (non-periodic).
        projection_axis : int
            Axis to project along (0=x, 1=y, 2=z). Default: 2 (z).
        kernel : SPHKernel
            SPH kernel type. Default: CUBIC_SPLINE.
        smoothing_length : float
            Fixed smoothing length. 0 = adaptive. Default: 0.
        n_threads : int
            Number of threads. 0 = auto. Default: 0.

        Returns
        -------
        SPHResult2D
            2D density projection result.

        Examples
        --------
        >>> result = ts.density.render_sph_density_2d(
        ...     positions,
        ...     center=(50.0, 50.0, 50.0),
        ...     box_width=10.0,
        ...     output_cells=256,
        ...     sim_box_size=100.0
        ... )
        >>> print(f"Rendered in {result.render_time_ms:.1f} ms")
        >>> print(result.density.shape)
        (256, 256)
        )pbdoc");

    // Utility function
    m.def("estimate_smoothing_length", &estimate_smoothing_length,
        py::arg("n_particles"),
        py::arg("box_size"),
        py::arg("neighbors_target") = 32,
        R"pbdoc(
        Estimate smoothing length from particle count and box size.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        box_size : float
            Box size containing particles.
        neighbors_target : int
            Target number of neighbors. Default: 32.

        Returns
        -------
        float
            Estimated smoothing length.
        )pbdoc");
}
