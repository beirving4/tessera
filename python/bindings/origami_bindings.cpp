/**
 * @file origami_bindings.cpp
 * @brief Python bindings for ORIGAMI morphology classification
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "origami/origami.h"

namespace py = pybind11;
using namespace tessera::origami;

void bind_origami(py::module& m) {
    // Create origami submodule
    auto origami_m = m.def_submodule("origami",
        R"pbdoc(
        ORIGAMI morphology classification for cosmic web structure.

        This module implements the ORIGAMI algorithm for classifying particles
        by their collapse dimensionality based on shell-crossing detection.

        Classes:
        - 0 (Void): No shell crossings, single-stream region
        - 1 (Wall): 1 shell crossing, sheet/wall structure
        - 2 (Filament): 2 shell crossings, filamentary structure
        - 3 (Halo): 3 shell crossings, virialized halo/knot

        Reference: Falck, Neyrinck, & Szalay 2012, ApJ 754, 126
        )pbdoc");

    // MorphologyClass enum
    py::enum_<MorphologyClass>(origami_m, "MorphologyClass",
        "Enumeration of cosmic web morphology classes.")
        .value("Void", MorphologyClass::Void, "No shell crossings (0)")
        .value("Wall", MorphologyClass::Wall, "1 shell crossing (1)")
        .value("Filament", MorphologyClass::Filament, "2 shell crossings (2)")
        .value("Halo", MorphologyClass::Halo, "3 shell crossings (3)")
        .export_values();

    // OrigamiConfig
    py::class_<OrigamiConfig>(origami_m, "OrigamiConfig",
        R"pbdoc(
        Configuration for ORIGAMI morphology computation.

        Attributes
        ----------
        lagrangian_grid_size : int
            Number of particles per dimension (N for N^3 total particles).
        box_size : float
            Physical size of the simulation box.
        n_threads : int
            Number of OpenMP threads. 0 = auto-detect.
        n_split : int
            Domain decomposition factor. The box is split into n_split^3
            subdomains for parallelization. Default: 2.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](int grid_size, double box_size, int n_threads, int n_split) {
            OrigamiConfig cfg;
            cfg.lagrangian_grid_size = grid_size;
            cfg.box_size = box_size;
            cfg.n_threads = n_threads;
            cfg.n_split = n_split;
            return cfg;
        }),
             py::arg("lagrangian_grid_size"),
             py::arg("box_size"),
             py::arg("n_threads") = 0,
             py::arg("n_split") = 2)
        .def_readwrite("lagrangian_grid_size", &OrigamiConfig::lagrangian_grid_size)
        .def_readwrite("box_size", &OrigamiConfig::box_size)
        .def_readwrite("n_threads", &OrigamiConfig::n_threads)
        .def_readwrite("n_split", &OrigamiConfig::n_split);

    // OrigamiResult
    py::class_<OrigamiResult>(origami_m, "OrigamiResult",
        R"pbdoc(
        Result of ORIGAMI morphology computation.

        Attributes
        ----------
        morphology : numpy.ndarray
            Per-particle morphology class (uint8), values 0-3.
        n_void, n_wall, n_filament, n_halo : int
            Particle counts per morphology class.
        f_void, f_wall, f_filament, f_halo : float
            Mass fractions per morphology class.
        particle_density : numpy.ndarray
            Per-particle density (if computed via sample_density_at_particles).
        morphology_grid : numpy.ndarray
            Dominant morphology per cell, shape (cells, cells, cells).
        void_fraction_grid, wall_fraction_grid, filament_fraction_grid, halo_fraction_grid : numpy.ndarray
            Per-class fraction grids, shape (cells, cells, cells).
        grid_cells : int
            Number of grid cells per dimension (if grids computed).
        )pbdoc")
        .def_readonly("n_void", &OrigamiResult::n_void)
        .def_readonly("n_wall", &OrigamiResult::n_wall)
        .def_readonly("n_filament", &OrigamiResult::n_filament)
        .def_readonly("n_halo", &OrigamiResult::n_halo)
        .def_readonly("f_void", &OrigamiResult::f_void)
        .def_readonly("f_wall", &OrigamiResult::f_wall)
        .def_readonly("f_filament", &OrigamiResult::f_filament)
        .def_readonly("f_halo", &OrigamiResult::f_halo)
        .def_readonly("v_void", &OrigamiResult::v_void)
        .def_readonly("v_wall", &OrigamiResult::v_wall)
        .def_readonly("v_filament", &OrigamiResult::v_filament)
        .def_readonly("v_halo", &OrigamiResult::v_halo)
        .def_readonly("grid_cells", &OrigamiResult::grid_cells)
        .def_property_readonly("morphology", [](const OrigamiResult& r) {
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(r.morphology.size())},
                {sizeof(uint8_t)},
                r.morphology.data()
            );
        }, "Per-particle morphology class (0=void, 1=wall, 2=filament, 3=halo)")
        .def_property_readonly("particle_density", [](const OrigamiResult& r) {
            if (r.particle_density.empty()) {
                return py::array_t<double>();
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.particle_density.size())},
                {sizeof(double)},
                r.particle_density.data()
            );
        }, "Per-particle density (from sample_density_at_particles)")
        .def_property_readonly("morphology_grid", [](const OrigamiResult& r) {
            if (r.morphology_grid.empty()) {
                return py::array_t<uint8_t>();
            }
            int c = r.grid_cells;
            return py::array_t<uint8_t>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(uint8_t) * c * c),
                 static_cast<py::ssize_t>(sizeof(uint8_t) * c),
                 static_cast<py::ssize_t>(sizeof(uint8_t))},
                r.morphology_grid.data()
            );
        }, "Dominant morphology per cell, shape (cells, cells, cells)")
        .def_property_readonly("void_fraction_grid", [](const OrigamiResult& r) {
            if (r.void_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.void_fraction_grid.data()
            );
        })
        .def_property_readonly("wall_fraction_grid", [](const OrigamiResult& r) {
            if (r.wall_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.wall_fraction_grid.data()
            );
        })
        .def_property_readonly("filament_fraction_grid", [](const OrigamiResult& r) {
            if (r.filament_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.filament_fraction_grid.data()
            );
        })
        .def_property_readonly("halo_fraction_grid", [](const OrigamiResult& r) {
            if (r.halo_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.halo_fraction_grid.data()
            );
        })
        .def_property_readonly("counts", [](const OrigamiResult& r) {
            return std::array<int64_t, 4>{r.n_void, r.n_wall, r.n_filament, r.n_halo};
        }, "Particle counts as tuple (n_void, n_wall, n_filament, n_halo)")
        .def_property_readonly("fractions", [](const OrigamiResult& r) {
            return std::array<double, 4>{r.f_void, r.f_wall, r.f_filament, r.f_halo};
        }, "Mass fractions as tuple (f_void, f_wall, f_filament, f_halo)")
        .def_property_readonly("volume_fractions", [](const OrigamiResult& r) {
            return std::array<double, 4>{r.v_void, r.v_wall, r.v_filament, r.v_halo};
        }, "Volume fractions as tuple (v_void, v_wall, v_filament, v_halo)");

    // compute_morphology
    origami_m.def("compute_morphology",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           const OrigamiConfig& config) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            int64_t n_particles = buf.shape[0];
            int64_t expected = static_cast<int64_t>(config.lagrangian_grid_size) *
                              config.lagrangian_grid_size * config.lagrangian_grid_size;
            if (n_particles != expected) {
                throw std::runtime_error(
                    "positions has " + std::to_string(n_particles) +
                    " particles, expected " + std::to_string(expected) +
                    " for lagrangian_grid_size=" + std::to_string(config.lagrangian_grid_size));
            }

            const double* data = static_cast<const double*>(buf.ptr);
            return compute_morphology(data, config);
        },
        py::arg("positions"),
        py::arg("config"),
        R"pbdoc(
        Compute ORIGAMI morphology classification for all particles.

        This implements the exact algorithm from Falck, Neyrinck, & Szalay (2012),
        including both Cartesian axis checks and diagonal checks for robustness.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Lagrangian order. Use sort_by_lagrangian_id()
            from the density module if particles are not already sorted.
        config : OrigamiConfig
            Configuration parameters.

        Returns
        -------
        OrigamiResult
            Result containing per-particle morphology and statistics.

        Examples
        --------
        >>> config = at.origami.OrigamiConfig(
        ...     lagrangian_grid_size=256,
        ...     box_size=256.0
        ... )
        >>> result = at.origami.compute_morphology(sorted_positions, config)
        >>> print(f"Halo fraction: {result.f_halo:.2%}")
        )pbdoc");

    // sample_density_at_particles
    origami_m.def("sample_density_at_particles",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> density_grid,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           double box_size,
           OrigamiResult& result) {
            auto density_buf = density_grid.request();
            auto pos_buf = positions.request();

            if (density_buf.ndim != 3) {
                throw std::runtime_error("density_grid must have shape (cells, cells, cells)");
            }
            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }

            int grid_cells = static_cast<int>(density_buf.shape[0]);
            int64_t n_particles = pos_buf.shape[0];

            // Allocate output
            result.particle_density.resize(n_particles);

            sample_density_at_particles(
                static_cast<const double*>(density_buf.ptr),
                grid_cells,
                static_cast<const double*>(pos_buf.ptr),
                n_particles,
                box_size,
                result.particle_density.data()
            );
        },
        py::arg("density_grid"),
        py::arg("positions"),
        py::arg("box_size"),
        py::arg("result"),
        R"pbdoc(
        Sample a 3D density field at particle positions.

        Uses trilinear interpolation to get density values at each particle's
        Eulerian position. Useful for computing conditional PDFs P(1+delta | class).

        Parameters
        ----------
        density_grid : ndarray, shape (cells, cells, cells)
            3D density field (in [z, y, x] ordering).
        positions : ndarray, shape (N, 3)
            Particle positions.
        box_size : float
            Simulation box size.
        result : OrigamiResult
            Result object to fill with particle_density array.

        Examples
        --------
        >>> # After computing density and morphology
        >>> at.origami.sample_density_at_particles(
        ...     density_3d, positions, box_size, morph_result)
        >>> halo_densities = morph_result.particle_density[morph_result.morphology == 3]
        )pbdoc");

    // deposit_morphology_to_grid
    origami_m.def("deposit_morphology_to_grid",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           double box_size,
           int grid_cells,
           OrigamiResult& result) {
            auto pos_buf = positions.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }

            int64_t n_particles = pos_buf.shape[0];

            if (result.morphology.empty()) {
                throw std::runtime_error("result.morphology is empty - run compute_morphology first");
            }
            if (static_cast<int64_t>(result.morphology.size()) != n_particles) {
                throw std::runtime_error("morphology size does not match positions");
            }

            deposit_morphology_to_grid(
                result.morphology.data(),
                static_cast<const double*>(pos_buf.ptr),
                n_particles,
                box_size,
                grid_cells,
                result
            );
        },
        py::arg("positions"),
        py::arg("box_size"),
        py::arg("grid_cells"),
        py::arg("result"),
        R"pbdoc(
        Deposit morphology classifications onto a 3D grid.

        Creates grid-based representations of the morphology field for visualization.
        Each cell gets the dominant class and per-class fractions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions.
        box_size : float
            Simulation box size.
        grid_cells : int
            Number of grid cells per dimension.
        result : OrigamiResult
            Result object with morphology array; will be filled with grid outputs.

        Examples
        --------
        >>> at.origami.deposit_morphology_to_grid(
        ...     positions, box_size, 128, morph_result)
        >>> plt.imshow(morph_result.halo_fraction_grid[64, :, :])  # z=64 slice
        )pbdoc");

    // morphology_name helper
    origami_m.def("morphology_name",
        [](uint8_t cls) { return morphology_name(cls); },
        py::arg("morphology_class"),
        R"pbdoc(
        Get string name for a morphology class.

        Parameters
        ----------
        morphology_class : int
            Morphology class (0-3).

        Returns
        -------
        str
            Name: "Void", "Wall", "Filament", or "Halo".
        )pbdoc");
}
