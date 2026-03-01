/**
 * @file origami_bindings.cpp
 * @brief Python bindings for ORIGAMI morphology classification
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "origami/origami.h"
#include "origami/pipeline.h"

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
        linear_regime_threshold : float
            Void fraction threshold for linear regime detection.
            If f_void >= this value, the field is considered to be in
            the linear regime (no shell-crossing). Default: 0.99.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](int grid_size, double box_size, int n_threads, int n_split, double linear_regime_threshold) {
            OrigamiConfig cfg;
            cfg.lagrangian_grid_size = grid_size;
            cfg.box_size = box_size;
            cfg.n_threads = n_threads;
            cfg.n_split = n_split;
            cfg.linear_regime_threshold = linear_regime_threshold;
            return cfg;
        }),
             py::arg("lagrangian_grid_size"),
             py::arg("box_size"),
             py::arg("n_threads") = 0,
             py::arg("n_split") = 2,
             py::arg("linear_regime_threshold") = 0.99)
        .def_readwrite("lagrangian_grid_size", &OrigamiConfig::lagrangian_grid_size)
        .def_readwrite("box_size", &OrigamiConfig::box_size)
        .def_readwrite("n_threads", &OrigamiConfig::n_threads)
        .def_readwrite("n_split", &OrigamiConfig::n_split)
        .def_readwrite("linear_regime_threshold", &OrigamiConfig::linear_regime_threshold,
            "Void fraction threshold for linear regime detection (default: 0.99)")
        .def_readwrite("cartesian_only", &OrigamiConfig::cartesian_only,
            "If True, only use Cartesian axes (x,y,z) for shell-crossing detection, "
            "skipping diagonal axis checks. Default: False (use all 4 axis sets per Falck+ 2012)");

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
        is_linear_regime : bool
            True if the density field is in the linear regime (f_void >= threshold).
            In the linear regime, no shell-crossing has occurred, so ORIGAMI
            classifies all particles as voids. The classification is not
            physically meaningful in this case.
        linear_regime_threshold : float
            Void fraction threshold used for linear regime detection.
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
        .def_readonly("is_linear_regime", &OrigamiResult::is_linear_regime,
            "True if field is in linear regime (no shell-crossing)")
        .def_readonly("linear_regime_threshold", &OrigamiResult::linear_regime_threshold,
            "Void fraction threshold used for linear regime detection")
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

    // =========================================================================
    // Pipeline API (unified ORIGAMI workflow)
    // =========================================================================

    // IdOrdering enum
    py::enum_<IdOrdering>(origami_m, "IdOrdering",
        R"pbdoc(
        ID ordering convention for Lagrangian grid mapping.

        GADGET-4 uses z-major: ID = 1 + z + y*N + x*N^2
        Some codes use x-major: ID = 1 + x + y*N + z*N^2
        )pbdoc")
        .value("XMajor", IdOrdering::XMajor, "ID = 1 + x + y*N + z*N^2 (x varies fastest)")
        .value("ZMajor", IdOrdering::ZMajor, "ID = 1 + z + y*N + x*N^2 (z varies fastest, GADGET-4 default)")
        .value("Auto", IdOrdering::Auto, "Automatically detect from particle data")
        .export_values();

    // id_ordering_name helper
    origami_m.def("id_ordering_name",
        [](IdOrdering ordering) { return id_ordering_name(ordering); },
        py::arg("ordering"),
        R"pbdoc(
        Get string name for an IdOrdering enum value.

        Parameters
        ----------
        ordering : IdOrdering
            The ordering enum value.

        Returns
        -------
        str
            Name: "x-major", "z-major", or "auto".
        )pbdoc");

    // OrigamiPipelineConfig
    py::class_<OrigamiPipelineConfig>(origami_m, "PipelineConfig",
        R"pbdoc(
        Configuration for the unified ORIGAMI pipeline.

        This configures all aspects of the pipeline including ID handling,
        sorting, morphology computation, and optional density/grid outputs.

        Attributes
        ----------
        lagrangian_grid_size : int
            Particles per dimension (N for N^3 total). Required.
        box_size : float
            Physical box size. Required.
        id_ordering : IdOrdering
            How particle IDs encode positions. Default: Auto (detect).
        id_offset : int
            Value to subtract from IDs (1 for GADGET-4). Default: 1.
        sort_in_place : bool
            Sort positions in-place (saves ~24GB for N=1024^3). Default: True.
        positions_already_sorted : bool
            Skip sorting if already in Lagrangian order. Default: False.
        n_split : int
            Domain decomposition for parallelization. Default: 2.
        linear_regime_threshold : float
            Void fraction threshold for linear regime. Default: 0.99.
        density_output_cells : int
            0 = skip density, >0 = compute at this resolution. Default: 0.
        density_n_samples : int
            Monte Carlo samples per tetrahedron. Default: 100.
        particle_mass : float
            Particle mass for density normalization. Default: 1.0.
        density_periodic : bool
            Periodic boundaries for density. Default: True.
        use_direct_particle_density : bool
            If True, use direct volume-based density instead of grid-based Monte Carlo.
            The direct method is more memory-efficient (~2.5x less) and faster (~50x),
            but may produce slightly different distributions at the extremes (very low
            and very high overdensities). Use as fallback for memory-constrained systems.
            Default: False.
        grid_cells : int
            0 = skip grid, >0 = deposit morphology at this resolution. Default: 0.
        sample_density_at_particles : bool
            Sample density field at particle positions. Default: True.
        n_threads : int
            Number of threads (0 = auto). Default: 0.
        seed : int
            Random seed for density (0 = time-based). Default: 0.
        validate_ids : bool
            Validate particle ID range and uniqueness. Default: True.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](int grid_size, double box_size) {
            return OrigamiPipelineConfig(grid_size, box_size);
        }),
            py::arg("lagrangian_grid_size"),
            py::arg("box_size"))
        .def_readwrite("lagrangian_grid_size", &OrigamiPipelineConfig::lagrangian_grid_size)
        .def_readwrite("box_size", &OrigamiPipelineConfig::box_size)
        .def_readwrite("id_ordering", &OrigamiPipelineConfig::id_ordering)
        .def_readwrite("id_offset", &OrigamiPipelineConfig::id_offset)
        .def_readwrite("sort_in_place", &OrigamiPipelineConfig::sort_in_place)
        .def_readwrite("positions_already_sorted", &OrigamiPipelineConfig::positions_already_sorted)
        .def_readwrite("n_split", &OrigamiPipelineConfig::n_split)
        .def_readwrite("linear_regime_threshold", &OrigamiPipelineConfig::linear_regime_threshold)
        .def_readwrite("density_output_cells", &OrigamiPipelineConfig::density_output_cells)
        .def_readwrite("density_n_samples", &OrigamiPipelineConfig::density_n_samples)
        .def_readwrite("particle_mass", &OrigamiPipelineConfig::particle_mass)
        .def_readwrite("density_periodic", &OrigamiPipelineConfig::density_periodic)
        .def_readwrite("use_direct_particle_density", &OrigamiPipelineConfig::use_direct_particle_density,
            "Use direct volume-based density instead of grid-based Monte Carlo. "
            "More memory-efficient and faster, but may differ slightly at distribution extremes. Default: False.")
        .def_readwrite("use_voronoi_density", &OrigamiPipelineConfig::use_voronoi_density,
            "Use Voronoi (VTFE) density from Eulerian cell volumes. "
            "Resolves high-density tail up to 10^4-10^6 without grid smoothing. "
            "Does not require Lagrangian sorting for density computation. "
            "Requires BUILD_WITH_VORO=ON. Default: False.")
        .def_readwrite("use_cic_density", &OrigamiPipelineConfig::use_cic_density,
            "Use CIC (Cloud-in-Cell) grid density. Fastest method but grid-smoothed "
            "(smoothing scale = cell_width). Does not require Lagrangian sorting. "
            "Reference: Klypin et al. 2018. Default: False.")
        .def_readwrite("cic_output_cells", &OrigamiPipelineConfig::cic_output_cells,
            "Grid resolution for CIC density (cells per dimension). Default: 256.")
        .def_readwrite("grid_cells", &OrigamiPipelineConfig::grid_cells)
        .def_readwrite("sample_density_at_particles", &OrigamiPipelineConfig::sample_density_at_particles)
        .def_readwrite("pdf_n_bins", &OrigamiPipelineConfig::pdf_n_bins,
            "Number of PDF bins (0 = skip PDF computation)")
        .def_readwrite("pdf_log_bins", &OrigamiPipelineConfig::pdf_log_bins,
            "Use logarithmic bins (recommended for density)")
        .def_readwrite("pdf_range_min", &OrigamiPipelineConfig::pdf_range_min,
            "Minimum bin edge (0 = auto-detect)")
        .def_readwrite("pdf_range_max", &OrigamiPipelineConfig::pdf_range_max,
            "Maximum bin edge (0 = auto-detect)")
        .def_readwrite("pdf_jackknife", &OrigamiPipelineConfig::pdf_jackknife,
            "Enable jackknife resampling for uncertainty estimation")
        .def_readwrite("pdf_jackknife_subboxes", &OrigamiPipelineConfig::pdf_jackknife_subboxes,
            "Sub-boxes per dimension for jackknife (2->8, 3->27)")
        .def_readwrite("pdf_volume_weighted", &OrigamiPipelineConfig::pdf_volume_weighted,
            "Compute volume-weighted PDF from density grid cells. "
            "Requires grid-based density (grid-MC or CIC). Default: False.")
        .def_readwrite("cartesian_only", &OrigamiPipelineConfig::cartesian_only,
            "If True, only use Cartesian axes (x,y,z) for shell-crossing detection, "
            "skipping diagonal axis checks. Useful for diagnosing the effect of diagonal axes "
            "on morphology classification. Default: False (use all 4 axis sets per Falck+ 2012)")
        .def_readwrite("n_threads", &OrigamiPipelineConfig::n_threads)
        .def_readwrite("seed", &OrigamiPipelineConfig::seed)
        .def_readwrite("validate_ids", &OrigamiPipelineConfig::validate_ids);

    // OrigamiPipelineResult
    py::class_<OrigamiPipelineResult>(origami_m, "PipelineResult",
        R"pbdoc(
        Complete result from the ORIGAMI pipeline.

        Contains all outputs from the unified pipeline including morphology
        classification, optional density field, and optional grid deposition.

        Attributes
        ----------
        morphology : numpy.ndarray
            Per-particle morphology class (uint8), values 0-3.
        n_void, n_wall, n_filament, n_halo : int
            Particle counts per morphology class.
        f_void, f_wall, f_filament, f_halo : float
            Mass fractions per morphology class.
        is_linear_regime : bool
            True if the field is in the linear regime.
        detected_ordering : IdOrdering
            The ID ordering detected (or specified).
        density_3d : numpy.ndarray
            3D density field (if density_output_cells > 0).
        density_cells : int
            Cells per dimension for density field.
        mean_density : float
            Mean density value.
        particle_density : numpy.ndarray
            Per-particle density (if sample_density_at_particles).
        morphology_grid : numpy.ndarray
            Dominant morphology per cell (if grid_cells > 0).
        *_fraction_grid : numpy.ndarray
            Per-class fraction grids (if grid_cells > 0).
        v_void, v_wall, v_filament, v_halo : float
            Volume fractions (from grid).
        sorting_performed : bool
            Whether positions were sorted.
        *_time_ms : float
            Timing diagnostics for each stage.
        )pbdoc")
        .def_readonly("n_void", &OrigamiPipelineResult::n_void)
        .def_readonly("n_wall", &OrigamiPipelineResult::n_wall)
        .def_readonly("n_filament", &OrigamiPipelineResult::n_filament)
        .def_readonly("n_halo", &OrigamiPipelineResult::n_halo)
        .def_readonly("f_void", &OrigamiPipelineResult::f_void)
        .def_readonly("f_wall", &OrigamiPipelineResult::f_wall)
        .def_readonly("f_filament", &OrigamiPipelineResult::f_filament)
        .def_readonly("f_halo", &OrigamiPipelineResult::f_halo)
        .def_readonly("is_linear_regime", &OrigamiPipelineResult::is_linear_regime)
        .def_readonly("linear_regime_threshold", &OrigamiPipelineResult::linear_regime_threshold)
        .def_readonly("detected_ordering", &OrigamiPipelineResult::detected_ordering)
        .def_readonly("density_cells", &OrigamiPipelineResult::density_cells)
        .def_readonly("density_cell_width", &OrigamiPipelineResult::density_cell_width)
        .def_readonly("mean_density", &OrigamiPipelineResult::mean_density)
        .def_readonly("min_tetra_volume", &OrigamiPipelineResult::min_tetra_volume,
            "Minimum tetrahedron volume (direct density only)")
        .def_readonly("max_tetra_volume", &OrigamiPipelineResult::max_tetra_volume,
            "Maximum tetrahedron volume (direct density only)")
        .def_readonly("mean_tetra_volume", &OrigamiPipelineResult::mean_tetra_volume,
            "Mean tetrahedron volume (direct density only)")
        .def_readonly("stddev_tetra_volume", &OrigamiPipelineResult::stddev_tetra_volume,
            "Standard deviation of tetrahedron volumes (direct density only)")
        .def_readonly("stddev2_tetra_volume", &OrigamiPipelineResult::stddev2_tetra_volume,
            "2 * standard deviation of tetrahedron volumes (direct density only)")
        .def_readonly("log_stddev_tetra_volume", &OrigamiPipelineResult::log_stddev_tetra_volume,
            "Log-normal standard deviation of tetrahedron volumes (direct density only)")
        .def_readonly("log_stddev2_tetra_volume", &OrigamiPipelineResult::log_stddev2_tetra_volume,
            "2 * log-normal standard deviation of tetrahedron volumes (direct density only)")
        .def_readonly("morph_grid_cells", &OrigamiPipelineResult::morph_grid_cells)
        .def_readonly("v_void", &OrigamiPipelineResult::v_void)
        .def_readonly("v_wall", &OrigamiPipelineResult::v_wall)
        .def_readonly("v_filament", &OrigamiPipelineResult::v_filament)
        .def_readonly("v_halo", &OrigamiPipelineResult::v_halo)
        .def_readonly("sorting_performed", &OrigamiPipelineResult::sorting_performed)
        .def_readonly("detection_time_ms", &OrigamiPipelineResult::detection_time_ms)
        .def_readonly("sorting_time_ms", &OrigamiPipelineResult::sorting_time_ms)
        .def_readonly("morphology_time_ms", &OrigamiPipelineResult::morphology_time_ms)
        .def_readonly("density_time_ms", &OrigamiPipelineResult::density_time_ms)
        .def_readonly("sampling_time_ms", &OrigamiPipelineResult::sampling_time_ms)
        .def_readonly("grid_time_ms", &OrigamiPipelineResult::grid_time_ms)
        .def_readonly("pdf_time_ms", &OrigamiPipelineResult::pdf_time_ms)
        .def_readonly("total_time_ms", &OrigamiPipelineResult::total_time_ms)
        .def_readonly("pdf_n_bins", &OrigamiPipelineResult::pdf_n_bins)
        .def_readonly("overdensity_mean", &OrigamiPipelineResult::overdensity_mean)
        .def_readonly("overdensity_median", &OrigamiPipelineResult::overdensity_median)
        .def_readonly("pdf_jackknife_enabled", &OrigamiPipelineResult::pdf_jackknife_enabled)
        .def_readonly("pdf_jackknife_n_subboxes", &OrigamiPipelineResult::pdf_jackknife_n_subboxes)
        // Per-class overdensity statistics (arrays of 5: [all, void, wall, filament, halo])
        .def_property_readonly("od_mean", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_mean.data());
        }, "Mean overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_stddev", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_stddev.data());
        }, "Standard deviation of overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_median", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_median.data());
        }, "Median overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p5", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p5.data());
        }, "5th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p10", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p10.data());
        }, "10th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p25", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p25.data());
        }, "25th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p75", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p75.data());
        }, "75th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p90", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p90.data());
        }, "90th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p95", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p95.data());
        }, "95th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("od_p99", [](const OrigamiPipelineResult& r) {
            return py::array_t<double>({5}, {sizeof(double)}, r.od_p99.data());
        }, "99th percentile overdensity per class [all, void, wall, filament, halo]")
        .def_property_readonly("morphology", [](const OrigamiPipelineResult& r) {
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(r.morphology.size())},
                {sizeof(uint8_t)},
                r.morphology.data()
            );
        }, "Per-particle morphology class (0=void, 1=wall, 2=filament, 3=halo)")
        .def_property_readonly("density_3d", [](const OrigamiPipelineResult& r) {
            if (r.density_3d.empty()) {
                return py::array_t<double>();
            }
            int c = r.density_cells;
            return py::array_t<double>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(double) * c * c),
                 static_cast<py::ssize_t>(sizeof(double) * c),
                 static_cast<py::ssize_t>(sizeof(double))},
                r.density_3d.data()
            );
        }, "3D density field, shape (cells, cells, cells)")
        .def_property_readonly("particle_density", [](const OrigamiPipelineResult& r) {
            if (r.particle_density.empty()) {
                return py::array_t<double>();
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.particle_density.size())},
                {sizeof(double)},
                r.particle_density.data()
            );
        }, "Per-particle density values")
        .def_property_readonly("particle_volume", [](const OrigamiPipelineResult& r) {
            if (r.particle_volume.empty()) {
                return py::array_t<double>();
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.particle_volume.size())},
                {sizeof(double)},
                r.particle_volume.data()
            );
        }, "Per-particle tessellation volume (direct density only). "
           "Smoothing length: h_i = particle_volume[i]^(1/3)")
        .def_property_readonly("morphology_grid", [](const OrigamiPipelineResult& r) {
            if (r.morphology_grid.empty()) {
                return py::array_t<uint8_t>();
            }
            int c = r.morph_grid_cells;
            return py::array_t<uint8_t>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(uint8_t) * c * c),
                 static_cast<py::ssize_t>(sizeof(uint8_t) * c),
                 static_cast<py::ssize_t>(sizeof(uint8_t))},
                r.morphology_grid.data()
            );
        }, "Dominant morphology per cell, shape (cells, cells, cells)")
        .def_property_readonly("void_fraction_grid", [](const OrigamiPipelineResult& r) {
            if (r.void_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.morph_grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.void_fraction_grid.data()
            );
        })
        .def_property_readonly("wall_fraction_grid", [](const OrigamiPipelineResult& r) {
            if (r.wall_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.morph_grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.wall_fraction_grid.data()
            );
        })
        .def_property_readonly("filament_fraction_grid", [](const OrigamiPipelineResult& r) {
            if (r.filament_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.morph_grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.filament_fraction_grid.data()
            );
        })
        .def_property_readonly("halo_fraction_grid", [](const OrigamiPipelineResult& r) {
            if (r.halo_fraction_grid.empty()) {
                return py::array_t<float>();
            }
            int c = r.morph_grid_cells;
            return py::array_t<float>(
                {c, c, c},
                {static_cast<py::ssize_t>(sizeof(float) * c * c),
                 static_cast<py::ssize_t>(sizeof(float) * c),
                 static_cast<py::ssize_t>(sizeof(float))},
                r.halo_fraction_grid.data()
            );
        })
        .def_property_readonly("counts", [](const OrigamiPipelineResult& r) {
            return std::array<int64_t, 4>{r.n_void, r.n_wall, r.n_filament, r.n_halo};
        }, "Particle counts as tuple (n_void, n_wall, n_filament, n_halo)")
        .def_property_readonly("fractions", [](const OrigamiPipelineResult& r) {
            return std::array<double, 4>{r.f_void, r.f_wall, r.f_filament, r.f_halo};
        }, "Mass fractions as tuple (f_void, f_wall, f_filament, f_halo)")
        .def_property_readonly("volume_fractions", [](const OrigamiPipelineResult& r) {
            return std::array<double, 4>{r.v_void, r.v_wall, r.v_filament, r.v_halo};
        }, "Volume fractions as tuple (v_void, v_wall, v_filament, v_halo)")
        // PDF bin information
        .def_property_readonly("pdf_bin_edges", [](const OrigamiPipelineResult& r) {
            if (r.pdf_bin_edges.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_bin_edges.size())},
                {sizeof(double)},
                r.pdf_bin_edges.data()
            );
        }, "PDF bin edges (n_bins + 1)")
        .def_property_readonly("pdf_bin_centers", [](const OrigamiPipelineResult& r) {
            if (r.pdf_bin_centers.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_bin_centers.size())},
                {sizeof(double)},
                r.pdf_bin_centers.data()
            );
        }, "PDF bin centers (n_bins)")
        // PDF arrays
        .def_property_readonly("pdf_all", [](const OrigamiPipelineResult& r) {
            if (r.pdf_all.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_all.size())},
                {sizeof(double)},
                r.pdf_all.data()
            );
        }, "PDF for all particles")
        .def_property_readonly("pdf_void", [](const OrigamiPipelineResult& r) {
            if (r.pdf_void.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_void.size())},
                {sizeof(double)},
                r.pdf_void.data()
            );
        }, "PDF for void particles")
        .def_property_readonly("pdf_wall", [](const OrigamiPipelineResult& r) {
            if (r.pdf_wall.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_wall.size())},
                {sizeof(double)},
                r.pdf_wall.data()
            );
        }, "PDF for wall particles")
        .def_property_readonly("pdf_filament", [](const OrigamiPipelineResult& r) {
            if (r.pdf_filament.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_filament.size())},
                {sizeof(double)},
                r.pdf_filament.data()
            );
        }, "PDF for filament particles")
        .def_property_readonly("pdf_halo", [](const OrigamiPipelineResult& r) {
            if (r.pdf_halo.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_halo.size())},
                {sizeof(double)},
                r.pdf_halo.data()
            );
        }, "PDF for halo particles")
        // Histogram counts
        .def_property_readonly("hist_all", [](const OrigamiPipelineResult& r) {
            if (r.hist_all.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_all.size())},
                {sizeof(int64_t)},
                r.hist_all.data()
            );
        }, "Histogram counts for all particles")
        .def_property_readonly("hist_void", [](const OrigamiPipelineResult& r) {
            if (r.hist_void.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_void.size())},
                {sizeof(int64_t)},
                r.hist_void.data()
            );
        }, "Histogram counts for void particles")
        .def_property_readonly("hist_wall", [](const OrigamiPipelineResult& r) {
            if (r.hist_wall.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_wall.size())},
                {sizeof(int64_t)},
                r.hist_wall.data()
            );
        }, "Histogram counts for wall particles")
        .def_property_readonly("hist_filament", [](const OrigamiPipelineResult& r) {
            if (r.hist_filament.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_filament.size())},
                {sizeof(int64_t)},
                r.hist_filament.data()
            );
        }, "Histogram counts for filament particles")
        .def_property_readonly("hist_halo", [](const OrigamiPipelineResult& r) {
            if (r.hist_halo.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_halo.size())},
                {sizeof(int64_t)},
                r.hist_halo.data()
            );
        }, "Histogram counts for halo particles")
        // Jackknife errors
        .def_property_readonly("pdf_all_error", [](const OrigamiPipelineResult& r) {
            if (r.pdf_all_error.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_all_error.size())},
                {sizeof(double)},
                r.pdf_all_error.data()
            );
        }, "Jackknife error for all PDF")
        .def_property_readonly("pdf_void_error", [](const OrigamiPipelineResult& r) {
            if (r.pdf_void_error.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_void_error.size())},
                {sizeof(double)},
                r.pdf_void_error.data()
            );
        }, "Jackknife error for void PDF")
        .def_property_readonly("pdf_wall_error", [](const OrigamiPipelineResult& r) {
            if (r.pdf_wall_error.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_wall_error.size())},
                {sizeof(double)},
                r.pdf_wall_error.data()
            );
        }, "Jackknife error for wall PDF")
        .def_property_readonly("pdf_filament_error", [](const OrigamiPipelineResult& r) {
            if (r.pdf_filament_error.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_filament_error.size())},
                {sizeof(double)},
                r.pdf_filament_error.data()
            );
        }, "Jackknife error for filament PDF")
        .def_property_readonly("pdf_halo_error", [](const OrigamiPipelineResult& r) {
            if (r.pdf_halo_error.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_halo_error.size())},
                {sizeof(double)},
                r.pdf_halo_error.data()
            );
        }, "Jackknife error for halo PDF")
        // Volume-weighted PDF
        .def_property_readonly("pdf_all_volume_weighted", [](const OrigamiPipelineResult& r) {
            if (r.pdf_all_volume_weighted.empty()) return py::array_t<double>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(r.pdf_all_volume_weighted.size())},
                {sizeof(double)},
                r.pdf_all_volume_weighted.data()
            );
        }, "Volume-weighted PDF from grid cells")
        .def_property_readonly("hist_all_volume_weighted", [](const OrigamiPipelineResult& r) {
            if (r.hist_all_volume_weighted.empty()) return py::array_t<int64_t>();
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.hist_all_volume_weighted.size())},
                {sizeof(int64_t)},
                r.hist_all_volume_weighted.data()
            );
        }, "Volume-weighted histogram counts from grid cells")
        // Mass-weighted mean density
        .def_readonly("rho_M", &OrigamiPipelineResult::rho_M,
            "Mass-weighted mean density = mean(particle_density)");

    // detect_id_ordering
    origami_m.def("detect_id_ordering",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
           double box_size) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }

            int64_t n_particles = pos_buf.shape[0];
            return detect_id_ordering(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                box_size
            );
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("box_size"),
        R"pbdoc(
        Detect ID ordering convention from particle positions and IDs.

        Uses displacement correlations between neighboring IDs to robustly
        detect the ordering even when particles have moved significantly
        and wrapped around periodic boundaries.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions, any order.
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        grid_size : int
            Lagrangian grid size (N for N^3).
        box_size : float
            Simulation box size.

        Returns
        -------
        IdOrdering
            Detected ordering (XMajor or ZMajor).

        Examples
        --------
        >>> ordering = ts.origami.detect_id_ordering(pos, ids, 256, 256.0)
        >>> print(ts.origami.id_ordering_name(ordering))
        )pbdoc");

    // sort_to_lagrangian_inplace
    origami_m.def("sort_to_lagrangian_inplace",
        [](py::array_t<double, py::array::c_style> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
           IdOrdering ordering,
           int64_t id_offset) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }
            if (!positions.writeable()) {
                throw std::runtime_error("positions array must be writeable for in-place sorting");
            }

            int64_t n_particles = pos_buf.shape[0];
            sort_to_lagrangian_inplace(
                static_cast<double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                ordering,
                id_offset
            );
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("ordering"),
        py::arg("id_offset") = 1,
        R"pbdoc(
        Sort particles to x-major Lagrangian order (in-place).

        Modifies the positions array in-place using cycle decomposition
        to minimize memory usage. For N=1024^3, this saves ~24GB compared
        to creating a copy.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions - WILL BE MODIFIED.
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        grid_size : int
            Lagrangian grid size (N for N^3).
        ordering : IdOrdering
            ID ordering convention (use detect_id_ordering if unknown).
        id_offset : int, optional
            Value to subtract from IDs. Default: 1 (GADGET-4).

        Examples
        --------
        >>> ordering = ts.origami.detect_id_ordering(pos, ids, 256, 256.0)
        >>> ts.origami.sort_to_lagrangian_inplace(pos, ids, 256, ordering)
        >>> # pos is now in Lagrangian order
        )pbdoc");

    // sort_to_lagrangian (out-of-place)
    origami_m.def("sort_to_lagrangian",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
           IdOrdering ordering,
           int64_t id_offset) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }

            int64_t n_particles = pos_buf.shape[0];

            // Create output array
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n_particles), 3};
            auto result = py::array_t<double>(shape);
            auto result_buf = result.request();

            sort_to_lagrangian(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                ordering,
                id_offset,
                static_cast<double*>(result_buf.ptr)
            );

            return result;
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("ordering"),
        py::arg("id_offset") = 1,
        R"pbdoc(
        Sort particles to x-major Lagrangian order (out-of-place).

        Creates a new sorted array, leaving input unchanged.
        Uses more memory but is safer for debugging.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions (not modified).
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        grid_size : int
            Lagrangian grid size (N for N^3).
        ordering : IdOrdering
            ID ordering convention.
        id_offset : int, optional
            Value to subtract from IDs. Default: 1.

        Returns
        -------
        ndarray, shape (N, 3)
            Sorted positions in Lagrangian order.

        Examples
        --------
        >>> sorted_pos = ts.origami.sort_to_lagrangian(pos, ids, 256, ordering)
        )pbdoc");

    // sort_particles_to_lagrangian (convenience function with auto-detection)
    origami_m.def("sort_particles_to_lagrangian",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
           double box_size) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }

            int64_t n_particles = pos_buf.shape[0];
            int64_t expected = static_cast<int64_t>(grid_size) * grid_size * grid_size;
            if (n_particles != expected) {
                throw std::runtime_error(
                    "positions has " + std::to_string(n_particles) +
                    " particles, expected " + std::to_string(expected) +
                    " for grid_size=" + std::to_string(grid_size));
            }

            IdOrdering detected_ordering;
            int64_t detected_offset;

            auto sorted = sort_particles_to_lagrangian(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                box_size,
                &detected_ordering,
                &detected_offset
            );

            // Create output array
            std::vector<ssize_t> shape = {static_cast<ssize_t>(n_particles), 3};
            auto result = py::array_t<double>(shape);
            auto result_buf = result.request();
            std::memcpy(result_buf.ptr, sorted.data(), sorted.size() * sizeof(double));

            // Return tuple of (sorted_positions, ordering_name, id_offset)
            return py::make_tuple(result, id_ordering_name(detected_ordering), detected_offset);
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("box_size"),
        R"pbdoc(
        Sort particles to Lagrangian order with automatic ID ordering detection.

        This is a convenience function that combines detect_id_ordering() and
        sort_to_lagrangian() into a single call. It automatically:
        1. Detects the ID ordering convention (x-major or z-major)
        2. Detects the ID offset (minimum particle ID)
        3. Sorts positions to x-major Lagrangian order

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions.
        particle_ids : ndarray, shape (N,)
            Particle IDs encoding Lagrangian positions.
        grid_size : int
            Lagrangian grid size (N for N^3).
        box_size : float
            Simulation box size.

        Returns
        -------
        tuple
            (sorted_positions, ordering_name, id_offset) where:
            - sorted_positions: ndarray (N, 3) in x-major Lagrangian order
            - ordering_name: str, detected ordering ("x-major" or "z-major")
            - id_offset: int, detected ID offset (minimum particle ID)

        Examples
        --------
        >>> sorted_pos, ordering, offset = ts.origami.sort_particles_to_lagrangian(
        ...     positions, particle_ids, grid_size=256, box_size=256.0)
        >>> print(f"Detected: {ordering}, offset={offset}")
        )pbdoc");

    // run_pipeline
    origami_m.def("run_pipeline",
        [](py::array_t<double, py::array::c_style> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           const OrigamiPipelineConfig& config) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }

            int64_t expected = static_cast<int64_t>(config.lagrangian_grid_size) *
                              config.lagrangian_grid_size * config.lagrangian_grid_size;
            if (pos_buf.shape[0] != expected) {
                throw std::runtime_error(
                    "positions has " + std::to_string(pos_buf.shape[0]) +
                    " particles, expected " + std::to_string(expected) +
                    " for lagrangian_grid_size=" + std::to_string(config.lagrangian_grid_size));
            }

            if (config.sort_in_place && !positions.writeable()) {
                throw std::runtime_error(
                    "positions array must be writeable when sort_in_place=True. "
                    "Use run_pipeline_safe() or make array writeable.");
            }

            return run_pipeline(
                static_cast<double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                config
            );
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("config"),
        R"pbdoc(
        Run the complete ORIGAMI analysis pipeline.

        This is the main entry point for ORIGAMI analysis. It handles:
        1. ID ordering detection (if ordering=Auto)
        2. ID validation (if validate_ids=True)
        3. Sorting to Lagrangian order (in-place if sort_in_place=True)
        4. ORIGAMI morphology classification
        5. Optional density field computation
        6. Optional density sampling at particles
        7. Optional grid deposition

        WARNING: If sort_in_place=True (default), the positions array WILL BE MODIFIED.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions - may be modified if sort_in_place=True.
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        config : PipelineConfig
            Pipeline configuration.

        Returns
        -------
        PipelineResult
            Complete result with all requested outputs.

        Examples
        --------
        >>> config = ts.origami.PipelineConfig(256, 256.0)
        >>> config.density_output_cells = 128  # Enable density
        >>> config.grid_cells = 64             # Enable grid deposition
        >>> result = ts.origami.run_pipeline(positions, particle_ids, config)
        >>> print(f"Halo fraction: {result.f_halo:.2%}")
        )pbdoc");

    // run_pipeline_safe
    origami_m.def("run_pipeline_safe",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           const OrigamiPipelineConfig& config) {
            auto pos_buf = positions.request();
            auto id_buf = particle_ids.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }
            if (pos_buf.shape[0] != id_buf.shape[0]) {
                throw std::runtime_error("positions and particle_ids must have same length");
            }

            int64_t expected = static_cast<int64_t>(config.lagrangian_grid_size) *
                              config.lagrangian_grid_size * config.lagrangian_grid_size;
            if (pos_buf.shape[0] != expected) {
                throw std::runtime_error(
                    "positions has " + std::to_string(pos_buf.shape[0]) +
                    " particles, expected " + std::to_string(expected) +
                    " for lagrangian_grid_size=" + std::to_string(config.lagrangian_grid_size));
            }

            return run_pipeline_safe(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                config
            );
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("config"),
        R"pbdoc(
        Run the complete ORIGAMI analysis pipeline (safe version).

        This version is safe but uses more memory. The positions array
        will not be modified, but an internal copy will be made for sorting.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions (not modified).
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        config : PipelineConfig
            Pipeline configuration.

        Returns
        -------
        PipelineResult
            Complete result with all requested outputs.

        Examples
        --------
        >>> result = ts.origami.run_pipeline_safe(positions, particle_ids, config)
        )pbdoc");

    // validate_particle_ids
    origami_m.def("validate_particle_ids",
        [](py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
           int64_t id_offset) {
            auto id_buf = particle_ids.request();

            if (id_buf.ndim != 1) {
                throw std::runtime_error("particle_ids must be 1D array");
            }

            int64_t n_particles = id_buf.shape[0];
            validate_particle_ids(
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                id_offset
            );
        },
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("id_offset") = 1,
        R"pbdoc(
        Validate particle IDs for ORIGAMI computation.

        Checks:
        - Particle count matches grid_size^3
        - ID range is [id_offset, id_offset + N^3 - 1]
        - All IDs are unique

        Parameters
        ----------
        particle_ids : ndarray, shape (N,)
            Particle IDs.
        grid_size : int
            Lagrangian grid size.
        id_offset : int, optional
            Value subtracted from IDs. Default: 1.

        Raises
        ------
        RuntimeError
            If validation fails.

        Examples
        --------
        >>> ts.origami.validate_particle_ids(ids, 256)  # Validates N=256^3
        )pbdoc");
}
