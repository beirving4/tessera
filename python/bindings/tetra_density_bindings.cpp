/**
 * @file tetra_density_bindings.cpp
 * @brief Python bindings for tetrahedron-based density field computation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "density/tetra_density.h"

namespace py = pybind11;
using namespace asymptotic_tetra::density;

void bind_tetra_density(py::module& m) {
    // TetraDensityConfig
    py::class_<TetraDensityConfig>(m, "TetraDensityConfig",
        R"pbdoc(
        Configuration for tetrahedron-based density field computation.
        
        Attributes
        ----------
        lagrangian_grid_size : int
            Number of particles per dimension (N for N^3 total particles).
            Must match the cube root of the total particle count.
        output_cells : int
            Number of cells per dimension in the output density grid.
        box_size : float
            Physical size of the simulation box.
        particle_mass : float
            Mass per particle in code units.
        n_samples : int
            Number of Monte Carlo samples per tetrahedron. Higher values
            give smoother results but take longer. Default: 100.
        n_threads : int
            Number of OpenMP threads. 0 = auto-detect.
        periodic : bool
            Whether to use periodic boundary conditions. Default: True.
        seed : int
            Random seed for reproducibility. 0 = use current time.
        subbox_enabled : bool
            If True, render only within a sub-region for high-resolution local
            density fields (e.g., around a halo). Default: False.
        subbox_origin : tuple
            Origin (x, y, z) of the sub-box in physical coordinates.
        subbox_width : tuple
            Width (dx, dy, dz) of the sub-box. Output grid will span this region.
        )pbdoc")
        .def(py::init<>())
        .def(py::init([](int grid_size, int output_cells, double box_size,
                        double particle_mass, int n_samples, int n_threads,
                        bool periodic, uint64_t seed) {
            TetraDensityConfig cfg;
            cfg.lagrangian_grid_size = grid_size;
            cfg.output_cells = output_cells;
            cfg.box_size = box_size;
            cfg.particle_mass = particle_mass;
            cfg.n_samples = n_samples;
            cfg.n_threads = n_threads;
            cfg.periodic = periodic;
            cfg.seed = seed;
            return cfg;
        }),
             py::arg("lagrangian_grid_size"),
             py::arg("output_cells"),
             py::arg("box_size"),
             py::arg("particle_mass") = 1.0,
             py::arg("n_samples") = 100,
             py::arg("n_threads") = 0,
             py::arg("periodic") = true,
             py::arg("seed") = 0)
        .def_readwrite("lagrangian_grid_size", &TetraDensityConfig::lagrangian_grid_size)
        .def_readwrite("output_cells", &TetraDensityConfig::output_cells)
        .def_readwrite("box_size", &TetraDensityConfig::box_size)
        .def_readwrite("particle_mass", &TetraDensityConfig::particle_mass)
        .def_readwrite("n_samples", &TetraDensityConfig::n_samples)
        .def_readwrite("n_threads", &TetraDensityConfig::n_threads)
        .def_readwrite("periodic", &TetraDensityConfig::periodic)
        .def_readwrite("seed", &TetraDensityConfig::seed)
        .def_readwrite("subbox_enabled", &TetraDensityConfig::subbox_enabled)
        .def_property("subbox_origin",
            [](const TetraDensityConfig& cfg) {
                return std::make_tuple(cfg.subbox_origin[0], cfg.subbox_origin[1], cfg.subbox_origin[2]);
            },
            [](TetraDensityConfig& cfg, const std::tuple<double, double, double>& origin) {
                cfg.subbox_origin[0] = std::get<0>(origin);
                cfg.subbox_origin[1] = std::get<1>(origin);
                cfg.subbox_origin[2] = std::get<2>(origin);
            },
            "Sub-box origin (x, y, z)")
        .def_property("subbox_width",
            [](const TetraDensityConfig& cfg) {
                return std::make_tuple(cfg.subbox_width[0], cfg.subbox_width[1], cfg.subbox_width[2]);
            },
            [](TetraDensityConfig& cfg, const std::tuple<double, double, double>& width) {
                cfg.subbox_width[0] = std::get<0>(width);
                cfg.subbox_width[1] = std::get<1>(width);
                cfg.subbox_width[2] = std::get<2>(width);
            },
            "Sub-box dimensions (dx, dy, dz)");
    
    // TetraDensityResult3D
    py::class_<TetraDensityResult3D>(m, "TetraDensityResult3D",
        R"pbdoc(
        Result of 3D tetrahedron-based density computation.
        
        Attributes
        ----------
        density : numpy.ndarray
            3D density field, shape (cells, cells, cells).
            Units: mass per volume.
        cells : int
            Number of cells per dimension.
        cell_width : float
            Physical width of each cell.
        total_mass : float
            Total mass deposited (for verification).
        mean_density : float
            Mean density of the field.
        n_tetrahedra : int
            Number of tetrahedra processed.
        )pbdoc")
        .def_readonly("cells", &TetraDensityResult3D::cells)
        .def_readonly("cell_width", &TetraDensityResult3D::cell_width)
        .def_readonly("total_mass", &TetraDensityResult3D::total_mass)
        .def_readonly("mean_density", &TetraDensityResult3D::mean_density)
        .def_readonly("n_tetrahedra", &TetraDensityResult3D::n_tetrahedra)
        .def_property_readonly("density", [](const TetraDensityResult3D& r) {
            // Return as 3D numpy array
            int c = r.cells;
            return py::array_t<double>(
                {c, c, c},
                {sizeof(double) * c * c, sizeof(double) * c, sizeof(double)},
                r.density.data()
            );
        }, "3D density field as numpy array (cells, cells, cells)");
    
    // TetraDensityResult2D
    py::class_<TetraDensityResult2D>(m, "TetraDensityResult2D",
        R"pbdoc(
        Result of 2D tetrahedron-based density computation (projection or slice).
        
        Attributes
        ----------
        density : numpy.ndarray
            2D surface density field, shape (cells, cells).
            Units: mass per area.
        cells : int
            Number of cells per dimension.
        cell_width : float
            Physical width of each cell.
        projection_axis : int
            Axis projected along (0=x, 1=y, 2=z).
        slice_min, slice_max : float
            Slice bounds (if applicable).
        mean_surface_density : float
            Mean surface density.
        )pbdoc")
        .def_readonly("cells", &TetraDensityResult2D::cells)
        .def_readonly("cell_width", &TetraDensityResult2D::cell_width)
        .def_readonly("projection_axis", &TetraDensityResult2D::projection_axis)
        .def_readonly("slice_min", &TetraDensityResult2D::slice_min)
        .def_readonly("slice_max", &TetraDensityResult2D::slice_max)
        .def_readonly("mean_surface_density", &TetraDensityResult2D::mean_surface_density)
        .def_property_readonly("density", [](const TetraDensityResult2D& r) {
            int c = r.cells;
            return py::array_t<double>(
                {c, c},
                {sizeof(double) * c, sizeof(double)},
                r.density.data()
            );
        }, "2D surface density field as numpy array (cells, cells)");
    
    // UnitTetraSamples
    py::class_<UnitTetraSamples, std::shared_ptr<UnitTetraSamples>>(m, "UnitTetraSamples",
        R"pbdoc(
        Pre-generated Monte Carlo samples for unit tetrahedra.
        
        Reuse these across multiple density computations for efficiency.
        
        Parameters
        ----------
        n_buffers : int
            Number of independent sample buffers.
        n_samples : int
            Number of samples per buffer.
        seed : int
            Random seed (0 = time-based).
        )pbdoc")
        .def(py::init<int, int, uint64_t>(),
             py::arg("n_buffers") = 64,
             py::arg("n_samples") = 100,
             py::arg("seed") = 0)
        .def_property_readonly("n_buffers", &UnitTetraSamples::n_buffers)
        .def_property_readonly("n_samples", &UnitTetraSamples::n_samples);
    
    // Main computation functions
    m.def("compute_tetra_density_3d",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           const TetraDensityConfig& config,
           std::shared_ptr<UnitTetraSamples> samples) {
            // Validate input
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
                    " for grid_size=" + std::to_string(config.lagrangian_grid_size));
            }
            
            const double* data = static_cast<const double*>(buf.ptr);
            return compute_tetra_density_3d(data, config, samples.get());
        },
        py::arg("positions"),
        py::arg("config"),
        py::arg("samples") = nullptr,
        R"pbdoc(
        Compute a 3D density field using tetrahedron tessellation.
        
        This implements the core gotetra algorithm with OpenMP parallelization.
        
        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Lagrangian order. Use sort_by_lagrangian_id()
            first if particles are not already sorted.
        config : TetraDensityConfig
            Computation configuration.
        samples : UnitTetraSamples, optional
            Pre-generated samples for reuse. If None, generates new samples.
        
        Returns
        -------
        TetraDensityResult3D
            Result containing the 3D density field.
        
        Examples
        --------
        >>> config = at.density.TetraDensityConfig(
        ...     lagrangian_grid_size=128,
        ...     output_cells=256,
        ...     box_size=100.0,
        ...     particle_mass=1.0,
        ...     n_samples=100
        ... )
        >>> result = at.density.compute_tetra_density_3d(positions, config)
        >>> print(result.density.shape)
        (256, 256, 256)
        )pbdoc");
    
    m.def("compute_tetra_density_2d_projection",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           const TetraDensityConfig& config,
           int projection_axis,
           std::shared_ptr<UnitTetraSamples> samples) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            const double* data = static_cast<const double*>(buf.ptr);
            return compute_tetra_density_2d_projection(data, config, projection_axis, samples.get());
        },
        py::arg("positions"),
        py::arg("config"),
        py::arg("projection_axis") = 2,
        py::arg("samples") = nullptr,
        R"pbdoc(
        Compute a 2D projected density field (column density).
        
        Projects all mass along one axis to create a 2D surface density map.
        
        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Lagrangian order.
        config : TetraDensityConfig
            Computation configuration.
        projection_axis : int
            Axis to project along (0=x, 1=y, 2=z). Default: 2 (z).
        samples : UnitTetraSamples, optional
            Pre-generated samples.
        
        Returns
        -------
        TetraDensityResult2D
            Result containing the 2D surface density field.
        )pbdoc");
    
    m.def("compute_tetra_density_2d_slice",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           const TetraDensityConfig& config,
           int projection_axis,
           double slice_min,
           double slice_max,
           std::shared_ptr<UnitTetraSamples> samples) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (N, 3)");
            }
            const double* data = static_cast<const double*>(buf.ptr);
            return compute_tetra_density_2d_slice(data, config, projection_axis,
                                                   slice_min, slice_max, samples.get());
        },
        py::arg("positions"),
        py::arg("config"),
        py::arg("projection_axis") = 2,
        py::arg("slice_min") = 0.0,
        py::arg("slice_max") = -1.0,
        py::arg("samples") = nullptr,
        R"pbdoc(
        Compute a 2D slice density field.
        
        Extracts and projects a thin slice of the volume.
        
        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in Lagrangian order.
        config : TetraDensityConfig
            Computation configuration.
        projection_axis : int
            Axis perpendicular to slice (0=x, 1=y, 2=z). Default: 2 (z).
        slice_min : float
            Start of slice along projection axis.
        slice_max : float
            End of slice. If negative, defaults to slice_min + 10% of box.
        samples : UnitTetraSamples, optional
            Pre-generated samples.
        
        Returns
        -------
        TetraDensityResult2D
            Result containing the slice surface density.
        )pbdoc");
    
    // Sorting function
    m.def("sort_by_lagrangian_id",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> particle_ids,
           int grid_size,
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
            
            py::ssize_t n_particles = pos_buf.shape[0];
            
            // Allocate output
            py::array_t<double> sorted(std::vector<py::ssize_t>{n_particles, 3});
            auto sorted_buf = sorted.request();
            
            sort_by_lagrangian_id(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const int64_t*>(id_buf.ptr),
                n_particles,
                grid_size,
                static_cast<double*>(sorted_buf.ptr),
                id_offset
            );
            
            return sorted;
        },
        py::arg("positions"),
        py::arg("particle_ids"),
        py::arg("grid_size"),
        py::arg("id_offset") = 1,
        R"pbdoc(
        Sort particles from arbitrary order to Lagrangian order.
        
        In GADGET-style simulations, particle IDs encode the original Lagrangian
        grid position. This function reorders particles so that particle i
        corresponds to Lagrangian position:
            ix = i % grid_size
            iy = (i / grid_size) % grid_size
            iz = i / grid_size^2
        
        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Particle positions in arbitrary order.
        particle_ids : ndarray, shape (N,)
            Particle IDs encoding Lagrangian positions.
        grid_size : int
            Lagrangian grid dimension (N for N^3 particles).
        id_offset : int
            Value to subtract from IDs. Default: 1 (for 1-indexed GADGET IDs).
        
        Returns
        -------
        sorted_positions : ndarray, shape (N, 3)
            Positions sorted into Lagrangian order.
        
        Examples
        --------
        >>> # Load GADGET snapshot
        >>> snap = at.io.read_gadget4_snapshot("snapshot.hdf5", ...)
        >>> pos = np.array(snap.particles[1].coordinates)
        >>> ids = np.array(snap.particles[1].particle_ids)
        >>> grid_size = int(round(len(pos) ** (1/3)))
        >>> sorted_pos = at.density.sort_by_lagrangian_id(pos, ids, grid_size)
        )pbdoc");
    
    // Helper to generate tetrahedron indices (useful for debugging/visualization)
    m.def("generate_tetra_indices",
        [](int grid_size, bool periodic) {
            auto indices = generate_tetra_indices(grid_size, periodic);
            py::ssize_t n_tetra = static_cast<py::ssize_t>(indices.size());
            
            py::array_t<int64_t> result(std::vector<py::ssize_t>{n_tetra, 4});
            auto buf = result.mutable_unchecked<2>();
            
            for (int64_t i = 0; i < n_tetra; ++i) {
                for (int j = 0; j < 4; ++j) {
                    buf(i, j) = indices[i][j];
                }
            }
            
            return result;
        },
        py::arg("grid_size"),
        py::arg("periodic") = true,
        R"pbdoc(
        Generate tetrahedron corner indices for a Lagrangian grid.
        
        Each Lagrangian cell is decomposed into 6 tetrahedra following the
        gotetra convention.
        
        Parameters
        ----------
        grid_size : int
            Particles per dimension.
        periodic : bool
            Include boundary-wrapping tetrahedra. Default: True.
        
        Returns
        -------
        indices : ndarray, shape (n_tetra, 4)
            Corner indices for each tetrahedron into the flattened particle array.
        )pbdoc");
}
