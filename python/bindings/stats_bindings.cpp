#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

#include "stats/histogram.h"

namespace py = pybind11;
using namespace tessera::stats;

void bind_stats(py::module_& m) {
    auto stats = m.def_submodule("stats", "Statistical functions for density field analysis");
    
    // HistogramResult struct
    py::class_<HistogramResult>(stats, "HistogramResult",
        R"pbdoc(
        Result of histogram computation.
        
        Attributes
        ----------
        counts : numpy.ndarray
            Number of values in each bin (length = n_bins)
        bin_edges : numpy.ndarray
            Bin edge values (length = n_bins + 1)
        n_below : int
            Count of values below first bin
        n_above : int
            Count of values above last bin
        n_total : int
            Total values processed
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("_counts", &HistogramResult::counts)
        .def_readwrite("_bin_edges", &HistogramResult::bin_edges)
        .def_property_readonly("counts", [](const HistogramResult& r) {
            return py::array_t<int64_t>(py::cast(r.counts));
        })
        .def_property_readonly("bin_edges", [](const HistogramResult& r) {
            return py::array_t<double>(py::cast(r.bin_edges));
        })
        .def_readonly("n_below", &HistogramResult::n_below)
        .def_readonly("n_above", &HistogramResult::n_above)
        .def_readonly("n_total", &HistogramResult::n_total);
    
    // WeightedHistogramResult struct
    py::class_<WeightedHistogramResult>(stats, "WeightedHistogramResult",
        R"pbdoc(
        Result of weighted histogram computation.
        
        Attributes
        ----------
        counts : numpy.ndarray
            Weighted sum in each bin (length = n_bins)
        bin_edges : numpy.ndarray
            Bin edge values (length = n_bins + 1)
        sum_below : float
            Weighted sum below first bin
        sum_above : float
            Weighted sum above last bin
        sum_total : float
            Total weighted sum
        n_total : int
            Total values processed
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("_counts", &WeightedHistogramResult::counts)
        .def_readwrite("_bin_edges", &WeightedHistogramResult::bin_edges)
        .def_property_readonly("counts", [](const WeightedHistogramResult& r) {
            return py::array_t<double>(py::cast(r.counts));
        })
        .def_property_readonly("bin_edges", [](const WeightedHistogramResult& r) {
            return py::array_t<double>(py::cast(r.bin_edges));
        })
        .def_readonly("sum_below", &WeightedHistogramResult::sum_below)
        .def_readonly("sum_above", &WeightedHistogramResult::sum_above)
        .def_readonly("sum_total", &WeightedHistogramResult::sum_total)
        .def_readonly("n_total", &WeightedHistogramResult::n_total);
    
    // histogram function
    stats.def("histogram",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> data,
           int n_bins,
           double range_min,
           double range_max,
           int n_threads) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("data must be 1-dimensional");
            }
            return histogram(
                static_cast<const double*>(buf.ptr),
                buf.size,
                n_bins,
                range_min,
                range_max,
                n_threads
            );
        },
        py::arg("data"),
        py::arg("n_bins"),
        py::arg("range_min") = 0.0,
        py::arg("range_max") = 0.0,
        py::arg("n_threads") = 0,
        R"pbdoc(
        Compute histogram with linearly-spaced bins.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data array (1D)
        n_bins : int
            Number of bins
        range_min : float, optional
            Minimum bin edge (auto-detect if >= range_max)
        range_max : float, optional
            Maximum bin edge
        n_threads : int, optional
            Number of threads (0 = auto)
        
        Returns
        -------
        HistogramResult
            Contains counts, bin_edges, n_below, n_above, n_total
        )pbdoc"
    );
    
    // histogram_log function
    stats.def("histogram_log",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> data,
           int n_bins,
           double range_min,
           double range_max,
           int n_threads) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("data must be 1-dimensional");
            }
            return histogram_log(
                static_cast<const double*>(buf.ptr),
                buf.size,
                n_bins,
                range_min,
                range_max,
                n_threads
            );
        },
        py::arg("data"),
        py::arg("n_bins"),
        py::arg("range_min") = 0.0,
        py::arg("range_max") = 0.0,
        py::arg("n_threads") = 0,
        R"pbdoc(
        Compute histogram with logarithmically-spaced bins.
        
        Values <= 0 are placed in underflow (n_below).
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data array (1D)
        n_bins : int
            Number of bins
        range_min : float, optional
            Minimum bin edge (must be > 0, auto-detect if <= 0)
        range_max : float, optional
            Maximum bin edge (auto-detect if <= range_min)
        n_threads : int, optional
            Number of threads (0 = auto)
        
        Returns
        -------
        HistogramResult
            Contains counts, bin_edges, n_below, n_above, n_total
        )pbdoc"
    );
    
    // histogram_log_weighted function
    stats.def("histogram_log_weighted",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> data,
           py::array_t<double, py::array::c_style | py::array::forcecast> weights,
           int n_bins,
           double range_min,
           double range_max,
           int n_threads) {
            py::buffer_info data_buf = data.request();
            py::buffer_info weight_buf = weights.request();
            if (data_buf.ndim != 1 || weight_buf.ndim != 1) {
                throw std::runtime_error("data and weights must be 1-dimensional");
            }
            if (data_buf.size != weight_buf.size) {
                throw std::runtime_error("data and weights must have same size");
            }
            return histogram_log_weighted(
                static_cast<const double*>(data_buf.ptr),
                static_cast<const double*>(weight_buf.ptr),
                data_buf.size,
                n_bins,
                range_min,
                range_max,
                n_threads
            );
        },
        py::arg("data"),
        py::arg("weights"),
        py::arg("n_bins"),
        py::arg("range_min"),
        py::arg("range_max"),
        py::arg("n_threads") = 0,
        R"pbdoc(
        Compute weighted histogram with logarithmically-spaced bins.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data array (1D)
        weights : numpy.ndarray
            Weight for each data value (same size as data)
        n_bins : int
            Number of bins
        range_min : float
            Minimum bin edge (must be > 0)
        range_max : float
            Maximum bin edge
        n_threads : int, optional
            Number of threads (0 = auto)
        
        Returns
        -------
        WeightedHistogramResult
            Contains counts, bin_edges, sum_below, sum_above, sum_total, n_total
        )pbdoc"
    );
    
    // histogram_to_pdf function
    stats.def("histogram_to_pdf",
        [](const HistogramResult& hist) {
            auto pdf = histogram_to_pdf(hist);
            return py::array_t<double>(py::cast(pdf));
        },
        py::arg("hist"),
        R"pbdoc(
        Compute PDF from histogram.
        
        Normalizes counts so that sum(PDF * bin_width) = 1.
        
        Parameters
        ----------
        hist : HistogramResult
            Histogram result from histogram() or histogram_log()
        
        Returns
        -------
        numpy.ndarray
            PDF values (length = n_bins)
        )pbdoc"
    );
    
    // bin_centers function
    stats.def("bin_centers",
        [](py::array_t<double> bin_edges, bool log_space) {
            py::buffer_info buf = bin_edges.request();
            std::vector<double> edges(
                static_cast<double*>(buf.ptr),
                static_cast<double*>(buf.ptr) + buf.size
            );
            auto centers = bin_centers(edges, log_space);
            return py::array_t<double>(py::cast(centers));
        },
        py::arg("bin_edges"),
        py::arg("log_space") = false,
        R"pbdoc(
        Get bin centers from bin edges.

        Parameters
        ----------
        bin_edges : numpy.ndarray
            Bin edge values
        log_space : bool, optional
            If True, use geometric mean; else arithmetic mean

        Returns
        -------
        numpy.ndarray
            Bin centers (length = len(bin_edges) - 1)
        )pbdoc"
    );

    // ============================================================================
    // Jackknife histogram bindings
    // ============================================================================

    // JackknifeHistogramResult struct
    py::class_<JackknifeHistogramResult>(stats, "JackknifeHistogramResult",
        R"pbdoc(
        Result of jackknife histogram computation with uncertainty estimates.

        Contains both global (full-box) statistics and jackknife estimates
        with uncertainties for histogram counts and PDFs.

        Attributes
        ----------
        bin_edges : numpy.ndarray
            Bin edge values (length = n_bins + 1)
        bin_centers : numpy.ndarray
            Bin center values (length = n_bins)
        n_bins : int
            Number of bins
        n_subboxes : int
            Number of sub-boxes used for jackknife
        global_counts : numpy.ndarray
            Full-box histogram counts
        global_pdf : numpy.ndarray
            Full-box PDF
        global_n_total : int
            Total particles in full box
        counts_mean : numpy.ndarray
            Jackknife mean of counts (per sub-box)
        counts_error : numpy.ndarray
            Jackknife standard error of counts
        pdf_mean : numpy.ndarray
            Jackknife mean of PDF
        pdf_error : numpy.ndarray
            Jackknife standard error of PDF
        subbox_counts : list
            Counts per sub-box [subbox][bin]
        subbox_pdf : list
            PDF per sub-box [subbox][bin]
        subbox_n_total : numpy.ndarray
            Total particles per sub-box
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("bin_edges", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.bin_edges));
        })
        .def_property_readonly("bin_centers", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.bin_centers));
        })
        .def_readonly("n_bins", &JackknifeHistogramResult::n_bins)
        .def_readonly("n_subboxes", &JackknifeHistogramResult::n_subboxes)
        .def_property_readonly("global_counts", [](const JackknifeHistogramResult& r) {
            return py::array_t<int64_t>(py::cast(r.global_counts));
        })
        .def_property_readonly("global_pdf", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.global_pdf));
        })
        .def_readonly("global_n_total", &JackknifeHistogramResult::global_n_total)
        .def_property_readonly("counts_mean", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.counts_mean));
        })
        .def_property_readonly("counts_error", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.counts_error));
        })
        .def_property_readonly("pdf_mean", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.pdf_mean));
        })
        .def_property_readonly("pdf_error", [](const JackknifeHistogramResult& r) {
            return py::array_t<double>(py::cast(r.pdf_error));
        })
        .def_property_readonly("subbox_counts", [](const JackknifeHistogramResult& r) {
            py::list result;
            for (const auto& subbox : r.subbox_counts) {
                result.append(py::array_t<int64_t>(py::cast(subbox)));
            }
            return result;
        })
        .def_property_readonly("subbox_pdf", [](const JackknifeHistogramResult& r) {
            py::list result;
            for (const auto& subbox : r.subbox_pdf) {
                result.append(py::array_t<double>(py::cast(subbox)));
            }
            return result;
        })
        .def_property_readonly("subbox_n_total", [](const JackknifeHistogramResult& r) {
            return py::array_t<int64_t>(py::cast(r.subbox_n_total));
        });

    // JackknifeConditionalResult struct
    py::class_<JackknifeConditionalResult>(stats, "JackknifeConditionalResult",
        R"pbdoc(
        Result of jackknife conditional histogram computation.

        Contains jackknife histograms for all particles and for each
        ORIGAMI morphology class (void, wall, filament, halo).

        Attributes
        ----------
        all : JackknifeHistogramResult
            All particles
        void_class : JackknifeHistogramResult
            Void particles (morphology = 0)
        wall_class : JackknifeHistogramResult
            Wall particles (morphology = 1)
        filament_class : JackknifeHistogramResult
            Filament particles (morphology = 2)
        halo_class : JackknifeHistogramResult
            Halo particles (morphology = 3)
        is_linear_regime : bool
            True if field is in linear regime (f_void >= 0.99)
        f_void : float
            Void fraction
        )pbdoc")
        .def(py::init<>())
        .def_readonly("all", &JackknifeConditionalResult::all)
        .def_readonly("void_class", &JackknifeConditionalResult::void_class)
        .def_readonly("wall_class", &JackknifeConditionalResult::wall_class)
        .def_readonly("filament_class", &JackknifeConditionalResult::filament_class)
        .def_readonly("halo_class", &JackknifeConditionalResult::halo_class)
        .def_readonly("is_linear_regime", &JackknifeConditionalResult::is_linear_regime)
        .def_readonly("f_void", &JackknifeConditionalResult::f_void);

    // compute_jackknife_histogram function
    stats.def("compute_jackknife_histogram",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> values,
           double box_size,
           int n_bins,
           double range_min,
           double range_max,
           bool log_bins,
           int n_subboxes_per_dim,
           int n_threads) {
            py::buffer_info pos_buf = positions.request();
            py::buffer_info val_buf = values.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (n_particles, 3)");
            }
            if (val_buf.ndim != 1) {
                throw std::runtime_error("values must be 1-dimensional");
            }
            if (pos_buf.shape[0] != val_buf.shape[0]) {
                throw std::runtime_error("positions and values must have same number of particles");
            }

            return compute_jackknife_histogram(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const double*>(val_buf.ptr),
                pos_buf.shape[0],
                box_size,
                n_bins,
                range_min,
                range_max,
                log_bins,
                n_subboxes_per_dim,
                n_threads
            );
        },
        py::arg("positions"),
        py::arg("values"),
        py::arg("box_size"),
        py::arg("n_bins"),
        py::arg("range_min"),
        py::arg("range_max"),
        py::arg("log_bins") = true,
        py::arg("n_subboxes_per_dim") = 2,
        py::arg("n_threads") = 0,
        R"pbdoc(
        Compute jackknife histogram with uncertainty estimates.

        Divides the simulation box into sub-boxes and computes leave-one-out
        jackknife estimates for both histogram counts and PDFs.

        Parameters
        ----------
        positions : numpy.ndarray
            Particle positions, shape (n_particles, 3), Eulerian coordinates
        values : numpy.ndarray
            Values to histogram (e.g., overdensity 1+delta)
        box_size : float
            Simulation box size
        n_bins : int
            Number of histogram bins
        range_min : float
            Minimum bin edge (must be > 0 for log bins)
        range_max : float
            Maximum bin edge
        log_bins : bool, optional
            If True, use logarithmic binning (default: True)
        n_subboxes_per_dim : int, optional
            Sub-boxes per dimension (2 -> 8 total, 3 -> 27, etc.)
        n_threads : int, optional
            Number of OpenMP threads (0 = auto)

        Returns
        -------
        JackknifeHistogramResult
            Contains global and jackknife statistics for counts and PDF
        )pbdoc"
    );

    // compute_jackknife_conditional_histogram function
    stats.def("compute_jackknife_conditional_histogram",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> values,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> morphology,
           double box_size,
           int n_bins,
           double range_min,
           double range_max,
           bool log_bins,
           int n_subboxes_per_dim,
           int n_threads) {
            py::buffer_info pos_buf = positions.request();
            py::buffer_info val_buf = values.request();
            py::buffer_info morph_buf = morphology.request();

            if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
                throw std::runtime_error("positions must have shape (n_particles, 3)");
            }
            if (val_buf.ndim != 1) {
                throw std::runtime_error("values must be 1-dimensional");
            }
            if (morph_buf.ndim != 1) {
                throw std::runtime_error("morphology must be 1-dimensional");
            }
            if (pos_buf.shape[0] != val_buf.shape[0] || pos_buf.shape[0] != morph_buf.shape[0]) {
                throw std::runtime_error("positions, values, and morphology must have same number of particles");
            }

            return compute_jackknife_conditional_histogram(
                static_cast<const double*>(pos_buf.ptr),
                static_cast<const double*>(val_buf.ptr),
                static_cast<const uint8_t*>(morph_buf.ptr),
                pos_buf.shape[0],
                box_size,
                n_bins,
                range_min,
                range_max,
                log_bins,
                n_subboxes_per_dim,
                n_threads
            );
        },
        py::arg("positions"),
        py::arg("values"),
        py::arg("morphology"),
        py::arg("box_size"),
        py::arg("n_bins"),
        py::arg("range_min"),
        py::arg("range_max"),
        py::arg("log_bins") = true,
        py::arg("n_subboxes_per_dim") = 2,
        py::arg("n_threads") = 0,
        R"pbdoc(
        Compute jackknife conditional histograms by ORIGAMI morphology class.

        Computes jackknife histograms for all particles and separately for each
        morphology class (void, wall, filament, halo).

        Parameters
        ----------
        positions : numpy.ndarray
            Particle positions, shape (n_particles, 3)
        values : numpy.ndarray
            Values to histogram (e.g., overdensity 1+delta)
        morphology : numpy.ndarray
            ORIGAMI morphology classification (0=void, 1=wall, 2=filament, 3=halo)
        box_size : float
            Simulation box size
        n_bins : int
            Number of histogram bins
        range_min : float
            Minimum bin edge
        range_max : float
            Maximum bin edge
        log_bins : bool, optional
            If True, use logarithmic binning (default: True)
        n_subboxes_per_dim : int, optional
            Sub-boxes per dimension
        n_threads : int, optional
            Number of OpenMP threads (0 = auto)

        Returns
        -------
        JackknifeConditionalResult
            Contains jackknife statistics for all particles and each morphology class
        )pbdoc"
    );
}
