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
}
