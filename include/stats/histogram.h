#pragma once

/**
 * @file histogram.h
 * @brief Statistical histogram functions for density field analysis
 * 
 * Provides numpy-like histogram functionality with support for:
 * - Linear and logarithmic binning
 * - OpenMP parallelization for large arrays
 * - Weighted histograms
 */

#include <vector>
#include <cstdint>

namespace asymptotic_tetra {
namespace stats {

/**
 * Result of histogram computation.
 * 
 * Mimics numpy.histogram return format:
 * - counts: histogram values (length = n_bins)
 * - bin_edges: bin boundaries (length = n_bins + 1)
 */
struct HistogramResult {
    std::vector<int64_t> counts;      ///< Number of values in each bin
    std::vector<double> bin_edges;    ///< Bin edge values (n_bins + 1)
    int64_t n_below;                  ///< Count of values below first bin
    int64_t n_above;                  ///< Count of values above last bin
    int64_t n_total;                  ///< Total values processed
};

/**
 * Result of weighted histogram computation.
 */
struct WeightedHistogramResult {
    std::vector<double> counts;       ///< Weighted sum in each bin
    std::vector<double> bin_edges;    ///< Bin edge values (n_bins + 1)
    double sum_below;                 ///< Weighted sum below first bin
    double sum_above;                 ///< Weighted sum above last bin
    double sum_total;                 ///< Total weighted sum
    int64_t n_total;                  ///< Total values processed
};

/**
 * Compute histogram with linearly-spaced bins.
 * 
 * @param data Input data array
 * @param n_values Number of values
 * @param n_bins Number of bins
 * @param range_min Minimum bin edge (if >= range_max, auto-detect from data)
 * @param range_max Maximum bin edge
 * @param n_threads Number of threads (0 = auto)
 * @return HistogramResult with counts and bin_edges
 */
HistogramResult histogram(
    const double* data,
    int64_t n_values,
    int n_bins,
    double range_min = 0.0,
    double range_max = 0.0,
    int n_threads = 0
);

/**
 * Compute histogram with logarithmically-spaced bins.
 * 
 * Values <= 0 are placed in underflow (n_below).
 * 
 * @param data Input data array
 * @param n_values Number of values
 * @param n_bins Number of bins
 * @param range_min Minimum bin edge (must be > 0, if <= 0 auto-detect)
 * @param range_max Maximum bin edge (if <= range_min, auto-detect)
 * @param n_threads Number of threads (0 = auto)
 * @return HistogramResult with counts and log-spaced bin_edges
 */
HistogramResult histogram_log(
    const double* data,
    int64_t n_values,
    int n_bins,
    double range_min = 0.0,
    double range_max = 0.0,
    int n_threads = 0
);

/**
 * Compute weighted histogram with logarithmically-spaced bins.
 * 
 * Each value contributes its weight to the bin sum.
 * 
 * @param data Input data array
 * @param weights Weight for each data value
 * @param n_values Number of values
 * @param n_bins Number of bins
 * @param range_min Minimum bin edge (must be > 0)
 * @param range_max Maximum bin edge
 * @param n_threads Number of threads (0 = auto)
 * @return WeightedHistogramResult
 */
WeightedHistogramResult histogram_log_weighted(
    const double* data,
    const double* weights,
    int64_t n_values,
    int n_bins,
    double range_min,
    double range_max,
    int n_threads = 0
);

/**
 * Compute PDF from histogram.
 * 
 * Normalizes counts so that sum(PDF * bin_width) = 1.
 * For log bins, this accounts for varying bin widths.
 * 
 * @param hist Histogram result
 * @return Vector of PDF values (length = n_bins)
 */
std::vector<double> histogram_to_pdf(const HistogramResult& hist);

/**
 * Get bin centers from bin edges.
 * 
 * @param bin_edges Bin edge values
 * @param log_space If true, use geometric mean; else arithmetic mean
 * @return Vector of bin centers (length = bin_edges.size() - 1)
 */
std::vector<double> bin_centers(const std::vector<double>& bin_edges, bool log_space = false);

} // namespace stats
} // namespace asymptotic_tetra
