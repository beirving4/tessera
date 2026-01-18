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

namespace tessera {
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

// ============================================================================
// Jackknife resampling for uncertainty estimation
// ============================================================================

/**
 * Result of jackknife histogram computation.
 *
 * Contains both global (full-box) statistics and jackknife estimates
 * with uncertainties for histogram counts and PDFs.
 */
struct JackknifeHistogramResult {
    // Bin information
    std::vector<double> bin_edges;      ///< Bin edge values (n_bins + 1)
    std::vector<double> bin_centers;    ///< Bin center values (n_bins)
    int n_bins;                          ///< Number of bins
    int n_subboxes;                      ///< Number of sub-boxes used

    // Global (full-box) results - for comparison with jackknife
    std::vector<int64_t> global_counts; ///< Full-box histogram counts
    std::vector<double> global_pdf;     ///< Full-box PDF
    int64_t global_n_total;             ///< Total particles in full box

    // Jackknife results for histogram counts
    std::vector<double> counts_mean;    ///< Jackknife mean of counts (per sub-box)
    std::vector<double> counts_error;   ///< Jackknife standard error of counts

    // Jackknife results for PDF
    std::vector<double> pdf_mean;       ///< Jackknife mean of PDF
    std::vector<double> pdf_error;      ///< Jackknife standard error of PDF

    // Per-subbox data (for detailed analysis)
    std::vector<std::vector<int64_t>> subbox_counts;  ///< Counts per sub-box [subbox][bin]
    std::vector<std::vector<double>> subbox_pdf;      ///< PDF per sub-box [subbox][bin]
    std::vector<int64_t> subbox_n_total;              ///< Total particles per sub-box
};

/**
 * Result of jackknife conditional histogram computation.
 *
 * Contains jackknife histograms for all particles and for each
 * ORIGAMI morphology class (void, wall, filament, halo).
 */
struct JackknifeConditionalResult {
    JackknifeHistogramResult all;           ///< All particles
    JackknifeHistogramResult void_class;    ///< Void particles (morphology = 0)
    JackknifeHistogramResult wall_class;    ///< Wall particles (morphology = 1)
    JackknifeHistogramResult filament_class; ///< Filament particles (morphology = 2)
    JackknifeHistogramResult halo_class;    ///< Halo particles (morphology = 3)

    // Linear regime detection
    bool is_linear_regime;                  ///< True if field is in linear regime
    double f_void;                          ///< Void fraction (for reference)
};

/**
 * Compute jackknife histogram with uncertainty estimates.
 *
 * Divides the simulation box into sub-boxes and computes leave-one-out
 * jackknife estimates for both histogram counts and PDFs.
 *
 * The jackknife mean and error are computed as:
 *   X̄_jk = (1/n) Σ X̄_(i)
 *   σ_jk = sqrt((n-1)/n Σ [X̄_(i) - X̄_jk]²)
 *
 * where X̄_(i) is the mean of all sub-boxes except i.
 *
 * @param positions Particle positions, shape (n_particles, 3), Eulerian coordinates
 * @param values Values to histogram (e.g., overdensity 1+δ)
 * @param n_particles Number of particles
 * @param box_size Simulation box size
 * @param n_bins Number of histogram bins
 * @param range_min Minimum bin edge (must be > 0 for log bins)
 * @param range_max Maximum bin edge
 * @param log_bins If true, use logarithmic binning
 * @param n_subboxes_per_dim Sub-boxes per dimension (2 -> 8 total, 3 -> 27, etc.)
 * @param n_threads Number of OpenMP threads (0 = auto)
 * @return JackknifeHistogramResult with global and jackknife statistics
 */
JackknifeHistogramResult compute_jackknife_histogram(
    const double* positions,
    const double* values,
    int64_t n_particles,
    double box_size,
    int n_bins,
    double range_min,
    double range_max,
    bool log_bins = true,
    int n_subboxes_per_dim = 2,
    int n_threads = 0
);

/**
 * Compute jackknife conditional histograms by ORIGAMI morphology class.
 *
 * Computes jackknife histograms for all particles and separately for each
 * morphology class (void, wall, filament, halo).
 *
 * @param positions Particle positions, shape (n_particles, 3)
 * @param values Values to histogram (e.g., overdensity 1+δ)
 * @param morphology ORIGAMI morphology classification (0-3)
 * @param n_particles Number of particles
 * @param box_size Simulation box size
 * @param n_bins Number of histogram bins
 * @param range_min Minimum bin edge
 * @param range_max Maximum bin edge
 * @param log_bins If true, use logarithmic binning
 * @param n_subboxes_per_dim Sub-boxes per dimension
 * @param n_threads Number of OpenMP threads (0 = auto)
 * @return JackknifeConditionalResult with per-class jackknife statistics
 */
JackknifeConditionalResult compute_jackknife_conditional_histogram(
    const double* positions,
    const double* values,
    const uint8_t* morphology,
    int64_t n_particles,
    double box_size,
    int n_bins,
    double range_min,
    double range_max,
    bool log_bins = true,
    int n_subboxes_per_dim = 2,
    int n_threads = 0
);

} // namespace stats
} // namespace tessera
