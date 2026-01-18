#include "stats/histogram.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tessera {
namespace stats {

namespace {

// Find min and max of positive values in parallel
std::pair<double, double> find_positive_range(const double* data, int64_t n_values, int n_threads) {
    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();
    
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    #pragma omp parallel
    {
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        
        #pragma omp for nowait
        for (int64_t i = 0; i < n_values; ++i) {
            double val = data[i];
            if (val > 0) {
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
        }
        
        #pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }
#else
    for (int64_t i = 0; i < n_values; ++i) {
        double val = data[i];
        if (val > 0) {
            if (val < global_min) global_min = val;
            if (val > global_max) global_max = val;
        }
    }
#endif
    
    return {global_min, global_max};
}

// Find absolute min and max in parallel
std::pair<double, double> find_range(const double* data, int64_t n_values, int n_threads) {
    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();
    
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    #pragma omp parallel
    {
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        
        #pragma omp for nowait
        for (int64_t i = 0; i < n_values; ++i) {
            double val = data[i];
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }
        
        #pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }
#else
    for (int64_t i = 0; i < n_values; ++i) {
        double val = data[i];
        if (val < global_min) global_min = val;
        if (val > global_max) global_max = val;
    }
#endif
    
    return {global_min, global_max};
}

} // anonymous namespace


HistogramResult histogram(
    const double* data,
    int64_t n_values,
    int n_bins,
    double range_min,
    double range_max,
    int n_threads
) {
    if (n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    
    // Auto-detect range if not specified
    if (range_min >= range_max) {
        auto [dmin, dmax] = find_range(data, n_values, n_threads);
        range_min = dmin;
        range_max = dmax;
        // Add small padding to include max value
        range_max += (range_max - range_min) * 1e-10;
    }
    
    HistogramResult result;
    result.counts.resize(n_bins, 0);
    result.bin_edges.resize(n_bins + 1);
    result.n_below = 0;
    result.n_above = 0;
    result.n_total = n_values;
    
    // Generate bin edges
    double bin_width = (range_max - range_min) / n_bins;
    for (int i = 0; i <= n_bins; ++i) {
        result.bin_edges[i] = range_min + i * bin_width;
    }
    
    // Compute histogram with parallel reduction
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    
    // Thread-local histograms
    std::vector<std::vector<int64_t>> local_counts(actual_threads, std::vector<int64_t>(n_bins, 0));
    std::vector<int64_t> local_below(actual_threads, 0);
    std::vector<int64_t> local_above(actual_threads, 0);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_counts = local_counts[tid];
        int64_t my_below = 0, my_above = 0;
        
        #pragma omp for nowait
        for (int64_t i = 0; i < n_values; ++i) {
            double val = data[i];
            if (val < range_min) {
                ++my_below;
            } else if (val >= range_max) {
                ++my_above;
            } else {
                int bin = static_cast<int>((val - range_min) / bin_width);
                bin = std::min(bin, n_bins - 1);  // Handle edge case
                ++my_counts[bin];
            }
        }
        
        local_below[tid] = my_below;
        local_above[tid] = my_above;
    }
    
    // Reduce
    for (int t = 0; t < actual_threads; ++t) {
        result.n_below += local_below[t];
        result.n_above += local_above[t];
        for (int b = 0; b < n_bins; ++b) {
            result.counts[b] += local_counts[t][b];
        }
    }
#else
    double inv_bin_width = 1.0 / bin_width;
    for (int64_t i = 0; i < n_values; ++i) {
        double val = data[i];
        if (val < range_min) {
            ++result.n_below;
        } else if (val >= range_max) {
            ++result.n_above;
        } else {
            int bin = static_cast<int>((val - range_min) * inv_bin_width);
            bin = std::min(bin, n_bins - 1);
            ++result.counts[bin];
        }
    }
#endif
    
    return result;
}


HistogramResult histogram_log(
    const double* data,
    int64_t n_values,
    int n_bins,
    double range_min,
    double range_max,
    int n_threads
) {
    if (n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    
    // Auto-detect range if not specified (positive values only)
    if (range_min <= 0 || range_max <= range_min) {
        auto [dmin, dmax] = find_positive_range(data, n_values, n_threads);
        if (dmin >= dmax) {
            throw std::runtime_error("No positive values found for log histogram");
        }
        if (range_min <= 0) range_min = dmin;
        if (range_max <= range_min) {
            range_max = dmax * 1.001;  // Small padding
        }
    }
    
    double log_min = std::log10(range_min);
    double log_max = std::log10(range_max);
    double log_bin_width = (log_max - log_min) / n_bins;
    
    HistogramResult result;
    result.counts.resize(n_bins, 0);
    result.bin_edges.resize(n_bins + 1);
    result.n_below = 0;
    result.n_above = 0;
    result.n_total = n_values;
    
    // Generate log-spaced bin edges
    for (int i = 0; i <= n_bins; ++i) {
        result.bin_edges[i] = std::pow(10.0, log_min + i * log_bin_width);
    }
    
    // Compute histogram
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    
    std::vector<std::vector<int64_t>> local_counts(actual_threads, std::vector<int64_t>(n_bins, 0));
    std::vector<int64_t> local_below(actual_threads, 0);
    std::vector<int64_t> local_above(actual_threads, 0);
    
    double inv_log_bin_width = 1.0 / log_bin_width;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_counts = local_counts[tid];
        int64_t my_below = 0, my_above = 0;
        
        #pragma omp for nowait
        for (int64_t i = 0; i < n_values; ++i) {
            double val = data[i];
            if (val <= 0 || val < range_min) {
                ++my_below;
            } else if (val >= range_max) {
                ++my_above;
            } else {
                double log_val = std::log10(val);
                int bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
                ++my_counts[bin];
            }
        }
        
        local_below[tid] = my_below;
        local_above[tid] = my_above;
    }
    
    // Reduce
    for (int t = 0; t < actual_threads; ++t) {
        result.n_below += local_below[t];
        result.n_above += local_above[t];
        for (int b = 0; b < n_bins; ++b) {
            result.counts[b] += local_counts[t][b];
        }
    }
#else
    double inv_log_bin_width = 1.0 / log_bin_width;
    for (int64_t i = 0; i < n_values; ++i) {
        double val = data[i];
        if (val <= 0 || val < range_min) {
            ++result.n_below;
        } else if (val >= range_max) {
            ++result.n_above;
        } else {
            double log_val = std::log10(val);
            int bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
            bin = std::max(0, std::min(bin, n_bins - 1));
            ++result.counts[bin];
        }
    }
#endif
    
    return result;
}


WeightedHistogramResult histogram_log_weighted(
    const double* data,
    const double* weights,
    int64_t n_values,
    int n_bins,
    double range_min,
    double range_max,
    int n_threads
) {
    if (n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    if (range_min <= 0 || range_max <= range_min) {
        throw std::invalid_argument("range_min must be > 0 and range_max > range_min");
    }
    
    double log_min = std::log10(range_min);
    double log_max = std::log10(range_max);
    double log_bin_width = (log_max - log_min) / n_bins;
    
    WeightedHistogramResult result;
    result.counts.resize(n_bins, 0.0);
    result.bin_edges.resize(n_bins + 1);
    result.sum_below = 0.0;
    result.sum_above = 0.0;
    result.sum_total = 0.0;
    result.n_total = n_values;
    
    // Generate log-spaced bin edges
    for (int i = 0; i <= n_bins; ++i) {
        result.bin_edges[i] = std::pow(10.0, log_min + i * log_bin_width);
    }
    
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    
    std::vector<std::vector<double>> local_counts(actual_threads, std::vector<double>(n_bins, 0.0));
    std::vector<double> local_below(actual_threads, 0.0);
    std::vector<double> local_above(actual_threads, 0.0);
    std::vector<double> local_total(actual_threads, 0.0);
    
    double inv_log_bin_width = 1.0 / log_bin_width;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_counts = local_counts[tid];
        double my_below = 0.0, my_above = 0.0, my_total = 0.0;
        
        #pragma omp for nowait
        for (int64_t i = 0; i < n_values; ++i) {
            double val = data[i];
            double w = weights[i];
            my_total += w;
            
            if (val <= 0 || val < range_min) {
                my_below += w;
            } else if (val >= range_max) {
                my_above += w;
            } else {
                double log_val = std::log10(val);
                int bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
                my_counts[bin] += w;
            }
        }
        
        local_below[tid] = my_below;
        local_above[tid] = my_above;
        local_total[tid] = my_total;
    }
    
    // Reduce
    for (int t = 0; t < actual_threads; ++t) {
        result.sum_below += local_below[t];
        result.sum_above += local_above[t];
        result.sum_total += local_total[t];
        for (int b = 0; b < n_bins; ++b) {
            result.counts[b] += local_counts[t][b];
        }
    }
#else
    double inv_log_bin_width = 1.0 / log_bin_width;
    for (int64_t i = 0; i < n_values; ++i) {
        double val = data[i];
        double w = weights[i];
        result.sum_total += w;
        
        if (val <= 0 || val < range_min) {
            result.sum_below += w;
        } else if (val >= range_max) {
            result.sum_above += w;
        } else {
            double log_val = std::log10(val);
            int bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
            bin = std::max(0, std::min(bin, n_bins - 1));
            result.counts[bin] += w;
        }
    }
#endif
    
    return result;
}


std::vector<double> histogram_to_pdf(const HistogramResult& hist) {
    int n_bins = static_cast<int>(hist.counts.size());
    std::vector<double> pdf(n_bins);
    
    // Count total in bins (exclude under/overflow)
    int64_t total_in_bins = 0;
    for (int i = 0; i < n_bins; ++i) {
        total_in_bins += hist.counts[i];
    }
    
    if (total_in_bins == 0) {
        return pdf;  // All zeros
    }
    
    // PDF: P(x) = count / (total * bin_width)
    // such that integral P(x) dx = 1
    double inv_total = 1.0 / static_cast<double>(total_in_bins);
    for (int i = 0; i < n_bins; ++i) {
        double bin_width = hist.bin_edges[i + 1] - hist.bin_edges[i];
        pdf[i] = static_cast<double>(hist.counts[i]) * inv_total / bin_width;
    }
    
    return pdf;
}


std::vector<double> bin_centers(const std::vector<double>& bin_edges, bool log_space) {
    if (bin_edges.size() < 2) {
        return {};
    }

    int n_bins = static_cast<int>(bin_edges.size()) - 1;
    std::vector<double> centers(n_bins);

    if (log_space) {
        // Geometric mean for log-spaced bins
        for (int i = 0; i < n_bins; ++i) {
            centers[i] = std::sqrt(bin_edges[i] * bin_edges[i + 1]);
        }
    } else {
        // Arithmetic mean
        for (int i = 0; i < n_bins; ++i) {
            centers[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1]);
        }
    }

    return centers;
}


// ============================================================================
// Jackknife histogram implementation
// ============================================================================

namespace {

// Assign particle to sub-box index based on position
inline int get_subbox_index(double x, double y, double z, double box_size, int n_per_dim) {
    // Wrap coordinates to [0, box_size)
    while (x < 0) x += box_size;
    while (x >= box_size) x -= box_size;
    while (y < 0) y += box_size;
    while (y >= box_size) y -= box_size;
    while (z < 0) z += box_size;
    while (z >= box_size) z -= box_size;

    double subbox_size = box_size / n_per_dim;
    int ix = std::min(static_cast<int>(x / subbox_size), n_per_dim - 1);
    int iy = std::min(static_cast<int>(y / subbox_size), n_per_dim - 1);
    int iz = std::min(static_cast<int>(z / subbox_size), n_per_dim - 1);

    return ix + iy * n_per_dim + iz * n_per_dim * n_per_dim;
}

// Convert histogram counts to PDF
std::vector<double> counts_to_pdf(const std::vector<int64_t>& counts,
                                   const std::vector<double>& bin_edges) {
    int n_bins = static_cast<int>(counts.size());
    std::vector<double> pdf(n_bins, 0.0);

    int64_t total = 0;
    for (int i = 0; i < n_bins; ++i) {
        total += counts[i];
    }

    if (total == 0) return pdf;

    double inv_total = 1.0 / static_cast<double>(total);
    for (int i = 0; i < n_bins; ++i) {
        double bin_width = bin_edges[i + 1] - bin_edges[i];
        pdf[i] = static_cast<double>(counts[i]) * inv_total / bin_width;
    }

    return pdf;
}

// Compute jackknife statistics from sub-box measurements
void compute_jackknife_stats(
    const std::vector<std::vector<double>>& subbox_values,  // [subbox][bin]
    int n_subboxes,
    int n_bins,
    std::vector<double>& jk_mean,
    std::vector<double>& jk_error
) {
    jk_mean.resize(n_bins, 0.0);
    jk_error.resize(n_bins, 0.0);

    if (n_subboxes < 2) {
        // Can't compute jackknife with < 2 subboxes
        for (int b = 0; b < n_bins; ++b) {
            if (n_subboxes == 1) {
                jk_mean[b] = subbox_values[0][b];
            }
        }
        return;
    }

    // For each bin, compute leave-one-out estimates
    for (int b = 0; b < n_bins; ++b) {
        // Compute sum of all subbox values for this bin
        double sum_all = 0.0;
        for (int s = 0; s < n_subboxes; ++s) {
            sum_all += subbox_values[s][b];
        }

        // Compute leave-one-out estimates: X̄_(i) = (sum - X_i) / (n-1)
        std::vector<double> leave_one_out(n_subboxes);
        for (int i = 0; i < n_subboxes; ++i) {
            leave_one_out[i] = (sum_all - subbox_values[i][b]) / (n_subboxes - 1);
        }

        // Jackknife mean: X̄_jk = (1/n) Σ X̄_(i)
        double jk_sum = 0.0;
        for (int i = 0; i < n_subboxes; ++i) {
            jk_sum += leave_one_out[i];
        }
        jk_mean[b] = jk_sum / n_subboxes;

        // Jackknife error: σ_jk = sqrt((n-1)/n Σ [X̄_(i) - X̄_jk]²)
        double variance_sum = 0.0;
        for (int i = 0; i < n_subboxes; ++i) {
            double diff = leave_one_out[i] - jk_mean[b];
            variance_sum += diff * diff;
        }
        jk_error[b] = std::sqrt((static_cast<double>(n_subboxes - 1) / n_subboxes) * variance_sum);
    }
}

} // anonymous namespace


JackknifeHistogramResult compute_jackknife_histogram(
    const double* positions,
    const double* values,
    int64_t n_particles,
    double box_size,
    int n_bins,
    double range_min,
    double range_max,
    bool log_bins,
    int n_subboxes_per_dim,
    int n_threads
) {
    if (n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    if (n_subboxes_per_dim <= 0) {
        throw std::invalid_argument("n_subboxes_per_dim must be positive");
    }

    int n_subboxes = n_subboxes_per_dim * n_subboxes_per_dim * n_subboxes_per_dim;

    // Auto-detect range if needed
    if (log_bins) {
        if (range_min <= 0 || range_max <= range_min) {
            auto [dmin, dmax] = find_positive_range(values, n_particles, n_threads);
            if (dmin >= dmax) {
                throw std::runtime_error("No positive values found for log histogram");
            }
            if (range_min <= 0) range_min = dmin;
            if (range_max <= range_min) range_max = dmax * 1.001;
        }
    } else {
        if (range_min >= range_max) {
            auto [dmin, dmax] = find_range(values, n_particles, n_threads);
            range_min = dmin;
            range_max = dmax + (dmax - dmin) * 1e-10;
        }
    }

    JackknifeHistogramResult result;
    result.n_bins = n_bins;
    result.n_subboxes = n_subboxes;
    result.bin_edges.resize(n_bins + 1);
    result.global_n_total = n_particles;

    // Generate bin edges
    if (log_bins) {
        double log_min = std::log10(range_min);
        double log_max = std::log10(range_max);
        double log_bin_width = (log_max - log_min) / n_bins;
        for (int i = 0; i <= n_bins; ++i) {
            result.bin_edges[i] = std::pow(10.0, log_min + i * log_bin_width);
        }
    } else {
        double bin_width = (range_max - range_min) / n_bins;
        for (int i = 0; i <= n_bins; ++i) {
            result.bin_edges[i] = range_min + i * bin_width;
        }
    }

    // Compute bin centers
    result.bin_centers = bin_centers(result.bin_edges, log_bins);

    // Initialize per-subbox storage
    result.subbox_counts.resize(n_subboxes, std::vector<int64_t>(n_bins, 0));
    result.subbox_n_total.resize(n_subboxes, 0);

    // Compute bin parameters for fast lookup
    double log_min = log_bins ? std::log10(range_min) : 0.0;
    double log_max = log_bins ? std::log10(range_max) : 0.0;
    double log_bin_width = log_bins ? (log_max - log_min) / n_bins : 0.0;
    double lin_bin_width = log_bins ? 0.0 : (range_max - range_min) / n_bins;
    double inv_log_bin_width = log_bins ? 1.0 / log_bin_width : 0.0;
    double inv_lin_bin_width = log_bins ? 0.0 : 1.0 / lin_bin_width;

#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    // Thread-local storage for subbox counts
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    std::vector<std::vector<std::vector<int64_t>>> thread_subbox_counts(
        actual_threads, std::vector<std::vector<int64_t>>(n_subboxes, std::vector<int64_t>(n_bins, 0))
    );
    std::vector<std::vector<int64_t>> thread_subbox_n_total(
        actual_threads, std::vector<int64_t>(n_subboxes, 0)
    );

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_counts = thread_subbox_counts[tid];
        auto& my_n_total = thread_subbox_n_total[tid];

        #pragma omp for nowait
        for (int64_t i = 0; i < n_particles; ++i) {
            double px = positions[i * 3 + 0];
            double py = positions[i * 3 + 1];
            double pz = positions[i * 3 + 2];
            double val = values[i];

            int subbox = get_subbox_index(px, py, pz, box_size, n_subboxes_per_dim);
            my_n_total[subbox]++;

            // Bin the value
            int bin = -1;
            if (log_bins) {
                if (val > 0 && val >= range_min && val < range_max) {
                    double log_val = std::log10(val);
                    bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                    bin = std::max(0, std::min(bin, n_bins - 1));
                }
            } else {
                if (val >= range_min && val < range_max) {
                    bin = static_cast<int>((val - range_min) * inv_lin_bin_width);
                    bin = std::max(0, std::min(bin, n_bins - 1));
                }
            }

            if (bin >= 0) {
                my_counts[subbox][bin]++;
            }
        }
    }

    // Reduce thread-local results
    for (int t = 0; t < actual_threads; ++t) {
        for (int s = 0; s < n_subboxes; ++s) {
            result.subbox_n_total[s] += thread_subbox_n_total[t][s];
            for (int b = 0; b < n_bins; ++b) {
                result.subbox_counts[s][b] += thread_subbox_counts[t][s][b];
            }
        }
    }
#else
    for (int64_t i = 0; i < n_particles; ++i) {
        double px = positions[i * 3 + 0];
        double py = positions[i * 3 + 1];
        double pz = positions[i * 3 + 2];
        double val = values[i];

        int subbox = get_subbox_index(px, py, pz, box_size, n_subboxes_per_dim);
        result.subbox_n_total[subbox]++;

        int bin = -1;
        if (log_bins) {
            if (val > 0 && val >= range_min && val < range_max) {
                double log_val = std::log10(val);
                bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
            }
        } else {
            if (val >= range_min && val < range_max) {
                bin = static_cast<int>((val - range_min) * inv_lin_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
            }
        }

        if (bin >= 0) {
            result.subbox_counts[subbox][bin]++;
        }
    }
#endif

    // Compute global counts by summing subbox counts
    result.global_counts.resize(n_bins, 0);
    for (int s = 0; s < n_subboxes; ++s) {
        for (int b = 0; b < n_bins; ++b) {
            result.global_counts[b] += result.subbox_counts[s][b];
        }
    }

    // Compute global PDF
    result.global_pdf = counts_to_pdf(result.global_counts, result.bin_edges);

    // Compute per-subbox PDFs
    result.subbox_pdf.resize(n_subboxes);
    for (int s = 0; s < n_subboxes; ++s) {
        result.subbox_pdf[s] = counts_to_pdf(result.subbox_counts[s], result.bin_edges);
    }

    // Convert counts to doubles for jackknife computation
    std::vector<std::vector<double>> subbox_counts_double(n_subboxes, std::vector<double>(n_bins));
    for (int s = 0; s < n_subboxes; ++s) {
        for (int b = 0; b < n_bins; ++b) {
            subbox_counts_double[s][b] = static_cast<double>(result.subbox_counts[s][b]);
        }
    }

    // Compute jackknife statistics for counts
    compute_jackknife_stats(subbox_counts_double, n_subboxes, n_bins,
                            result.counts_mean, result.counts_error);

    // Compute jackknife statistics for PDFs
    compute_jackknife_stats(result.subbox_pdf, n_subboxes, n_bins,
                            result.pdf_mean, result.pdf_error);

    return result;
}


JackknifeConditionalResult compute_jackknife_conditional_histogram(
    const double* positions,
    const double* values,
    const uint8_t* morphology,
    int64_t n_particles,
    double box_size,
    int n_bins,
    double range_min,
    double range_max,
    bool log_bins,
    int n_subboxes_per_dim,
    int n_threads
) {
    if (n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    if (n_subboxes_per_dim <= 0) {
        throw std::invalid_argument("n_subboxes_per_dim must be positive");
    }

    int n_subboxes = n_subboxes_per_dim * n_subboxes_per_dim * n_subboxes_per_dim;
    const int n_classes = 4;  // void, wall, filament, halo

    // Auto-detect range if needed
    if (log_bins) {
        if (range_min <= 0 || range_max <= range_min) {
            auto [dmin, dmax] = find_positive_range(values, n_particles, n_threads);
            if (dmin >= dmax) {
                throw std::runtime_error("No positive values found for log histogram");
            }
            if (range_min <= 0) range_min = dmin;
            if (range_max <= range_min) range_max = dmax * 1.001;
        }
    } else {
        if (range_min >= range_max) {
            auto [dmin, dmax] = find_range(values, n_particles, n_threads);
            range_min = dmin;
            range_max = dmax + (dmax - dmin) * 1e-10;
        }
    }

    // Generate bin edges
    std::vector<double> bin_edges(n_bins + 1);
    if (log_bins) {
        double log_min = std::log10(range_min);
        double log_max = std::log10(range_max);
        double log_bin_width = (log_max - log_min) / n_bins;
        for (int i = 0; i <= n_bins; ++i) {
            bin_edges[i] = std::pow(10.0, log_min + i * log_bin_width);
        }
    } else {
        double bin_width = (range_max - range_min) / n_bins;
        for (int i = 0; i <= n_bins; ++i) {
            bin_edges[i] = range_min + i * bin_width;
        }
    }

    std::vector<double> bin_ctrs = bin_centers(bin_edges, log_bins);

    // Initialize per-subbox, per-class storage
    // [class][subbox][bin]
    std::vector<std::vector<std::vector<int64_t>>> class_subbox_counts(
        n_classes + 1,  // +1 for "all"
        std::vector<std::vector<int64_t>>(n_subboxes, std::vector<int64_t>(n_bins, 0))
    );
    std::vector<std::vector<int64_t>> class_subbox_n_total(
        n_classes + 1,
        std::vector<int64_t>(n_subboxes, 0)
    );

    // Count particles per class (for linear regime detection)
    std::vector<int64_t> class_counts(n_classes, 0);

    // Bin parameters
    double log_min = log_bins ? std::log10(range_min) : 0.0;
    double log_bin_width = log_bins ? (std::log10(range_max) - log_min) / n_bins : 0.0;
    double lin_bin_width = log_bins ? 0.0 : (range_max - range_min) / n_bins;
    double inv_log_bin_width = log_bins ? 1.0 / log_bin_width : 0.0;
    double inv_lin_bin_width = log_bins ? 0.0 : 1.0 / lin_bin_width;

#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    // Thread-local storage
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> thread_class_subbox_counts(
        actual_threads,
        std::vector<std::vector<std::vector<int64_t>>>(
            n_classes + 1,
            std::vector<std::vector<int64_t>>(n_subboxes, std::vector<int64_t>(n_bins, 0))
        )
    );
    std::vector<std::vector<std::vector<int64_t>>> thread_class_subbox_n_total(
        actual_threads,
        std::vector<std::vector<int64_t>>(n_classes + 1, std::vector<int64_t>(n_subboxes, 0))
    );
    std::vector<std::vector<int64_t>> thread_class_counts(
        actual_threads, std::vector<int64_t>(n_classes, 0)
    );

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_counts = thread_class_subbox_counts[tid];
        auto& my_n_total = thread_class_subbox_n_total[tid];
        auto& my_class_counts = thread_class_counts[tid];

        #pragma omp for nowait
        for (int64_t i = 0; i < n_particles; ++i) {
            double px = positions[i * 3 + 0];
            double py = positions[i * 3 + 1];
            double pz = positions[i * 3 + 2];
            double val = values[i];
            int cls = morphology[i];
            if (cls >= n_classes) cls = n_classes - 1;  // Clamp

            my_class_counts[cls]++;

            int subbox = get_subbox_index(px, py, pz, box_size, n_subboxes_per_dim);
            my_n_total[n_classes][subbox]++;  // "all" class
            my_n_total[cls][subbox]++;

            // Bin the value
            int bin = -1;
            if (log_bins) {
                if (val > 0 && val >= range_min && val < range_max) {
                    double log_val = std::log10(val);
                    bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                    bin = std::max(0, std::min(bin, n_bins - 1));
                }
            } else {
                if (val >= range_min && val < range_max) {
                    bin = static_cast<int>((val - range_min) * inv_lin_bin_width);
                    bin = std::max(0, std::min(bin, n_bins - 1));
                }
            }

            if (bin >= 0) {
                my_counts[n_classes][subbox][bin]++;  // "all" class
                my_counts[cls][subbox][bin]++;
            }
        }
    }

    // Reduce
    for (int t = 0; t < actual_threads; ++t) {
        for (int c = 0; c < n_classes; ++c) {
            class_counts[c] += thread_class_counts[t][c];
        }
        for (int c = 0; c <= n_classes; ++c) {
            for (int s = 0; s < n_subboxes; ++s) {
                class_subbox_n_total[c][s] += thread_class_subbox_n_total[t][c][s];
                for (int b = 0; b < n_bins; ++b) {
                    class_subbox_counts[c][s][b] += thread_class_subbox_counts[t][c][s][b];
                }
            }
        }
    }
#else
    for (int64_t i = 0; i < n_particles; ++i) {
        double px = positions[i * 3 + 0];
        double py = positions[i * 3 + 1];
        double pz = positions[i * 3 + 2];
        double val = values[i];
        int cls = morphology[i];
        if (cls >= n_classes) cls = n_classes - 1;

        class_counts[cls]++;

        int subbox = get_subbox_index(px, py, pz, box_size, n_subboxes_per_dim);
        class_subbox_n_total[n_classes][subbox]++;
        class_subbox_n_total[cls][subbox]++;

        int bin = -1;
        if (log_bins) {
            if (val > 0 && val >= range_min && val < range_max) {
                double log_val = std::log10(val);
                bin = static_cast<int>((log_val - log_min) * inv_log_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
            }
        } else {
            if (val >= range_min && val < range_max) {
                bin = static_cast<int>((val - range_min) * inv_lin_bin_width);
                bin = std::max(0, std::min(bin, n_bins - 1));
            }
        }

        if (bin >= 0) {
            class_subbox_counts[n_classes][subbox][bin]++;
            class_subbox_counts[cls][subbox][bin]++;
        }
    }
#endif

    // Helper to build JackknifeHistogramResult from subbox data
    auto build_result = [&](int class_idx) -> JackknifeHistogramResult {
        JackknifeHistogramResult res;
        res.n_bins = n_bins;
        res.n_subboxes = n_subboxes;
        res.bin_edges = bin_edges;
        res.bin_centers = bin_ctrs;
        res.subbox_counts = class_subbox_counts[class_idx];
        res.subbox_n_total = class_subbox_n_total[class_idx];

        // Global counts
        res.global_counts.resize(n_bins, 0);
        res.global_n_total = 0;
        for (int s = 0; s < n_subboxes; ++s) {
            res.global_n_total += res.subbox_n_total[s];
            for (int b = 0; b < n_bins; ++b) {
                res.global_counts[b] += res.subbox_counts[s][b];
            }
        }

        // Global PDF
        res.global_pdf = counts_to_pdf(res.global_counts, bin_edges);

        // Per-subbox PDFs
        res.subbox_pdf.resize(n_subboxes);
        for (int s = 0; s < n_subboxes; ++s) {
            res.subbox_pdf[s] = counts_to_pdf(res.subbox_counts[s], bin_edges);
        }

        // Jackknife for counts
        std::vector<std::vector<double>> counts_double(n_subboxes, std::vector<double>(n_bins));
        for (int s = 0; s < n_subboxes; ++s) {
            for (int b = 0; b < n_bins; ++b) {
                counts_double[s][b] = static_cast<double>(res.subbox_counts[s][b]);
            }
        }
        compute_jackknife_stats(counts_double, n_subboxes, n_bins,
                                res.counts_mean, res.counts_error);

        // Jackknife for PDFs
        compute_jackknife_stats(res.subbox_pdf, n_subboxes, n_bins,
                                res.pdf_mean, res.pdf_error);

        return res;
    };

    JackknifeConditionalResult result;
    result.all = build_result(n_classes);      // "all" is at index n_classes
    result.void_class = build_result(0);
    result.wall_class = build_result(1);
    result.filament_class = build_result(2);
    result.halo_class = build_result(3);

    // Linear regime detection
    result.f_void = static_cast<double>(class_counts[0]) / static_cast<double>(n_particles);
    result.is_linear_regime = (result.f_void >= 0.99);

    return result;
}

} // namespace stats
} // namespace tessera
