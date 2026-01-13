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

} // namespace stats
} // namespace tessera
