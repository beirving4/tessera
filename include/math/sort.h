#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

namespace tessera {
namespace math {

/**
 * Reverse a vector in place.
 * 
 * @param xs Vector to reverse
 * @return Reference to the reversed vector
 */
inline std::vector<double>& reverse(std::vector<double>& xs) {
    int n1 = static_cast<int>(xs.size()) - 1;
    int n2 = static_cast<int>(xs.size()) / 2;
    for (int i = 0; i < n2; ++i) {
        std::swap(xs[i], xs[n1 - i]);
    }
    return xs;
}

/**
 * Shell sort implementation for sorting indices by values.
 */
inline void shell_sort(std::vector<double>& xs) {
    int n = static_cast<int>(xs.size());
    int gap = n / 2;
    
    while (gap > 0) {
        for (int i = gap; i < n; ++i) {
            double temp = xs[i];
            int j = i;
            while (j >= gap && xs[j - gap] > temp) {
                xs[j] = xs[j - gap];
                j -= gap;
            }
            xs[j] = temp;
        }
        gap /= 2;
    }
}

/**
 * Partition for quickselect.
 */
inline int partition(std::vector<double>& xs, int lo, int hi) {
    double pivot = xs[hi];
    int i = lo;
    for (int j = lo; j < hi; ++j) {
        if (xs[j] < pivot) {
            std::swap(xs[i], xs[j]);
            i++;
        }
    }
    std::swap(xs[i], xs[hi]);
    return i;
}

/**
 * Quickselect to find the k-th smallest element.
 * Modifies the input array.
 * 
 * @param xs Vector of values
 * @param k Index of element to find (0-based)
 * @return k-th smallest element
 */
inline double quickselect(std::vector<double>& xs, int k) {
    int lo = 0;
    int hi = static_cast<int>(xs.size()) - 1;
    
    while (lo < hi) {
        int p = partition(xs, lo, hi);
        if (p == k) {
            return xs[k];
        } else if (p < k) {
            lo = p + 1;
        } else {
            hi = p - 1;
        }
    }
    return xs[lo];
}

/**
 * Find the median of a vector.
 * Note: This modifies the input vector!
 * 
 * @param xs Vector of values (will be partially reordered)
 * @return Median value
 */
inline double median(std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }
    
    int n = static_cast<int>(xs.size());
    int mid = n / 2;
    
    if (n % 2 == 1) {
        return quickselect(xs, mid);
    } else {
        // For even length, average the two middle elements
        double a = quickselect(xs, mid - 1);
        double b = quickselect(xs, mid);
        return (a + b) / 2.0;
    }
}

/**
 * Find the median of a vector (non-modifying version).
 * 
 * @param xs Vector of values
 * @return Median value
 */
inline double median_copy(const std::vector<double>& xs) {
    std::vector<double> copy = xs;
    return median(copy);
}

/**
 * Get indices that would sort the array.
 * 
 * @param xs Vector of values
 * @return Vector of indices such that xs[indices[i]] is sorted
 */
inline std::vector<int> argsort(const std::vector<double>& xs) {
    std::vector<int> indices(xs.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    
    std::sort(indices.begin(), indices.end(),
        [&xs](int a, int b) { return xs[a] < xs[b]; });
    
    return indices;
}

/**
 * Get indices that would sort the array in descending order.
 */
inline std::vector<int> argsort_descending(const std::vector<double>& xs) {
    std::vector<int> indices(xs.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    
    std::sort(indices.begin(), indices.end(),
        [&xs](int a, int b) { return xs[a] > xs[b]; });
    
    return indices;
}

/**
 * Percentile calculation.
 * 
 * @param xs Vector of values (will be partially reordered)
 * @param p Percentile (0.0 to 1.0)
 * @return Value at the given percentile
 */
inline double percentile(std::vector<double>& xs, double p) {
    if (xs.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(xs.begin(), xs.end());
    if (p >= 1.0) return *std::max_element(xs.begin(), xs.end());
    
    double idx = p * (xs.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = lo + 1;
    double frac = idx - lo;
    
    if (hi >= static_cast<int>(xs.size())) {
        return quickselect(xs, lo);
    }
    
    double val_lo = quickselect(xs, lo);
    double val_hi = quickselect(xs, hi);
    return val_lo + frac * (val_hi - val_lo);
}

} // namespace math
} // namespace tessera
