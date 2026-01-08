#pragma once

#include <vector>
#include <stdexcept>

namespace asymptotic_tetra {
namespace math {

/**
 * Compute numerical derivatives of discrete data points.
 * 
 * This function computes derivatives using finite difference formulas.
 * Supports 2nd and 4th order accuracy.
 * 
 * @param xs X coordinates (must be sorted, but not necessarily uniform)
 * @param ys Y values at each x
 * @param order Accuracy order: 2 or 4
 * @return Vector of derivative values dy/dx at each point
 */
inline std::vector<double> deriv(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    int order
) {
    int n = static_cast<int>(xs.size());
    
    if (static_cast<int>(ys.size()) != n) {
        throw std::runtime_error("deriv: xs and ys must have same length");
    }
    
    std::vector<double> out(n);
    
    if (order == 0) {
        // Identity (no derivative)
        for (int i = 0; i < n; ++i) {
            out[i] = ys[i];
        }
    } else if (order == 2) {
        // 2nd order central difference
        if (n < 3) {
            throw std::runtime_error("deriv: need at least 3 points for order 2");
        }
        
        // Interior points: central difference
        for (int i = 1; i < n - 1; ++i) {
            out[i] = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1]);
        }
        
        // Boundary points: forward/backward difference (2nd order)
        out[0] = (-3.0 * ys[0] + 4.0 * ys[1] - ys[2]) / (xs[2] - xs[0]);
        out[n - 1] = (3.0 * ys[n - 1] - 4.0 * ys[n - 2] + ys[n - 3]) / (xs[n - 1] - xs[n - 3]);
        
    } else if (order == 4) {
        // 4th order central difference
        if (n < 5) {
            throw std::runtime_error("deriv: need at least 5 points for order 4");
        }
        
        // Interior points
        for (int i = 2; i < n - 2; ++i) {
            out[i] = (-ys[i + 2] + 8.0 * ys[i + 1] - 8.0 * ys[i - 1] + ys[i - 2]) /
                     (3.0 * (xs[i + 2] - xs[i - 2]));
        }
        
        // Boundary points: 4th order one-sided formulas
        out[0] = (-3.0 * ys[4] + 16.0 * ys[3] - 36.0 * ys[2] + 48.0 * ys[1] - 25.0 * ys[0]) /
                 (3.0 * (xs[4] - xs[0]));
        
        out[1] = (-3.0 * ys[0] - 10.0 * ys[1] + 18.0 * ys[2] - 6.0 * ys[3] + ys[4]) /
                 (3.0 * (xs[4] - xs[0]));
        
        out[n - 2] = (-3.0 * ys[n - 1] - 10.0 * ys[n - 2] + 18.0 * ys[n - 3] - 
                      6.0 * ys[n - 4] + ys[n - 5]) /
                     (3.0 * (xs[n - 5] - xs[n - 1]));
        
        out[n - 1] = (-3.0 * ys[n - 5] + 16.0 * ys[n - 4] - 36.0 * ys[n - 3] + 
                      48.0 * ys[n - 2] - 25.0 * ys[n - 1]) /
                     (3.0 * (xs[n - 5] - xs[n - 1]));
    } else {
        throw std::runtime_error("deriv: order must be 0, 2, or 4");
    }
    
    return out;
}

/**
 * Compute numerical derivatives, storing in-place.
 * 
 * @param xs X coordinates
 * @param ys Y values
 * @param order Accuracy order: 2 or 4
 * @param out Output vector for derivatives
 */
inline void deriv_at(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    int order,
    std::vector<double>& out
) {
    int n = static_cast<int>(xs.size());
    
    if (static_cast<int>(ys.size()) != n || static_cast<int>(out.size()) != n) {
        throw std::runtime_error("deriv_at: xs, ys, and out must have same length");
    }
    
    auto result = deriv(xs, ys, order);
    out = std::move(result);
}

/**
 * Compute second derivative.
 * 
 * @param xs X coordinates
 * @param ys Y values
 * @param order Accuracy order for first derivative (applied twice)
 * @return Vector of second derivative values d²y/dx²
 */
inline std::vector<double> deriv2(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    int order = 2
) {
    auto dy = deriv(xs, ys, order);
    return deriv(xs, dy, order);
}

/**
 * Compute gradient of uniformly spaced data.
 * This is a simpler version for uniform grids.
 * 
 * @param ys Y values
 * @param dx Spacing between x points
 * @return Gradient values
 */
inline std::vector<double> gradient(const std::vector<double>& ys, double dx) {
    int n = static_cast<int>(ys.size());
    if (n < 2) return {};
    
    std::vector<double> out(n);
    
    // Central differences for interior
    for (int i = 1; i < n - 1; ++i) {
        out[i] = (ys[i + 1] - ys[i - 1]) / (2.0 * dx);
    }
    
    // One-sided differences at boundaries
    out[0] = (ys[1] - ys[0]) / dx;
    out[n - 1] = (ys[n - 1] - ys[n - 2]) / dx;
    
    return out;
}

} // namespace math
} // namespace asymptotic_tetra
