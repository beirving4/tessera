#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace tessera {
namespace math {

/**
 * Binary search to find the interval containing x.
 * Returns index i such that xs[i] <= x < xs[i+1].
 */
inline int search(const std::vector<double>& xs, double x) {
    int lo = 0;
    int hi = static_cast<int>(xs.size()) - 1;
    
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xs[mid] > x) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

// ============================================================================
// 1D Interpolators
// ============================================================================

/**
 * 1D linear interpolation.
 */
class Linear {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    
    Linear() = default;
    
    Linear(const std::vector<double>& x, const std::vector<double>& y) 
        : xs(x), ys(y) {
        if (xs.size() != ys.size() || xs.size() < 2) {
            throw std::runtime_error("Linear: need at least 2 points with matching sizes");
        }
    }
    
    double eval(double x) const {
        int i = search(xs, x);
        i = std::max(0, std::min(i, static_cast<int>(xs.size()) - 2));
        
        double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
        return ys[i] + t * (ys[i + 1] - ys[i]);
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals) const {
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i]);
        }
        return result;
    }
};

/**
 * Cubic spline interpolation (natural boundary conditions).
 */
class Spline {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> y2;  // Second derivatives
    
    Spline() = default;
    
    Spline(const std::vector<double>& x, const std::vector<double>& y) 
        : xs(x), ys(y) {
        if (xs.size() != ys.size() || xs.size() < 3) {
            throw std::runtime_error("Spline: need at least 3 points with matching sizes");
        }
        compute_spline();
    }
    
    double eval(double x) const {
        int i = search(xs, x);
        i = std::max(0, std::min(i, static_cast<int>(xs.size()) - 2));
        
        double h = xs[i + 1] - xs[i];
        double a = (xs[i + 1] - x) / h;
        double b = (x - xs[i]) / h;
        
        return a * ys[i] + b * ys[i + 1] + 
               ((a * a * a - a) * y2[i] + (b * b * b - b) * y2[i + 1]) * (h * h) / 6.0;
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals) const {
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i]);
        }
        return result;
    }

private:
    void compute_spline() {
        int n = static_cast<int>(xs.size());
        y2.resize(n);
        std::vector<double> u(n - 1);
        
        // Natural spline: y2[0] = y2[n-1] = 0
        y2[0] = 0.0;
        u[0] = 0.0;
        
        // Decomposition
        for (int i = 1; i < n - 1; ++i) {
            double sig = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
            double p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            u[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) - 
                   (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
            u[i] = (6.0 * u[i] / (xs[i + 1] - xs[i - 1]) - sig * u[i - 1]) / p;
        }
        
        // Back substitution
        y2[n - 1] = 0.0;
        for (int k = n - 2; k >= 0; --k) {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }
    }
};

// ============================================================================
// 2D Interpolators
// ============================================================================

/**
 * 2D bilinear interpolation on a regular grid.
 */
class BiLinear {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> vals;  // Row-major: vals[iy * nx + ix]
    int nx = 0, ny = 0;
    
    BiLinear() = default;
    
    BiLinear(const std::vector<double>& x, const std::vector<double>& y,
             const std::vector<double>& v)
        : xs(x), ys(y), vals(v), nx(static_cast<int>(x.size())), ny(static_cast<int>(y.size())) {
        if (static_cast<int>(vals.size()) != nx * ny) {
            throw std::runtime_error("BiLinear: vals size must be nx * ny");
        }
    }
    
    double eval(double x, double y) const {
        int ix = search(xs, x);
        int iy = search(ys, y);
        ix = std::max(0, std::min(ix, nx - 2));
        iy = std::max(0, std::min(iy, ny - 2));
        
        double tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]);
        double ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy]);
        
        double v00 = vals[iy * nx + ix];
        double v10 = vals[iy * nx + ix + 1];
        double v01 = vals[(iy + 1) * nx + ix];
        double v11 = vals[(iy + 1) * nx + ix + 1];
        
        return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 +
               (1 - tx) * ty * v01 + tx * ty * v11;
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals,
                                  const std::vector<double>& yvals) const {
        if (xvals.size() != yvals.size()) {
            throw std::runtime_error("BiLinear::eval_all: x and y must have same size");
        }
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i], yvals[i]);
        }
        return result;
    }
};

/**
 * 2D bicubic interpolation on a regular grid.
 */
class BiCubic {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> vals;
    int nx = 0, ny = 0;
    
    BiCubic() = default;
    
    BiCubic(const std::vector<double>& x, const std::vector<double>& y,
            const std::vector<double>& v)
        : xs(x), ys(y), vals(v), nx(static_cast<int>(x.size())), ny(static_cast<int>(y.size())) {
        if (static_cast<int>(vals.size()) != nx * ny) {
            throw std::runtime_error("BiCubic: vals size must be nx * ny");
        }
    }
    
    double eval(double x, double y) const {
        int ix = search(xs, x);
        int iy = search(ys, y);
        ix = std::max(1, std::min(ix, nx - 3));
        iy = std::max(1, std::min(iy, ny - 3));
        
        double tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]);
        double ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy]);
        
        // Get 4x4 neighborhood
        double p[4][4];
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                p[j][i] = vals[(iy - 1 + j) * nx + (ix - 1 + i)];
            }
        }
        
        // Cubic interpolation in x for each row, then in y
        double col[4];
        for (int j = 0; j < 4; ++j) {
            col[j] = cubic_interp(tx, p[j][0], p[j][1], p[j][2], p[j][3]);
        }
        return cubic_interp(ty, col[0], col[1], col[2], col[3]);
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals,
                                  const std::vector<double>& yvals) const {
        if (xvals.size() != yvals.size()) {
            throw std::runtime_error("BiCubic::eval_all: x and y must have same size");
        }
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i], yvals[i]);
        }
        return result;
    }

private:
    static double cubic_interp(double t, double p0, double p1, double p2, double p3) {
        return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 +
               t * (3.0 * (p1 - p2) + p3 - p0)));
    }
};

// ============================================================================
// 3D Interpolators  
// ============================================================================

/**
 * 3D trilinear interpolation on a regular grid.
 */
class TriLinear {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<double> vals;  // vals[iz * ny * nx + iy * nx + ix]
    int nx = 0, ny = 0, nz = 0;
    
    TriLinear() = default;
    
    TriLinear(const std::vector<double>& x, const std::vector<double>& y,
              const std::vector<double>& z, const std::vector<double>& v)
        : xs(x), ys(y), zs(z), vals(v),
          nx(static_cast<int>(x.size())), ny(static_cast<int>(y.size())), nz(static_cast<int>(z.size())) {
        if (static_cast<int>(vals.size()) != nx * ny * nz) {
            throw std::runtime_error("TriLinear: vals size must be nx * ny * nz");
        }
    }
    
    double eval(double x, double y, double z) const {
        int ix = search(xs, x);
        int iy = search(ys, y);
        int iz = search(zs, z);
        ix = std::max(0, std::min(ix, nx - 2));
        iy = std::max(0, std::min(iy, ny - 2));
        iz = std::max(0, std::min(iz, nz - 2));
        
        double tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]);
        double ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy]);
        double tz = (z - zs[iz]) / (zs[iz + 1] - zs[iz]);
        
        // 8 corner values
        auto idx = [this](int i, int j, int k) { return k * ny * nx + j * nx + i; };
        
        double c000 = vals[idx(ix, iy, iz)];
        double c100 = vals[idx(ix + 1, iy, iz)];
        double c010 = vals[idx(ix, iy + 1, iz)];
        double c110 = vals[idx(ix + 1, iy + 1, iz)];
        double c001 = vals[idx(ix, iy, iz + 1)];
        double c101 = vals[idx(ix + 1, iy, iz + 1)];
        double c011 = vals[idx(ix, iy + 1, iz + 1)];
        double c111 = vals[idx(ix + 1, iy + 1, iz + 1)];
        
        // Interpolate
        double c00 = c000 * (1 - tx) + c100 * tx;
        double c01 = c001 * (1 - tx) + c101 * tx;
        double c10 = c010 * (1 - tx) + c110 * tx;
        double c11 = c011 * (1 - tx) + c111 * tx;
        
        double c0 = c00 * (1 - ty) + c10 * ty;
        double c1 = c01 * (1 - ty) + c11 * ty;
        
        return c0 * (1 - tz) + c1 * tz;
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals,
                                  const std::vector<double>& yvals,
                                  const std::vector<double>& zvals) const {
        if (xvals.size() != yvals.size() || xvals.size() != zvals.size()) {
            throw std::runtime_error("TriLinear::eval_all: x, y, z must have same size");
        }
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i], yvals[i], zvals[i]);
        }
        return result;
    }
};

/**
 * 3D tricubic interpolation on a regular grid.
 */
class TriCubic {
public:
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<double> vals;
    int nx = 0, ny = 0, nz = 0;
    
    TriCubic() = default;
    
    TriCubic(const std::vector<double>& x, const std::vector<double>& y,
             const std::vector<double>& z, const std::vector<double>& v)
        : xs(x), ys(y), zs(z), vals(v),
          nx(static_cast<int>(x.size())), ny(static_cast<int>(y.size())), nz(static_cast<int>(z.size())) {
        if (static_cast<int>(vals.size()) != nx * ny * nz) {
            throw std::runtime_error("TriCubic: vals size must be nx * ny * nz");
        }
    }
    
    double eval(double x, double y, double z) const {
        int ix = search(xs, x);
        int iy = search(ys, y);
        int iz = search(zs, z);
        ix = std::max(1, std::min(ix, nx - 3));
        iy = std::max(1, std::min(iy, ny - 3));
        iz = std::max(1, std::min(iz, nz - 3));
        
        double tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]);
        double ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy]);
        double tz = (z - zs[iz]) / (zs[iz + 1] - zs[iz]);
        
        auto idx = [this](int i, int j, int k) { return k * ny * nx + j * nx + i; };
        
        // Interpolate in z first, then y, then x
        double slices[4];
        for (int dz = 0; dz < 4; ++dz) {
            double rows[4];
            for (int dy = 0; dy < 4; ++dy) {
                double p[4];
                for (int dx = 0; dx < 4; ++dx) {
                    p[dx] = vals[idx(ix - 1 + dx, iy - 1 + dy, iz - 1 + dz)];
                }
                rows[dy] = cubic_interp(tx, p[0], p[1], p[2], p[3]);
            }
            slices[dz] = cubic_interp(ty, rows[0], rows[1], rows[2], rows[3]);
        }
        return cubic_interp(tz, slices[0], slices[1], slices[2], slices[3]);
    }
    
    std::vector<double> eval_all(const std::vector<double>& xvals,
                                  const std::vector<double>& yvals,
                                  const std::vector<double>& zvals) const {
        if (xvals.size() != yvals.size() || xvals.size() != zvals.size()) {
            throw std::runtime_error("TriCubic::eval_all: x, y, z must have same size");
        }
        std::vector<double> result(xvals.size());
        for (size_t i = 0; i < xvals.size(); ++i) {
            result[i] = eval(xvals[i], yvals[i], zvals[i]);
        }
        return result;
    }

private:
    static double cubic_interp(double t, double p0, double p1, double p2, double p3) {
        return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 +
               t * (3.0 * (p1 - p2) + p3 - p0)));
    }
};

} // namespace math
} // namespace tessera
