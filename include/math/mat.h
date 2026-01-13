#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace tessera {
namespace math {

/**
 * Matrix class for linear algebra operations.
 * 
 * This is a simple row-major matrix implementation matching the original
 * gotetra interface. For performance-critical code, consider using Eigen directly.
 */
class Matrix {
public:
    std::vector<double> vals;
    int width = 0;
    int height = 0;
    
    Matrix() = default;
    
    Matrix(int w, int h) : vals(w * h, 0.0), width(w), height(h) {}
    
    Matrix(const std::vector<double>& values, int w, int h) 
        : vals(values), width(w), height(h) {
        if (static_cast<int>(values.size()) != w * h) {
            throw std::runtime_error("vals size must equal width * height");
        }
    }
    
    /**
     * Access element at (row, col).
     */
    double& at(int row, int col) { return vals[row * width + col]; }
    const double& at(int row, int col) const { return vals[row * width + col]; }
    
    /**
     * Multiply this matrix by another.
     */
    Matrix mult(const Matrix& other) const {
        if (width != other.height) {
            throw std::runtime_error("Incompatible matrix dimensions for multiplication");
        }
        
        Matrix result(other.width, height);
        mult_at(other, result);
        return result;
    }
    
    /**
     * Multiply and store result in-place.
     */
    void mult_at(const Matrix& other, Matrix& out) const {
        if (width != other.height) {
            throw std::runtime_error("Incompatible matrix dimensions");
        }
        
        std::fill(out.vals.begin(), out.vals.end(), 0.0);
        
        for (int i = 0; i < height; ++i) {
            int off = i * width;
            for (int j = 0; j < other.width; ++j) {
                int out_idx = i * other.width + j;
                for (int k = 0; k < width; ++k) {
                    out.vals[out_idx] += vals[off + k] * other.vals[k * other.width + j];
                }
            }
        }
    }
    
    /**
     * Multiply matrix by vector: out = M * v
     */
    static void mult_vec(const Matrix& m, const std::vector<double>& v, 
                         std::vector<double>& out) {
        if (m.width != static_cast<int>(v.size()) || 
            m.height != static_cast<int>(out.size())) {
            throw std::runtime_error("Shape error in mult_vec");
        }
        
        std::fill(out.begin(), out.end(), 0.0);
        int offset = 0;
        for (int j = 0; j < m.height; ++j) {
            for (int i = 0; i < m.width; ++i) {
                out[j] += m.vals[offset + i] * v[i];
            }
            offset += m.width;
        }
    }
    
    /**
     * Multiply vector by matrix: out = v * M
     */
    static void vec_mult(const std::vector<double>& v, const Matrix& m,
                         std::vector<double>& out) {
        if (m.height != static_cast<int>(v.size()) ||
            m.width != static_cast<int>(out.size())) {
            throw std::runtime_error("Shape error in vec_mult");
        }
        
        std::fill(out.begin(), out.end(), 0.0);
        for (int i = 0; i < m.width; ++i) {
            double sum = 0.0;
            for (int j = 0; j < m.height; ++j) {
                sum += v[j] * m.vals[i + m.width * j];
            }
            out[i] = sum;
        }
    }
    
    /**
     * Transpose the matrix.
     */
    Matrix transpose() const {
        Matrix result(height, width);
        transpose_at(result);
        return result;
    }
    
    void transpose_at(Matrix& out) const {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                out.vals[j * height + i] = vals[i * width + j];
            }
        }
    }
    
    /**
     * Create an identity matrix.
     */
    static Matrix identity(int n) {
        Matrix m(n, n);
        for (int i = 0; i < n; ++i) {
            m.vals[i * n + i] = 1.0;
        }
        return m;
    }
};

/**
 * LU decomposition factors for efficient matrix operations.
 * Uses Crout's algorithm with partial pivoting.
 */
class LUFactors {
public:
    Matrix lu;
    std::vector<int> perm;  // Permutation vector
    double d = 1.0;  // Sign of permutation
    
    LUFactors() = default;
    
    explicit LUFactors(int n) : lu(n, n), perm(n), d(1.0) {
        for (int i = 0; i < n; ++i) perm[i] = i;
    }
    
    /**
     * Compute LU decomposition of a matrix using Crout's algorithm.
     */
    void factorize(const Matrix& m) {
        if (m.width != m.height) {
            throw std::runtime_error("Matrix must be square for LU decomposition");
        }
        
        int n = m.width;
        lu = m;
        perm.resize(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        d = 1.0;
        
        for (int k = 0; k < n; ++k) {
            // Find pivot
            double max_val = 0.0;
            int max_row = k;
            for (int i = k; i < n; ++i) {
                double val = std::abs(lu.vals[i * n + k]);
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }
            
            if (max_val == 0.0) {
                throw std::runtime_error("Singular matrix");
            }
            
            // Swap rows
            if (max_row != k) {
                for (int j = 0; j < n; ++j) {
                    std::swap(lu.vals[k * n + j], lu.vals[max_row * n + j]);
                }
                std::swap(perm[k], perm[max_row]);
                d = -d;
            }
            
            // Eliminate column
            for (int i = k + 1; i < n; ++i) {
                lu.vals[i * n + k] /= lu.vals[k * n + k];
                for (int j = k + 1; j < n; ++j) {
                    lu.vals[i * n + j] -= lu.vals[i * n + k] * lu.vals[k * n + j];
                }
            }
        }
    }
    
    /**
     * Solve M * x = b for x.
     */
    std::vector<double> solve_vector(const std::vector<double>& b) const {
        std::vector<double> x(b.size());
        solve_vector_at(b, x);
        return x;
    }
    
    void solve_vector_at(const std::vector<double>& b, std::vector<double>& x) const {
        int n = lu.width;
        if (n != static_cast<int>(b.size()) || n != static_cast<int>(x.size())) {
            throw std::runtime_error("Vector size mismatch");
        }
        
        // Apply permutation to b
        std::vector<double> pb(n);
        for (int i = 0; i < n; ++i) {
            pb[i] = b[perm[i]];
        }
        
        // Forward substitution: L * y = Pb
        std::vector<double> y(n);
        for (int i = 0; i < n; ++i) {
            double sum = pb[i];
            for (int j = 0; j < i; ++j) {
                sum -= lu.vals[i * n + j] * y[j];
            }
            y[i] = sum;  // L has 1s on diagonal
        }
        
        // Back substitution: U * x = y
        for (int i = n - 1; i >= 0; --i) {
            double sum = y[i];
            for (int j = i + 1; j < n; ++j) {
                sum -= lu.vals[i * n + j] * x[j];
            }
            x[i] = sum / lu.vals[i * n + i];
        }
    }
    
    /**
     * Compute the inverse matrix.
     */
    Matrix invert() const {
        int n = lu.width;
        Matrix inv(n, n);
        
        std::vector<double> e(n, 0.0);
        std::vector<double> col(n);
        
        for (int j = 0; j < n; ++j) {
            // Create j-th unit vector
            if (j > 0) e[j-1] = 0.0;
            e[j] = 1.0;
            
            // Solve A * col = e_j
            solve_vector_at(e, col);
            
            // Store as j-th column of inverse
            for (int i = 0; i < n; ++i) {
                inv.vals[i * n + j] = col[i];
            }
        }
        
        return inv;
    }
    
    void invert_at(Matrix& out) const {
        out = invert();
    }
    
    /**
     * Compute the determinant.
     */
    double determinant() const {
        double det = d;
        int n = lu.width;
        for (int i = 0; i < n; ++i) {
            det *= lu.vals[i * n + i];
        }
        return det;
    }
};

// Convenience functions on Matrix

/**
 * Compute LU decomposition.
 */
inline LUFactors lu_decompose(const Matrix& m) {
    LUFactors luf(m.width);
    luf.factorize(m);
    return luf;
}

/**
 * Invert a matrix.
 */
inline Matrix invert(const Matrix& m) {
    return lu_decompose(m).invert();
}

/**
 * Compute determinant.
 */
inline double determinant(const Matrix& m) {
    return lu_decompose(m).determinant();
}

/**
 * Solve M * x = b for x.
 */
inline std::vector<double> solve(const Matrix& m, const std::vector<double>& b) {
    return lu_decompose(m).solve_vector(b);
}

} // namespace math
} // namespace tessera
