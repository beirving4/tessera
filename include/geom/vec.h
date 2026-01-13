#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace tessera {
namespace geom {

/**
 * Vec3f represents a 3D vector using single-precision floats.
 * Provides operations for periodic boundary conditions commonly used
 * in cosmological simulations.
 */
class Vec3f {
public:
    std::array<float, 3> data;

    // Constructors
    Vec3f() : data{0.0f, 0.0f, 0.0f} {}
    Vec3f(float x, float y, float z) : data{x, y, z} {}
    explicit Vec3f(const std::array<float, 3>& arr) : data(arr) {}

    // Element access
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }

    float x() const { return data[0]; }
    float y() const { return data[1]; }
    float z() const { return data[2]; }

    float& x() { return data[0]; }
    float& y() { return data[1]; }
    float& z() { return data[2]; }

    // Scaling operations
    Vec3f scale(double k) const {
        return Vec3f(
            static_cast<float>(data[0] * k),
            static_cast<float>(data[1] * k),
            static_cast<float>(data[2] * k)
        );
    }

    Vec3f& scale_self(double k) {
        for (int i = 0; i < 3; ++i) {
            data[i] = static_cast<float>(data[i] * k);
        }
        return *this;
    }

    // Modulo operation for periodic boundaries
    Vec3f mod(double width) const {
        Vec3f result;
        float w = static_cast<float>(width);
        for (int i = 0; i < 3; ++i) {
            result[i] = data[i];
            if (result[i] >= w) {
                result[i] -= w;
            } else if (result[i] < 0) {
                result[i] += w;
            }
        }
        return result;
    }

    Vec3f& mod_self(double width) {
        float w = static_cast<float>(width);
        for (int i = 0; i < 3; ++i) {
            if (data[i] >= w) {
                data[i] -= w;
            } else if (data[i] < 0) {
                data[i] += w;
            }
        }
        return *this;
    }

    // Addition
    Vec3f add(const Vec3f& other) const {
        return Vec3f(
            data[0] + other[0],
            data[1] + other[1],
            data[2] + other[2]
        );
    }

    Vec3f& add_self(const Vec3f& other) {
        for (int i = 0; i < 3; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    Vec3f operator+(const Vec3f& other) const { return add(other); }
    Vec3f& operator+=(const Vec3f& other) { return add_self(other); }

    // Subtraction with periodic boundary conditions
    Vec3f sub(const Vec3f& other, double width) const {
        Vec3f result;
        float w = static_cast<float>(width);
        float w2 = w / 2.0f;

        for (int i = 0; i < 3; ++i) {
            float diff = data[i] - other[i];
            if (diff < -w) {
                diff += w;
            } else if (diff >= w) {
                diff -= w;
            }

            if (diff > w2) {
                diff -= w;
            } else if (diff < -w2) {
                diff += w;
            }
            result[i] = diff;
        }
        return result;
    }

    // Simple subtraction (no periodic boundary)
    Vec3f operator-(const Vec3f& other) const {
        return Vec3f(
            data[0] - other[0],
            data[1] - other[1],
            data[2] - other[2]
        );
    }

    // Dot product
    double dot(const Vec3f& other) const {
        double sum = 0.0;
        for (int i = 0; i < 3; ++i) {
            sum += static_cast<double>(data[i] * other[i]);
        }
        return sum;
    }

    // Cross product
    Vec3f cross(const Vec3f& other) const {
        return Vec3f(
            data[1] * other[2] - data[2] * other[1],
            data[2] * other[0] - data[0] * other[2],
            data[0] * other[1] - data[1] * other[0]
        );
    }

    Vec3f& cross_self(const Vec3f& other) {
        float out0 = data[1] * other[2] - data[2] * other[1];
        float out1 = data[2] * other[0] - data[0] * other[2];
        float out2 = data[0] * other[1] - data[1] * other[0];
        data[0] = out0;
        data[1] = out1;
        data[2] = out2;
        return *this;
    }

    // Norm
    double norm() const {
        return std::sqrt(dot(*this));
    }

    // Scalar multiplication
    Vec3f operator*(float scalar) const {
        return Vec3f(data[0] * scalar, data[1] * scalar, data[2] * scalar);
    }

    Vec3f& operator*=(float scalar) {
        for (int i = 0; i < 3; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }
};

// Free function for scalar * Vec3f
inline Vec3f operator*(float scalar, const Vec3f& v) {
    return v * scalar;
}

/**
 * Vec3d represents a 3D vector using double-precision floats.
 */
class Vec3d {
public:
    std::array<double, 3> data;

    Vec3d() : data{0.0, 0.0, 0.0} {}
    Vec3d(double x, double y, double z) : data{x, y, z} {}
    explicit Vec3d(const std::array<double, 3>& arr) : data(arr) {}

    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }

    double x() const { return data[0]; }
    double y() const { return data[1]; }
    double z() const { return data[2]; }
};

} // namespace geom
} // namespace tessera
