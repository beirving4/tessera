#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include "vec.h"
#include "grid.h"

namespace asymptotic_tetra {

// Forward declaration
namespace math { class Generator; }

namespace geom {

constexpr double TETRA_EPS = 5e-5;
constexpr int TETRA_DIR_COUNT = 6;
constexpr int TETRA_CENTERED_COUNT = 8;

// Direction lookup tables for tetrahedron configurations
extern const int64_t DIRS[TETRA_DIR_COUNT][2][3];
extern const int64_t CENTERS[TETRA_CENTERED_COUNT][3][3];

/**
 * TetraIdxs are the indices of particles which are the corners of a tetrahedron.
 */
class TetraIdxs {
public:
    std::array<int64_t, 4> indices = {0, 0, 0, 0};

    TetraIdxs() = default;
    TetraIdxs(int64_t idx, int64_t count_width, int64_t skip, int dir) {
        init(idx, count_width, skip, dir);
    }

    int64_t& operator[](size_t i) { return indices[i]; }
    const int64_t& operator[](size_t i) const { return indices[i]; }

    /**
     * Initialize indices using the given anchor point, count width, skip, and direction.
     */
    TetraIdxs& init(int64_t idx, int64_t count_width, int64_t skip, int dir);

    /**
     * Initialize using Cartesian coordinates.
     */
    TetraIdxs& init_cartesian(int64_t x, int64_t y, int64_t z, int64_t count_width, int dir);

    /**
     * Initialize for centered configurations.
     */
    std::pair<TetraIdxs*, bool> init_centered(int64_t idx, int64_t count_width, int64_t skip, int dir);

private:
    static int64_t compress_coords(int64_t x, int64_t y, int64_t z,
                                   int64_t dx, int64_t dy, int64_t dz,
                                   int64_t count_width);
    static std::pair<int64_t, bool> compress_coords_check(
        int64_t x, int64_t y, int64_t z,
        int64_t dx, int64_t dy, int64_t dz,
        int64_t count_width);
};

/**
 * Tetra is a tetrahedron with points inside a box with periodic boundary conditions.
 * Contains cached volume and barycenter for efficiency.
 */
class Tetra {
public:
    std::array<Vec3f, 4> corners;

    Tetra() = default;
    Tetra(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2, const Vec3f& c3) {
        init(c0, c1, c2, c3);
    }

    /**
     * Initialize the tetrahedron with given corners.
     */
    Tetra& init(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2, const Vec3f& c3);

    /**
     * Compute the volume of the tetrahedron.
     */
    double volume();

    /**
     * Check if the tetrahedron contains the given point.
     */
    bool contains(const Vec3f& v);

    /**
     * Compute the barycenter of the tetrahedron.
     */
    const Vec3f& barycenter();

    /**
     * Get cell bounds aligned to the given cell width.
     */
    CellBounds cell_bounds(double cell_width);

    /**
     * Fill cell bounds in-place.
     */
    void cell_bounds_at(double cell_width, CellBounds& cb);

    /**
     * Randomly sample points uniformly within the tetrahedron.
     */
    void random_sample(math::Generator& gen, std::vector<double>& rand_buf, 
                       std::vector<Vec3f>& vec_buf);

    /**
     * Distribute points from a unit cube to within this tetrahedron.
     */
    void distribute(const std::vector<double>& xs, const std::vector<double>& ys,
                    const std::vector<double>& zs, std::vector<Vec3f>& vec_buf);

    /**
     * Distribute points from a unit tetrahedron to this tetrahedron.
     */
    void distribute_tetra(const std::vector<Vec3f>& pts, std::vector<Vec3f>& out);

    /**
     * Same as distribute_tetra but output is double precision.
     */
    void distribute_tetra64(const std::vector<Vec3f>& pts, std::vector<std::array<double, 3>>& out);

private:
    double volume_val = 0.0;
    Vec3f bary;
    bool volume_valid = false;
    bool bary_valid = false;

    // Buffers for volume calculation
    Vec3f vb_buf1, vb_buf2, vb_buf3;
    // Buffers for sampling
    std::array<Vec3f, 4> sb_d, sb_c;

    double signed_volume(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2, const Vec3f& c3);
    static bool eps_eq(double x, double y, double eps);
};

/**
 * Distribute a set of points in a unit cube across a unit tetrahedron.
 * Uses the Rocchini-Cignoni algorithm.
 */
void distribute_unit(std::vector<Vec3f>& vec_buf);

/**
 * Helper to find min/max.
 */
inline std::pair<float, float> min_max(float x, float old_min, float old_max) {
    if (x > old_max) return {old_min, x};
    if (x < old_min) return {x, old_max};
    return {old_min, old_max};
}

/**
 * Get width of tetrahedron in a given dimension.
 */
inline double tetra_width(const Tetra& tet) {
    float min_x = tet.corners[0][0];
    float max_x = tet.corners[0][0];
    for (int i = 1; i < 4; ++i) {
        float x = tet.corners[i][0];
        if (x < min_x) min_x = x;
        else if (x > max_x) max_x = x;
    }
    return static_cast<double>(max_x - min_x);
}

/**
 * Check if any corner has negative coordinate in dimension j.
 */
inline bool coord_neg(const Tetra& tet, int j) {
    for (int i = 0; i < 4; ++i) {
        if (tet.corners[i][j] < 0) return true;
    }
    return false;
}

/**
 * Add width to any negative coordinates in dimension j.
 */
inline void mod_coord(std::vector<Vec3f>& buf, int j, float width) {
    for (auto& v : buf) {
        if (v[j] < 0) v[j] += width;
    }
}

} // namespace geom
} // namespace asymptotic_tetra
