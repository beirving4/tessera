#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <cmath>
#include "vec.h"

namespace tessera {
namespace geom {

/**
 * CellBounds represents a bounding box aligned to grid cells.
 */
struct CellBounds {
    std::array<int, 3> origin = {0, 0, 0};
    std::array<int, 3> width = {0, 0, 0};

    CellBounds() = default;
    CellBounds(const std::array<int, 3>& o, const std::array<int, 3>& w)
        : origin(o), width(w) {}

    bool operator==(const CellBounds& other) const {
        return origin == other.origin && width == other.width;
    }

    bool operator!=(const CellBounds& other) const {
        return !(*this == other);
    }

    /**
     * Check if two cell bounds intersect within a periodic domain.
     */
    bool intersect(const CellBounds& other, int cells) const {
        for (int i = 0; i < 3; ++i) {
            int o_small, w_small, o_big, w_big;
            if (width[i] < other.width[i]) {
                o_small = origin[i];
                w_small = width[i];
                o_big = other.origin[i];
                w_big = other.width[i];
            } else {
                o_small = other.origin[i];
                w_small = other.width[i];
                o_big = origin[i];
                w_big = width[i];
            }

            int e_small = o_small + w_small;
            int be_small = bound_helper(e_small, o_big, cells);
            int bo_small = bound_helper(o_small, o_big, cells);

            if (!(be_small < w_big || bo_small < w_big)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check intersection without periodic wrapping.
     */
    bool intersect_unbounded(const CellBounds& other) const {
        for (int i = 0; i < 3; ++i) {
            int o_low, o_high, w_low;
            if (origin[i] < other.origin[i]) {
                o_low = origin[i];
                o_high = other.origin[i];
                w_low = width[i];
            } else {
                o_low = other.origin[i];
                o_high = origin[i];
                w_low = other.width[i];
            }
            if (o_low + w_low <= o_high) {
                return false;
            }
        }
        return true;
    }

    /**
     * Scale vectors for segment-based operations.
     */
    void scale_vecs_segment(std::vector<Vec3f>& vs, int cells, double box_width) const {
        float f_cells = static_cast<float>(cells);
        float f_width = static_cast<float>(box_width);
        float scale = f_cells / f_width;

        Vec3f origin_vec(
            static_cast<float>(origin[0]),
            static_cast<float>(origin[1]),
            static_cast<float>(origin[2])
        );

        for (auto& v : vs) {
            for (int j = 0; j < 3; ++j) {
                v[j] *= scale;
                v[j] -= origin_vec[j];
                if (v[j] < 0) v[j] += f_cells;
            }
        }
    }

    /**
     * Scale vectors for domain-based operations.
     */
    void scale_vecs_domain(const CellBounds& cb, std::vector<Vec3f>& vs, 
                           int cells, double box_width) const {
        float f_cells = static_cast<float>(cells);
        float f_width = static_cast<float>(box_width);
        float scale = f_cells / f_width;

        Vec3f origin_vec(
            static_cast<float>(origin[0]),
            static_cast<float>(origin[1]),
            static_cast<float>(origin[2])
        );

        Vec3f diff(
            static_cast<float>(origin[0] - cb.origin[0]),
            static_cast<float>(origin[1] - cb.origin[1]),
            static_cast<float>(origin[2] - cb.origin[2])
        );

        for (int i = 0; i < 3; ++i) {
            if (diff[i] < -f_cells / 2) {
                diff[i] += f_cells;
            } else if (diff[i] > f_cells / 2) {
                diff[i] -= f_cells;
            }
        }

        for (auto& v : vs) {
            for (int j = 0; j < 3; ++j) {
                v[j] *= scale;
                v[j] -= origin_vec[j];
                if (v[j] < 0) v[j] += f_cells;
                v[j] += diff[j];
            }
        }
    }

private:
    static int bound_helper(int x, int origin, int width) {
        int diff = x - origin;
        if (diff < 0) return diff + width;
        if (diff > width) return diff - width;
        return diff;
    }
};

/**
 * Grid provides an interface for reasoning over a 1D slice as if it were a 3D grid.
 */
class Grid {
public:
    CellBounds bounds;
    int length = 0;    // width[0]
    int area = 0;      // width[0] * width[1]
    int volume = 0;    // width[0] * width[1] * width[2]

    Grid() = default;
    Grid(const std::array<int, 3>& origin, const std::array<int, 3>& width) {
        init(origin, width);
    }

    void init(const std::array<int, 3>& origin, const std::array<int, 3>& width) {
        bounds.origin = origin;
        bounds.width = width;

        length = width[0];
        area = width[0] * width[1];
        volume = width[0] * width[1] * width[2];

        for (int i = 0; i < 3; ++i) {
            u_bounds[i] = origin[i] + width[i];
        }
    }

    /**
     * Returns the grid index corresponding to a set of coordinates.
     */
    int idx(int x, int y, int z) const {
        return (x - bounds.origin[0]) + 
               (y - bounds.origin[1]) * length +
               (z - bounds.origin[2]) * area;
    }

    /**
     * Returns an index and true if the given coordinates are valid, false otherwise.
     */
    std::pair<int, bool> idx_check(int x, int y, int z) const {
        if (!bounds_check(x, y, z)) {
            return {-1, false};
        }
        return {idx(x, y, z), true};
    }

    /**
     * Returns true if the given coordinates are within the Grid.
     */
    bool bounds_check(int x, int y, int z) const {
        return (bounds.origin[0] <= x && bounds.origin[1] <= y && bounds.origin[2] <= z) &&
               (x < u_bounds[0] && y < u_bounds[1] && z < u_bounds[2]);
    }

    /**
     * Returns the x, y, z coordinates of a point from its grid index.
     */
    std::tuple<int, int, int> coords(int idx) const {
        int x = idx % length;
        int y = (idx % area) / length;
        int z = idx / area;
        return {x, y, z};
    }

protected:
    std::array<int, 3> u_bounds = {0, 0, 0};
};

/**
 * GridLocation is a Grid which also specifies a physical location within
 * a periodic super-grid.
 */
class GridLocation : public Grid {
public:
    int cells = 0;
    double box_width = 0.0;

    GridLocation() = default;
    GridLocation(const std::array<int, 3>& origin, const std::array<int, 3>& width,
                 double bw, int c) {
        init(origin, width, bw, c);
    }

    void init(const std::array<int, 3>& origin, const std::array<int, 3>& width,
              double bw, int c) {
        Grid::init(origin, width);
        box_width = bw;
        cells = c;
    }

    // Derivative operations are implemented in deriv.h/cpp
};

// Helper functions
inline int bound_value(int x, int cells) {
    if (x < 0) return x + cells;
    if (x >= cells) return x - cells;
    return x;
}

inline int int_floor(float x) {
    if (x > 0) return static_cast<int>(x);
    return static_cast<int>(x - 1);
}

} // namespace geom
} // namespace tessera
