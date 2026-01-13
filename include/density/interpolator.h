#pragma once

#include <memory>
#include <vector>
#include "geom/vec.h"
#include "geom/grid.h"
#include "geom/tetra.h"
#include "density/buffer.h"
#include "math/rand.h"

namespace tessera {
namespace density {

/**
 * Abstract interpolator interface for distributing particle mass/values onto a grid.
 */
class Interpolator {
public:
    virtual ~Interpolator() = default;

    /**
     * Interpolate values from particles onto the buffer grid.
     * 
     * @param buf Output buffer to accumulate values
     * @param xs Position array
     * @param vs Velocity array (may be empty if not needed)
     * @param pt_val Value per point
     * @param weights Weight buffer (NilBuffer if not needed)
     * @param low Starting index
     * @param high Ending index
     * @param jump Step size (for parallel processing)
     */
    virtual void interpolate(
        Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        Buffer& weights,
        int low, int high, int jump
    ) = 0;

    /// The bounding box around the domain being written to
    virtual const geom::CellBounds* domain_cell_bounds() const = 0;

    /// The bounding box around the buffer being written from
    virtual const geom::CellBounds* buffer_cell_bounds() const = 0;

    /// Number of cells per dimension
    virtual int cells() const = 0;
};

/**
 * Monte Carlo interpolator using tetrahedron-based sampling.
 * 
 * This distributes particle mass onto a grid by sampling points uniformly
 * within each tetrahedron formed by the phase-space sheet tessellation.
 */
class MonteCarlo : public Interpolator {
public:
    /**
     * Create a Monte Carlo interpolator.
     * 
     * @param seg_width Segment width (number of particles per side)
     * @param points Number of sample points per tetrahedron
     * @param cells Number of grid cells per dimension
     * @param skip Subsampling factor
     * @param unit_bufs Pre-generated unit cube sample buffers
     * @param sub_intr Sub-interpolator for depositing points to grid
     */
    MonteCarlo(
        int64_t seg_width,
        int points,
        int cells,
        int64_t skip,
        std::vector<std::vector<geom::Vec3f>>& unit_bufs,
        std::shared_ptr<Interpolator> sub_intr
    );

    void interpolate(
        Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        Buffer& weights,
        int low, int high, int jump
    ) override;

    const geom::CellBounds* domain_cell_bounds() const override {
        return sub_intr_->domain_cell_bounds();
    }

    const geom::CellBounds* buffer_cell_bounds() const override {
        return sub_intr_->buffer_cell_bounds();
    }

    int cells() const override {
        return sub_intr_->cells();
    }

private:
    std::shared_ptr<Interpolator> sub_intr_;
    int64_t seg_width_;
    int points_;
    int cells_;
    int64_t skip_;
    
    math::Generator gen_;
    
    // Buffers
    geom::TetraIdxs idx_buf_;
    geom::Tetra tet_;
    geom::Tetra vtet_;
    
    std::vector<std::vector<geom::Vec3f>>& unit_bufs_;
    std::vector<geom::Vec3f> vec_buf_;

    // Helper functions
    static int64_t index(int64_t x, int64_t y, int64_t z, int64_t cells) {
        return x + y * cells + z * cells * cells;
    }

    static void coords(int64_t idx, int64_t cells, int64_t& x, int64_t& y, int64_t& z) {
        x = idx % cells;
        y = (idx % (cells * cells)) / cells;
        z = idx / (cells * cells);
    }

    static bool coord_neg(const geom::Tetra& tet, int j) {
        for (int i = 0; i < 4; i++) {
            if (tet.corners[i][j] < 0) return true;
        }
        return false;
    }

    static void mod_coord(std::vector<geom::Vec3f>& buf, int j, float width) {
        for (auto& v : buf) {
            if (v[j] < 0) v[j] += width;
        }
    }

    static double tet_width(const geom::Tetra& tet) {
        float min_x = tet.corners[0][0];
        float max_x = tet.corners[0][0];
        for (int i = 1; i < 4; i++) {
            float x = tet.corners[i][0];
            if (x < min_x) min_x = x;
            else if (x > max_x) max_x = x;
        }
        return static_cast<double>(max_x - min_x);
    }
};

/**
 * Helper to subtract cell bounds origins.
 */
inline void cb_subtr(const geom::CellBounds& cb1, const geom::CellBounds& cb2,
                     int& i, int& j, int& k) {
    i = cb1.origin[0] - cb2.origin[0];
    j = cb1.origin[1] - cb2.origin[1];
    k = cb1.origin[2] - cb2.origin[2];
}

} // namespace density
} // namespace tessera
