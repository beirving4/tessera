#pragma once

#include <memory>
#include <array>
#include "geom/vec.h"
#include "geom/grid.h"
#include "density/buffer.h"
#include "density/interpolator.h"
#include "io/headers.h"

namespace asymptotic_tetra {
namespace render {

// Forward declarations
class Overlap;

/**
 * Box configuration for rendering.
 */
struct BoxConfig {
    double x = 0;
    double y = 0;
    double z = 0;
    double x_width = 0;
    double y_width = 0;
    double z_width = 0;
    std::string projection_axis;  // "X", "Y", "Z", or empty for 3D

    bool is_projection() const {
        return !projection_axis.empty();
    }
};

/**
 * Abstract Box interface representing a rendering domain.
 */
class Box {
public:
    virtual ~Box() = default;

    /// Create an overlap calculator for a given sheet segment
    virtual std::unique_ptr<Overlap> overlap(const io::SheetHeader& hd) = 0;

    /// Cell span (width) in each dimension
    virtual std::array<int, 3> cell_span() const = 0;

    /// Cell origin in each dimension
    virtual std::array<int, 3> cell_origin() const = 0;

    /// Physical width of each cell
    virtual double cell_width() const = 0;

    /// Number of cells per dimension
    virtual int num_cells() const = 0;

    /// Number of sample points per tetrahedron
    virtual int points() const = 0;

    /// Get the output buffer
    virtual density::Buffer& vals() = 0;

    /// Get projection axis (-1 and false if 3D)
    virtual std::pair<int, bool> projection_axis() const = 0;
};

/**
 * Abstract Overlap interface for computing overlap between segments and boxes.
 */
class Overlap : public density::Interpolator {
public:
    virtual ~Overlap() = default;

    /// Calculate buffer size required
    virtual int buffer_size() const = 0;

    /// Scale vectors into overlap's code units
    virtual void scale_vecs(std::vector<geom::Vec3f>& vs, 
                           geom::CellBounds& vcb) = 0;

    /// Add buffer contents to grid
    virtual void add(density::Buffer& buf, density::Buffer& grid) = 0;
};

/**
 * Base overlap implementation with common functionality.
 */
class BaseOverlap : public Overlap {
public:
    const geom::CellBounds* buffer_cell_bounds() const override {
        return &buf_cb_;
    }

    const geom::CellBounds* domain_cell_bounds() const override {
        return &dom_cb_;
    }

    int cells() const override {
        return cells_;
    }

    // Made public for Box implementations to initialize
    geom::CellBounds buf_cb_;
    geom::CellBounds dom_cb_;
    int cells_ = 0;
    double box_width_ = 0;
};

/**
 * 3D overlap for segment-based coordinate system.
 */
class SegmentOverlap3D : public BaseOverlap {
public:
    int buffer_size() const override {
        return buf_cb_.width[0] * buf_cb_.width[1] * buf_cb_.width[2];
    }

    void scale_vecs(std::vector<geom::Vec3f>& vs, 
                   geom::CellBounds& vcb) override;

    void add(density::Buffer& buf, density::Buffer& grid) override;

    void interpolate(
        density::Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        density::Buffer& weights,
        int low, int high, int jump
    ) override;
};

/**
 * 3D overlap for domain-based coordinate system.
 */
class DomainOverlap3D : public BaseOverlap {
public:
    int buffer_size() const override {
        return buf_cb_.width[0] * buf_cb_.width[1] * buf_cb_.width[2];
    }

    void scale_vecs(std::vector<geom::Vec3f>& vs,
                   geom::CellBounds& vcb) override;

    void add(density::Buffer& buf, density::Buffer& grid) override;

    void interpolate(
        density::Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        density::Buffer& weights,
        int low, int high, int jump
    ) override;
};

/**
 * 2D (projected) overlap for segment-based coordinate system.
 */
class SegmentOverlap2D : public BaseOverlap {
public:
    int buffer_size() const override;

    void scale_vecs(std::vector<geom::Vec3f>& vs,
                   geom::CellBounds& vcb) override;

    void add(density::Buffer& buf, density::Buffer& grid) override;

    void interpolate(
        density::Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        density::Buffer& weights,
        int low, int high, int jump
    ) override;

    int proj_ = 2;  // projection axis (0=X, 1=Y, 2=Z)
};

/**
 * 2D (projected) overlap for domain-based coordinate system.
 */
class DomainOverlap2D : public BaseOverlap {
public:
    int buffer_size() const override;

    void scale_vecs(std::vector<geom::Vec3f>& vs,
                   geom::CellBounds& vcb) override;

    void add(density::Buffer& buf, density::Buffer& grid) override;

    void interpolate(
        density::Buffer& buf,
        const std::vector<geom::Vec3f>& xs,
        const std::vector<geom::Vec3f>& vs,
        double pt_val,
        density::Buffer& weights,
        int low, int high, int jump
    ) override;

    int proj_ = 2;  // projection axis
};

/**
 * 3D Box implementation.
 */
class Box3D : public Box {
public:
    Box3D(double box_width, int pts, int cells, 
          density::Quantity q, const BoxConfig& config);

    std::unique_ptr<Overlap> overlap(const io::SheetHeader& hd) override;

    std::array<int, 3> cell_span() const override { return cb_.width; }
    std::array<int, 3> cell_origin() const override { return cb_.origin; }
    double cell_width() const override { return cell_width_; }
    int num_cells() const override { return cells_; }
    int points() const override { return pts_; }
    density::Buffer& vals() override { return *bvals_; }
    std::pair<int, bool> projection_axis() const override { return {-1, false}; }

private:
    geom::CellBounds cb_;
    std::unique_ptr<density::Buffer> bvals_;
    int pts_;
    int cells_;
    double cell_width_;
    density::Quantity q_;
};

/**
 * 2D (projected) Box implementation.
 */
class Box2D : public Box {
public:
    Box2D(double box_width, int pts, int cells,
          density::Quantity q, const BoxConfig& config);

    std::unique_ptr<Overlap> overlap(const io::SheetHeader& hd) override;

    std::array<int, 3> cell_span() const override { return cb_.width; }
    std::array<int, 3> cell_origin() const override { return cb_.origin; }
    double cell_width() const override { return cell_width_; }
    int num_cells() const override { return cells_; }
    int points() const override { return pts_; }
    density::Buffer& vals() override { return *bvals_; }
    std::pair<int, bool> projection_axis() const override { return {proj_, true}; }

private:
    geom::CellBounds cb_;
    std::unique_ptr<density::Buffer> bvals_;
    int pts_;
    int cells_;
    double cell_width_;
    density::Quantity q_;
    int proj_;  // 0=X, 1=Y, 2=Z
};

/**
 * Factory function to create a Box.
 */
std::unique_ptr<Box> create_box(
    double box_width, int pts, int cells,
    density::Quantity q, const BoxConfig& config
);

// Utility functions
inline int int_floor(float x) {
    return x > 0 ? static_cast<int>(x) : static_cast<int>(x - 1);
}

inline int bound(int x, int cells) {
    if (x < 0) return x + cells;
    else if (x >= cells) return x - cells;
    return x;
}

} // namespace render
} // namespace asymptotic_tetra
