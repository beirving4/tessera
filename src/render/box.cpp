#include "render/box.h"
#include <cmath>

namespace tessera {
namespace render {

// ============================================================================
// Box implementations
// ============================================================================

Box3D::Box3D(double box_width, int pts, int cells,
             density::Quantity q, const BoxConfig& config)
    : pts_(pts), cells_(cells), q_(q)
{
    cell_width_ = box_width / static_cast<double>(cells);
    
    std::array<double, 3> origin = {config.x, config.y, config.z};
    std::array<double, 3> width = {config.x_width, config.y_width, config.z_width};

    for (int j = 0; j < 3; j++) {
        cb_.origin[j] = static_cast<int>(std::floor(origin[j] / cell_width_));
        cb_.width[j] = 1 + static_cast<int>(
            std::floor((width[j] + origin[j]) / cell_width_)
        );
        cb_.width[j] -= cb_.origin[j];
    }

    int len = cb_.width[0] * cb_.width[1] * cb_.width[2];
    geom::GridLocation g(cb_.origin, cb_.width, box_width, cells);
    bvals_ = density::create_buffer(q, len, pts, g);
}

std::unique_ptr<Overlap> Box3D::overlap(const io::SheetHeader& hd) {
    auto seg = std::make_unique<SegmentOverlap3D>();
    auto dom = std::make_unique<DomainOverlap3D>();

    seg->box_width_ = cell_width_ * static_cast<double>(cells_);
    dom->box_width_ = cell_width_ * static_cast<double>(cells_);
    seg->cells_ = cells_;
    dom->cells_ = cells_;

    dom->dom_cb_ = cb_;
    seg->dom_cb_ = cb_;
    dom->buf_cb_ = cb_;
    seg->buf_cb_ = hd.cell_bounds(cells_);

    if (seg->buffer_size() <= dom->buffer_size()) {
        return seg;
    } else {
        return dom;
    }
}

Box2D::Box2D(double box_width, int pts, int cells,
             density::Quantity q, const BoxConfig& config)
    : pts_(pts), cells_(cells), q_(q)
{
    if (config.projection_axis == "X") {
        proj_ = 0;
    } else if (config.projection_axis == "Y") {
        proj_ = 1;
    } else {
        proj_ = 2;  // Default to Z
    }

    cell_width_ = box_width / static_cast<double>(cells);
    
    std::array<double, 3> origin = {config.x, config.y, config.z};
    std::array<double, 3> width = {config.x_width, config.y_width, config.z_width};

    for (int j = 0; j < 3; j++) {
        cb_.origin[j] = static_cast<int>(std::floor(origin[j] / cell_width_));
        cb_.width[j] = 1 + static_cast<int>(
            std::floor((width[j] + origin[j]) / cell_width_)
        );
        cb_.width[j] -= cb_.origin[j];
    }

    int i_dim = 0, j_dim = 1;
    if (proj_ == 0) { i_dim = 1; j_dim = 2; }
    if (proj_ == 1) { i_dim = 0; j_dim = 2; }

    int len = cb_.width[i_dim] * cb_.width[j_dim];
    geom::GridLocation g(cb_.origin, cb_.width, box_width, cells);
    bvals_ = density::create_buffer(q, len, pts, g);
}

std::unique_ptr<Overlap> Box2D::overlap(const io::SheetHeader& hd) {
    auto seg = std::make_unique<SegmentOverlap2D>();
    auto dom = std::make_unique<DomainOverlap2D>();

    seg->box_width_ = cell_width_ * static_cast<double>(cells_);
    dom->box_width_ = cell_width_ * static_cast<double>(cells_);
    seg->cells_ = cells_;
    dom->cells_ = cells_;
    seg->proj_ = proj_;
    dom->proj_ = proj_;

    dom->dom_cb_ = cb_;
    seg->dom_cb_ = cb_;
    dom->buf_cb_ = cb_;
    seg->buf_cb_ = hd.cell_bounds(cells_);

    if (seg->buffer_size() <= dom->buffer_size() ||
        (dom->buf_cb_.width[0] >= cells_ / 2 ||
         dom->buf_cb_.width[1] >= cells_ / 2 ||
         dom->buf_cb_.width[2] >= cells_ / 2)) {
        return seg;
    } else {
        return dom;
    }
}

std::unique_ptr<Box> create_box(
    double box_width, int pts, int cells,
    density::Quantity q, const BoxConfig& config
) {
    if (config.is_projection() && density::can_project(q)) {
        return std::make_unique<Box2D>(box_width, pts, cells, q, config);
    } else {
        return std::make_unique<Box3D>(box_width, pts, cells, q, config);
    }
}

// ============================================================================
// 2D Overlap implementations
// ============================================================================

int SegmentOverlap2D::buffer_size() const {
    int prod = 1;
    for (int i = 0; i < 3; i++) {
        if (i == proj_) continue;
        prod *= buf_cb_.width[i];
    }
    return prod;
}

int DomainOverlap2D::buffer_size() const {
    int prod = 1;
    for (int i = 0; i < 3; i++) {
        if (i == proj_) continue;
        prod *= buf_cb_.width[i];
    }
    return prod;
}

void SegmentOverlap2D::scale_vecs(std::vector<geom::Vec3f>& vs,
                                   geom::CellBounds& vcb) {
    vcb.scale_vecs_segment(vs, cells_, box_width_);
}

void DomainOverlap2D::scale_vecs(std::vector<geom::Vec3f>& vs,
                                  geom::CellBounds& vcb) {
    vcb.scale_vecs_domain(buf_cb_, vs, cells_, box_width_);
}

void DomainOverlap2D::add(density::Buffer& bbuf, density::Buffer& bgrid) {
    auto* buf_num = bbuf.count_buffer();
    auto* grid_num = bgrid.count_buffer();
    bool valid = (buf_num != nullptr);

    auto* buf = bbuf.scalar_buffer();
    if (buf) {
        auto* grid = bgrid.scalar_buffer();
        for (size_t i = 0; i < buf->size(); i++) {
            (*grid)[i] += (*buf)[i];
            if (valid) (*grid_num)[i] += (*buf_num)[i];
        }
    } else {
        auto* buf = bbuf.vector_buffer();
        auto* grid = bgrid.vector_buffer();
        if (buf && grid) {
            for (size_t i = 0; i < buf->size(); i++) {
                (*grid)[i][0] += (*buf)[i][0];
                (*grid)[i][1] += (*buf)[i][1];
                (*grid)[i][2] += (*buf)[i][2];
                if (valid) (*grid_num)[i] += (*buf_num)[i];
            }
        }
    }
}

void SegmentOverlap2D::add(density::Buffer& bbuf, density::Buffer& bgrid) {
    auto* buf_num = bbuf.count_buffer();
    auto* grid_num = bgrid.count_buffer();
    bool valid = (buf_num != nullptr);

    int i_dim = 0, j_dim = 1;
    if (proj_ == 0) { i_dim = 1; j_dim = 2; }
    if (proj_ == 1) { i_dim = 0; j_dim = 2; }

    auto* buf = bbuf.scalar_buffer();
    if (buf) {
        auto* grid = bgrid.scalar_buffer();

        for (int j_buf = 0; j_buf < buf_cb_.width[j_dim]; j_buf++) {
            int j_dom = j_buf + (buf_cb_.origin[j_dim] - dom_cb_.origin[j_dim]);
            if (j_dom < 0) j_dom += cells_;
            else if (j_dom >= cells_) j_dom -= cells_;

            if (j_dom >= dom_cb_.width[j_dim]) continue;
            int flat_dom_j = j_dom * dom_cb_.width[i_dim];
            int flat_buf_j = j_buf * buf_cb_.width[i_dim];

            for (int i_buf = 0; i_buf < buf_cb_.width[i_dim]; i_buf++) {
                int i_dom = i_buf + (buf_cb_.origin[i_dim] - dom_cb_.origin[i_dim]);
                if (i_dom < 0) i_dom += cells_;
                else if (i_dom >= cells_) i_dom -= cells_;
                if (i_dom >= dom_cb_.width[i_dim]) continue;

                int dom_idx = i_dom + flat_dom_j;
                int buf_idx = i_buf + flat_buf_j;
                (*grid)[dom_idx] += (*buf)[buf_idx];
                if (valid) (*grid_num)[dom_idx] += (*buf_num)[buf_idx];
            }
        }
    } else {
        auto* buf = bbuf.vector_buffer();
        auto* grid = bgrid.vector_buffer();
        if (buf && grid) {
            for (int j_buf = 0; j_buf < buf_cb_.width[j_dim]; j_buf++) {
                int j_dom = j_buf + (buf_cb_.origin[j_dim] - dom_cb_.origin[j_dim]);
                if (j_dom < 0) j_dom += cells_;
                else if (j_dom >= cells_) j_dom -= cells_;

                if (j_dom >= dom_cb_.width[j_dim]) continue;
                int flat_dom_j = j_dom * dom_cb_.width[i_dim];
                int flat_buf_j = j_buf * buf_cb_.width[i_dim];

                for (int i_buf = 0; i_buf < buf_cb_.width[i_dim]; i_buf++) {
                    int i_dom = i_buf + (buf_cb_.origin[i_dim] - dom_cb_.origin[i_dim]);
                    if (i_dom < 0) i_dom += cells_;
                    else if (i_dom >= cells_) i_dom -= cells_;
                    if (i_dom >= dom_cb_.width[i_dim]) continue;

                    int dom_idx = i_dom + flat_dom_j;
                    int buf_idx = i_buf + flat_buf_j;
                    (*grid)[dom_idx][0] += (*buf)[buf_idx][0];
                    (*grid)[dom_idx][1] += (*buf)[buf_idx][1];
                    (*grid)[dom_idx][2] += (*buf)[buf_idx][2];
                    if (valid) (*grid_num)[dom_idx] += (*buf_num)[buf_idx];
                }
            }
        }
    }
}

void SegmentOverlap2D::interpolate(
    density::Buffer& bbuf,
    const std::vector<geom::Vec3f>& xs,
    const std::vector<geom::Vec3f>& /* vs */,
    double pt_val,
    density::Buffer& bweights,
    int low, int high, int jump
) {
    int i_dim = 0, j_dim = 1, k_dim = 2;
    if (proj_ == 0) { i_dim = 1; j_dim = 2; k_dim = 0; }
    if (proj_ == 1) { i_dim = 0; j_dim = 2; k_dim = 1; }
    
    int k_offset = dom_cb_.origin[k_dim] - buf_cb_.origin[k_dim];
    int length = buf_cb_.width[i_dim];
    pt_val /= static_cast<double>(dom_cb_.width[k_dim]);

    auto* buf = bbuf.scalar_buffer();
    auto* wbuf = bweights.scalar_buffer();
    auto* vbuf = bbuf.vector_buffer();
    auto* vwbuf = bweights.vector_buffer();
    auto* counts = bbuf.count_buffer();
    bool count_valid = (counts != nullptr);
    int wlen = bweights.length();

    for (int idx = low; idx < high; idx += jump) {
        const auto& pt = xs[idx];
        int i = static_cast<int>(pt[i_dim]);
        int j = static_cast<int>(pt[j_dim]);
        int k = int_floor(pt[k_dim]);
        int kk = bound(k - k_offset, cells_);

        if (kk < dom_cb_.width[k_dim] && kk >= 0) {
            int buf_idx = i + j * length;
            if (vbuf && vwbuf) {
                for (int dim = 0; dim < 3; dim++) {
                    (*vbuf)[buf_idx][dim] += pt_val * (*vwbuf)[idx][dim];
                }
                if (count_valid) (*counts)[buf_idx]++;
            } else if (wlen == 0) {
                (*buf)[buf_idx] += pt_val;
                if (count_valid) (*counts)[buf_idx]++;
            } else if (buf && wbuf) {
                (*buf)[buf_idx] += pt_val * (*wbuf)[idx];
                if (count_valid) (*counts)[buf_idx]++;
            }
        }
    }
}

void DomainOverlap2D::interpolate(
    density::Buffer& bbuf,
    const std::vector<geom::Vec3f>& xs,
    const std::vector<geom::Vec3f>& /* vs */,
    double pt_val,
    density::Buffer& bweights,
    int low, int high, int jump
) {
    int i_dim = 0, j_dim = 1, k_dim = 2;
    if (proj_ == 0) { i_dim = 1; j_dim = 2; k_dim = 0; }
    else if (proj_ == 1) { i_dim = 0; j_dim = 2; k_dim = 1; }

    int length = buf_cb_.width[i_dim];
    pt_val /= static_cast<double>(buf_cb_.width[k_dim]);

    auto* buf = bbuf.scalar_buffer();
    auto* wbuf = bweights.scalar_buffer();
    auto* vbuf = bbuf.vector_buffer();
    auto* vwbuf = bweights.vector_buffer();
    auto* counts = bbuf.count_buffer();
    bool count_valid = (counts != nullptr);
    int wlen = bweights.length();

    for (int idx = low; idx < high; idx += jump) {
        const auto& pt = xs[idx];
        int i = bound(int_floor(pt[i_dim]), cells_);
        if (i < buf_cb_.width[i_dim]) {
            int j = bound(int_floor(pt[j_dim]), cells_);
            if (j < buf_cb_.width[j_dim]) {
                int k = bound(int_floor(pt[k_dim]), cells_);
                if (k < buf_cb_.width[k_dim]) {
                    int buf_idx = i + j * length;
                    if (vbuf && vwbuf) {
                        for (int dim = 0; dim < 3; dim++) {
                            (*vbuf)[buf_idx][dim] += pt_val * (*vwbuf)[idx][dim];
                        }
                        if (count_valid) (*counts)[buf_idx]++;
                    } else if (wlen == 0) {
                        (*buf)[buf_idx] += pt_val;
                        if (count_valid) (*counts)[buf_idx]++;
                    } else if (buf && wbuf) {
                        (*buf)[buf_idx] += pt_val * (*wbuf)[idx];
                        if (count_valid) (*counts)[buf_idx]++;
                    }
                }
            }
        }
    }
}

// ============================================================================
// 3D Overlap implementations
// ============================================================================

void SegmentOverlap3D::scale_vecs(std::vector<geom::Vec3f>& vs,
                                   geom::CellBounds& vcb) {
    vcb.scale_vecs_segment(vs, cells_, box_width_);
}

void DomainOverlap3D::scale_vecs(std::vector<geom::Vec3f>& vs,
                                  geom::CellBounds& vcb) {
    vcb.scale_vecs_domain(buf_cb_, vs, cells_, box_width_);
}

void DomainOverlap3D::add(density::Buffer& bbuf, density::Buffer& bgrid) {
    auto* buf_num = bbuf.count_buffer();
    auto* grid_num = bgrid.count_buffer();
    bool valid = (buf_num != nullptr);

    auto* buf = bbuf.scalar_buffer();
    if (buf) {
        auto* grid = bgrid.scalar_buffer();
        for (size_t i = 0; i < buf->size(); i++) {
            (*grid)[i] += (*buf)[i];
            if (valid) (*grid_num)[i] += (*buf_num)[i];
        }
    } else {
        auto* buf = bbuf.vector_buffer();
        auto* grid = bgrid.vector_buffer();
        if (buf && grid) {
            for (size_t i = 0; i < buf->size(); i++) {
                (*grid)[i][0] += (*buf)[i][0];
                (*grid)[i][1] += (*buf)[i][1];
                (*grid)[i][2] += (*buf)[i][2];
                if (valid) (*grid_num)[i] += (*buf_num)[i];
            }
        }
    }
}

void SegmentOverlap3D::add(density::Buffer& bbuf, density::Buffer& bgrid) {
    auto* buf_num = bbuf.count_buffer();
    auto* grid_num = bgrid.count_buffer();
    bool valid = (buf_num != nullptr);

    auto* buf = bbuf.scalar_buffer();
    if (buf) {
        auto* grid = bgrid.scalar_buffer();

        for (int z_buf = 0; z_buf < buf_cb_.width[2]; z_buf++) {
            int z_dom = z_buf + (buf_cb_.origin[2] - dom_cb_.origin[2]);
            if (z_dom < 0) z_dom += cells_;
            else if (z_dom >= cells_) z_dom -= cells_;
            if (z_dom >= dom_cb_.width[2]) continue;

            int flat_dom_z = z_dom * dom_cb_.width[1] * dom_cb_.width[0];
            int flat_buf_z = z_buf * buf_cb_.width[1] * buf_cb_.width[0];

            for (int y_buf = 0; y_buf < buf_cb_.width[1]; y_buf++) {
                int y_dom = y_buf + (buf_cb_.origin[1] - dom_cb_.origin[1]);
                if (y_dom < 0) y_dom += cells_;
                else if (y_dom >= cells_) y_dom -= cells_;
                if (y_dom >= dom_cb_.width[1]) continue;

                int flat_dom_y = y_dom * dom_cb_.width[0];
                int flat_buf_y = y_buf * buf_cb_.width[0];

                for (int x_buf = 0; x_buf < buf_cb_.width[0]; x_buf++) {
                    int x_dom = x_buf + (buf_cb_.origin[0] - dom_cb_.origin[0]);
                    if (x_dom < 0) x_dom += cells_;
                    else if (x_dom >= cells_) x_dom -= cells_;
                    if (x_dom >= dom_cb_.width[0]) continue;

                    int dom_idx = x_dom + flat_dom_y + flat_dom_z;
                    int buf_idx = x_buf + flat_buf_y + flat_buf_z;
                    (*grid)[dom_idx] += (*buf)[buf_idx];
                    if (valid) (*grid_num)[dom_idx] += (*buf_num)[buf_idx];
                }
            }
        }
    } else {
        auto* buf = bbuf.vector_buffer();
        auto* grid = bgrid.vector_buffer();
        if (buf && grid) {
            for (int z_buf = 0; z_buf < buf_cb_.width[2]; z_buf++) {
                int z_dom = z_buf + (buf_cb_.origin[2] - dom_cb_.origin[2]);
                if (z_dom < 0) z_dom += cells_;
                else if (z_dom >= cells_) z_dom -= cells_;
                if (z_dom >= dom_cb_.width[2]) continue;

                int flat_dom_z = z_dom * dom_cb_.width[1] * dom_cb_.width[0];
                int flat_buf_z = z_buf * buf_cb_.width[1] * buf_cb_.width[0];

                for (int y_buf = 0; y_buf < buf_cb_.width[1]; y_buf++) {
                    int y_dom = y_buf + (buf_cb_.origin[1] - dom_cb_.origin[1]);
                    if (y_dom < 0) y_dom += cells_;
                    else if (y_dom >= cells_) y_dom -= cells_;
                    if (y_dom >= dom_cb_.width[1]) continue;

                    int flat_dom_y = y_dom * dom_cb_.width[0];
                    int flat_buf_y = y_buf * buf_cb_.width[0];

                    for (int x_buf = 0; x_buf < buf_cb_.width[0]; x_buf++) {
                        int x_dom = x_buf + (buf_cb_.origin[0] - dom_cb_.origin[0]);
                        if (x_dom < 0) x_dom += cells_;
                        else if (x_dom >= cells_) x_dom -= cells_;
                        if (x_dom >= dom_cb_.width[0]) continue;

                        int dom_idx = x_dom + flat_dom_y + flat_dom_z;
                        int buf_idx = x_buf + flat_buf_y + flat_buf_z;
                        (*grid)[dom_idx][0] += (*buf)[buf_idx][0];
                        (*grid)[dom_idx][1] += (*buf)[buf_idx][1];
                        (*grid)[dom_idx][2] += (*buf)[buf_idx][2];
                        if (valid) (*grid_num)[dom_idx] += (*buf_num)[buf_idx];
                    }
                }
            }
        }
    }
}

void SegmentOverlap3D::interpolate(
    density::Buffer& bbuf,
    const std::vector<geom::Vec3f>& xs,
    const std::vector<geom::Vec3f>& /* vs */,
    double pt_val,
    density::Buffer& bweights,
    int low, int high, int jump
) {
    int length = buf_cb_.width[0];
    int area = buf_cb_.width[0] * buf_cb_.width[1];

    auto* buf = bbuf.scalar_buffer();
    auto* wbuf = bweights.scalar_buffer();
    auto* vbuf = bbuf.vector_buffer();
    auto* vwbuf = bweights.vector_buffer();
    auto* counts = bbuf.count_buffer();
    bool count_valid = (counts != nullptr);
    int wlen = bweights.length();

    if (vbuf && vwbuf) {
        for (int idx = low; idx < high; idx += jump) {
            const auto& pt = xs[idx];
            int x = static_cast<int>(pt[0]);
            int y = static_cast<int>(pt[1]);
            int z = static_cast<int>(pt[2]);
            int buf_idx = x + y * length + z * area;
            for (int dim = 0; dim < 3; dim++) {
                (*vbuf)[buf_idx][dim] += pt_val * (*vwbuf)[idx][dim];
            }
            if (count_valid) (*counts)[buf_idx]++;
        }
    } else if (wlen == 0) {
        for (int idx = low; idx < high; idx += jump) {
            const auto& pt = xs[idx];
            int x = static_cast<int>(pt[0]);
            int y = static_cast<int>(pt[1]);
            int z = static_cast<int>(pt[2]);
            int buf_idx = x + y * length + z * area;
            (*buf)[buf_idx] += pt_val;
            if (count_valid) (*counts)[buf_idx]++;
        }
    } else if (buf && wbuf) {
        for (int idx = low; idx < high; idx += jump) {
            const auto& pt = xs[idx];
            int x = static_cast<int>(pt[0]);
            int y = static_cast<int>(pt[1]);
            int z = static_cast<int>(pt[2]);
            int buf_idx = x + y * length + z * area;
            (*buf)[buf_idx] += pt_val * (*wbuf)[idx];
            if (count_valid) (*counts)[buf_idx]++;
        }
    }
}

void DomainOverlap3D::interpolate(
    density::Buffer& bbuf,
    const std::vector<geom::Vec3f>& xs,
    const std::vector<geom::Vec3f>& /* vs */,
    double pt_val,
    density::Buffer& bweights,
    int low, int high, int jump
) {
    int length = buf_cb_.width[0];
    int area = buf_cb_.width[0] * buf_cb_.width[1];

    auto* buf = bbuf.scalar_buffer();
    auto* wbuf = bweights.scalar_buffer();
    auto* vbuf = bbuf.vector_buffer();
    auto* vwbuf = bweights.vector_buffer();
    auto* counts = bbuf.count_buffer();
    bool count_valid = (counts != nullptr);
    int wlen = bweights.length();

    for (int idx = low; idx < high; idx += jump) {
        const auto& pt = xs[idx];
        int i = bound(int_floor(pt[0]), cells_);
        if (i < buf_cb_.width[0]) {
            int j = bound(int_floor(pt[1]), cells_);
            if (j < buf_cb_.width[1]) {
                int k = bound(int_floor(pt[2]), cells_);
                if (k < buf_cb_.width[2]) {
                    int buf_idx = i + j * length + k * area;
                    if (vbuf && vwbuf) {
                        for (int dim = 0; dim < 3; dim++) {
                            (*vbuf)[buf_idx][dim] += pt_val * (*vwbuf)[idx][dim];
                        }
                        if (count_valid) (*counts)[buf_idx]++;
                    } else if (wlen == 0) {
                        (*buf)[buf_idx] += pt_val;
                        if (count_valid) (*counts)[buf_idx]++;
                    } else if (buf && wbuf) {
                        (*buf)[buf_idx] += pt_val * (*wbuf)[idx];
                        if (count_valid) (*counts)[buf_idx]++;
                    }
                }
            }
        }
    }
}

} // namespace render
} // namespace tessera
