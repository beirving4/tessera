#include "density/interpolator.h"
#include <cmath>

namespace asymptotic_tetra {
namespace density {

MonteCarlo::MonteCarlo(
    int64_t seg_width,
    int points,
    int cells,
    int64_t skip,
    std::vector<std::vector<geom::Vec3f>>& unit_bufs,
    std::shared_ptr<Interpolator> sub_intr
) : sub_intr_(sub_intr),
    seg_width_(seg_width),
    points_(points),
    cells_(cells),
    skip_(skip),
    gen_(math::Generator::new_time_seed(math::GeneratorType::Xorshift)),
    unit_bufs_(unit_bufs),
    vec_buf_(points)
{
}

void MonteCarlo::interpolate(
    Buffer& buf,
    const std::vector<geom::Vec3f>& xs,
    const std::vector<geom::Vec3f>& vs,
    double pt_val,
    Buffer& weights,
    int low, int high, int jump
) {
    int64_t grid_width = seg_width_ + 1;
    int64_t idx_width = seg_width_ / skip_;

    bool req_vel = requires_velocity(buf.quantity());
    if (!req_vel) {
        pt_val = pt_val / static_cast<double>(points_) / 6.0 *
            static_cast<double>(skip_ * skip_ * skip_);
    }

    geom::CellBounds tet_cb;

    geom::CellBounds rel_cb;
    rel_cb.width = domain_cell_bounds()->width;
    cb_subtr(*domain_cell_bounds(), *buffer_cell_bounds(),
             rel_cb.origin[0], rel_cb.origin[1], rel_cb.origin[2]);

    int64_t jump64 = static_cast<int64_t>(jump);

    // Determine if we need to apply periodic wrapping
    bool never_mod = (*sub_intr_->domain_cell_bounds() != *sub_intr_->buffer_cell_bounds()) ||
        (cells() / 2 > rel_cb.width[0] &&
         cells() / 2 > rel_cb.width[1] &&
         cells() / 2 > rel_cb.width[2]);

    for (int64_t idx = static_cast<int64_t>(low); idx < static_cast<int64_t>(high); idx += jump64) {
        int64_t x, y, z;
        coords(idx, idx_width, x, y, z);
        int64_t grid_idx = index(x * skip_, y * skip_, z * skip_, grid_width);

        for (int dir = 0; dir < 6; dir++) {
            idx_buf_.init(grid_idx, grid_width, skip_, dir);

            tet_.init(
                xs[idx_buf_[0]],
                xs[idx_buf_[1]],
                xs[idx_buf_[2]],
                xs[idx_buf_[3]]
            );

            if (req_vel) {
                vtet_.init(
                    vs[idx_buf_[0]],
                    vs[idx_buf_[1]],
                    vs[idx_buf_[2]],
                    vs[idx_buf_[3]]
                );
            }

            tet_.cell_bounds_at(1.0f, tet_cb);

            if (!tet_cb.intersect(rel_cb, cells_)) continue;

            int buf_idx = gen_.uniform_int(0, static_cast<int>(unit_bufs_.size()));
            tet_.distribute_tetra(unit_bufs_[buf_idx], vec_buf_);

            // Apply periodic wrapping if needed
            if (!never_mod) {
                for (int j = 0; j < 3; j++) {
                    if (coord_neg(tet_, j)) {
                        mod_coord(vec_buf_, j, static_cast<float>(cells()));
                    }
                }
            }

            // Handle velocity-weighted buffers
            Buffer* bweights = &weights;
            
            if (req_vel) {
                auto* vbuf = weights.vector_buffer();
                if (vbuf) {
                    vtet_.distribute_tetra64(unit_bufs_[buf_idx], *vbuf);
                }
            }

            sub_intr_->interpolate(
                buf, vec_buf_, vs, pt_val, *bweights, 0, points_, 1
            );
        }
    }
}

} // namespace density
} // namespace asymptotic_tetra
