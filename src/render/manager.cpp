#include "render/manager.h"
#include "math/rand.h"
#include "geom/tetra.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <mutex>

namespace tessera {
namespace render {

std::vector<std::vector<geom::Vec3f>> generate_unit_bufs(int n_unit, int pts) {
    math::Generator gen = math::Generator::new_time_seed(math::GeneratorType::Tausworthe);
    std::vector<std::vector<geom::Vec3f>> bufs(n_unit);

    for (int bi = 0; bi < n_unit; bi++) {
        bufs[bi].resize(pts);
        for (int j = 0; j < pts; j++) {
            for (int k = 0; k < 3; k++) {
                bufs[bi][j][k] = static_cast<float>(gen.uniform(0.0, 1.0));
            }
        }
        geom::distribute_unit(bufs[bi]);
    }

    return bufs;
}

Manager::Manager(
    const std::vector<std::string>& files,
    std::vector<std::unique_ptr<Box>>& boxes,
    bool log_flag,
    density::Quantity q
) : log_(log_flag), q_(q)
{
    // Find max points needed
    int max_points = 0;
    for (const auto& b : boxes) {
        if (b->points() > max_points) max_points = b->points();
    }
    
    // Generate unit buffers
    unit_bufs_ = generate_unit_bufs(UNIT_BUF_COUNT, max_points);

    skip_ = 1;

    // Set up workers
    workers_ = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    workspaces_.resize(workers_);

    // Set up renderers
    renderers_.resize(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++) {
        renderers_[i].box = boxes[i].get();
        auto origin = boxes[i]->cell_origin();
        auto span = boxes[i]->cell_span();
        renderers_[i].cb.origin = origin;
        renderers_[i].cb.width = span;
        
        renderers_[i].g = std::make_unique<geom::GridLocation>(
            origin, span,
            boxes[i]->cell_width() * static_cast<double>(boxes[i]->num_cells()),
            boxes[i]->num_cells()
        );
    }

    // Filter files to only those that intersect with boxes
    int max_buf_size = 0;
    
    for (const auto& file : files) {
        // Read header
        // Note: In a real implementation, you'd call io::read_sheet_header_at(file, hd_)
        // For now, we'll just track files
        
        bool intersect = false;
        for (size_t i = 0; i < boxes.size(); i++) {
            int cells = boxes[i]->num_cells();
            geom::CellBounds h_cb = hd_.cell_bounds(cells);

            if (h_cb.intersect(renderers_[i].cb, cells)) {
                auto over = boxes[i]->overlap(hd_);
                int buf_size = over->buffer_size();
                if (buf_size > max_buf_size) max_buf_size = buf_size;

                renderers_[i].valid_segs[file] = true;
                intersect = true;
            }
        }

        if (intersect) {
            files_.push_back(file);
        }
    }

    // Initialize xs/vs buffers
    xs_.resize(hd_.grid_count);
    scaled_xs_.resize(hd_.grid_count);
    if (requires_velocity(q)) {
        vs_.resize(hd_.grid_count);
    }

    if (log_) {
        std::cout << "Workspace buffer size: " << max_buf_size 
                  << ". Number of workers: " << workers_ << std::endl;
    }

    // Initialize workspaces
    for (int i = 0; i < workers_; i++) {
        workspaces_[i].buf = density::create_buffer(
            q, max_buf_size, max_points, *renderers_[0].g
        );
    }
}

void Manager::subsample(int subsample_length) {
    if (!is_pow_two(skip_)) {
        throw std::runtime_error("Skip increment must be power of two");
    }
    skip_ = subsample_length;
}

void Manager::render_density() {
    for (const auto& file : files_) {
        render_density_from_file(file);
    }
}

void Manager::render_density_from_file(const std::string& file) {
    if (log_) {
        std::cout << "Rendering file " << file << std::endl;
    }

    load_file(file);

    std::vector<int> out(workers_);
    std::mutex out_mutex;

    for (auto& r : renderers_) {
        if (!r.requires_file(file)) continue;

        x_cb_ = hd_.cell_bounds(r.box->num_cells());
        r.over = r.box->overlap(hd_);
        r.cb.origin = r.box->cell_origin();
        r.cb.width = r.box->cell_span();
        scale_xs(r);
        init_workspaces(r);

        // Multi-threaded interpolation
        std::vector<std::thread> threads;
        for (int id = 0; id < workers_ - 1; id++) {
            threads.emplace_back([this, id, &r, &out, &out_mutex]() {
                chan_interpolate(id, r, out);
            });
        }

        // Last worker runs in main thread
        chan_interpolate(workers_ - 1, r, out);

        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }

        // Add results from all workers
        for (int i = 0; i < workers_; i++) {
            r.over->add(*workspaces_[i].buf, r.box->vals());
        }
    }
}

void Manager::load_file(const std::string& /* file */) {
    // In a real implementation, this would:
    // 1. Read sheet header from file
    // 2. Read positions into xs_
    // 3. Read velocities into vs_ if needed
    //
    // io::read_sheet_header_at(file, hd_);
    // io::read_sheet_positions_at(file, xs_);
    // if (requires_velocity(q_)) {
    //     io::read_sheet_velocities_at(file, vs_);
    // }
}

void Manager::scale_xs(Renderer& r) {
    scaled_xs_ = xs_;  // Copy
    r.over->scale_vecs(scaled_xs_, x_cb_);
}

void Manager::init_workspaces(Renderer& r) {
    int seg_frac = static_cast<int>(hd_.segment_width) / skip_;
    int seg_len = seg_frac * seg_frac * seg_frac;

    // Trim unit buffers to actual points needed
    for (auto& buf : unit_bufs_) {
        buf.resize(r.box->points());
    }

    for (int id = 0; id < workers_; id++) {
        workspaces_[id].intr = std::make_shared<density::MonteCarlo>(
            hd_.segment_width,
            r.box->points(),
            r.box->num_cells(),
            static_cast<int64_t>(skip_),
            unit_bufs_,
            std::shared_ptr<density::Interpolator>(r.over.get(), [](density::Interpolator*){})
        );

        workspaces_[id].buf->slice(0, r.over->buffer_size());
        workspaces_[id].buf->set_grid_location(*r.g);
        workspaces_[id].buf->clear();

        workspaces_[id].low_x = id;
        workspaces_[id].high_x = seg_len;
    }
}

double Manager::pt_val(const Renderer& r) const {
    if (requires_velocity(q_)) {
        return 1.0;
    } else {
        double frac = static_cast<double>(r.box->num_cells()) / 
                      static_cast<double>(hd_.count_width);
        return frac * frac * frac;
    }
}

void Manager::chan_interpolate(int id, Renderer& r, std::vector<int>& out) {
    auto& w = workspaces_[id];

    // Use tetrahedron-based interpolation
    static density::NilBuffer nil_buf;
    w.intr->interpolate(
        *w.buf, scaled_xs_, vs_,
        pt_val(r), nil_buf,
        w.low_x, w.high_x, workers_
    );

    out[id] = id;
}

} // namespace render
} // namespace tessera
