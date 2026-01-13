#include "render/hist.h"
#include "render/manager.h"  // for generate_unit_bufs
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cctype>
#include <random>

namespace tessera {
namespace render {

// ============================================================================
// GridHeader I/O
// ============================================================================

GridHeader read_grid_header(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open grid file: " + filename);
    }
    
    GridHeader hd;
    file.read(reinterpret_cast<char*>(&hd), sizeof(GridHeader));
    
    if (!file) {
        throw std::runtime_error("Failed to read grid header from: " + filename);
    }
    
    return hd;
}

std::vector<double> read_grid(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open grid file: " + filename);
    }
    
    GridHeader hd;
    file.read(reinterpret_cast<char*>(&hd), sizeof(GridHeader));
    
    if (!file) {
        throw std::runtime_error("Failed to read grid header from: " + filename);
    }
    
    if (hd.type.is_vector_grid == 1) {
        throw std::runtime_error("read_grid() can only read scalar grids");
    }
    
    // Calculate grid size
    int64_t size = hd.loc.pixel_span[0] * hd.loc.pixel_span[1] * hd.loc.pixel_span[2];
    
    // Read float32 values
    std::vector<float> val32s(size);
    file.read(reinterpret_cast<char*>(val32s.data()), size * sizeof(float));
    
    if (!file) {
        throw std::runtime_error("Failed to read grid data from: " + filename);
    }
    
    // Convert to float64
    std::vector<double> vals(size);
    for (int64_t i = 0; i < size; i++) {
        vals[i] = static_cast<double>(val32s[i]);
    }
    
    return vals;
}

// ============================================================================
// HistInfo
// ============================================================================

bool HistInfo::is_log() const {
    std::string lower_scale = scale;
    for (auto& c : lower_scale) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower_scale == "log";
}

// ============================================================================
// HistBox
// ============================================================================

HistBox HistBox::from_config(double x, double y, double z,
                              double x_width, double y_width, double z_width,
                              int bins) {
    HistBox box;
    box.origin = {x, y, z};
    box.span = {x_width, y_width, z_width};
    box.centers.resize(bins);
    box.counts.resize(bins, 0);
    return box;
}

bool HistBox::contains(const geom::Vec3f& v, double L) const {
    for (int k = 0; k < 3; k++) {
        double a = origin[k];
        double a_span = span[k];
        double b = static_cast<double>(v[k]);
        
        // Normalize b to [0, L)
        if (b >= L) b -= L;
        
        // Check if b is within [a, a + a_span) with periodic wrapping
        bool in_range = (b > a && a + a_span > b) ||
                        (b < a && a + a_span - L > b);
        if (!in_range) return false;
    }
    return true;
}

// ============================================================================
// HistManager
// ============================================================================

HistManager::HistManager(
    const std::vector<std::string>& files,
    std::vector<HistBox>& boxes,
    int points,
    const std::string& quantity,
    const std::string& grid_file
) : files_(files), boxes_(boxes), quantity_(quantity)
{
    if (files.empty()) {
        throw std::runtime_error("No files provided to HistManager");
    }
    
    // Read first header to get grid dimensions
    // Note: In a real implementation, you'd use io::read_sheet_header_at
    // For now, we initialize xs_ based on expected size
    
    // Create unit buffers for Monte Carlo sampling
    unit_bufs_ = generate_unit_bufs(UNIT_BUF_COUNT, points);
    
    // Set up workers
    workers_ = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    
    // Read grid file
    grid_hd_ = std::make_unique<GridHeader>(read_grid_header(grid_file));
    grid_ = read_grid(grid_file);
}

void HistManager::subsample(int skip) {
    if (!is_pow_two(skip)) {
        throw std::runtime_error("Subsample skip must be power of two");
    }
    skip_ = skip;
}

std::vector<double> HistManager::hist_centers(const HistInfo& info) {
    double min_val = info.min;
    double max_val = info.max;
    
    bool is_log = info.is_log();
    if (is_log) {
        min_val = std::log10(min_val);
        max_val = std::log10(max_val);
    }
    
    double dx = (max_val - min_val) / static_cast<double>(info.bins);
    
    std::vector<double> centers(info.bins);
    for (int i = 0; i < info.bins; i++) {
        centers[i] = min_val + dx * (static_cast<double>(i) + 0.5);
        if (is_log) {
            centers[i] = std::pow(10.0, centers[i]);
        }
    }
    
    return centers;
}

void HistManager::hist(const HistInfo& info) {
    // Set up workspaces
    int pts = static_cast<int>(unit_bufs_[0].size());
    
    hists_.resize(workers_);
    qs_.resize(workers_);
    in_box_.resize(workers_);
    vec_bufs_.resize(workers_);
    
    for (int i = 0; i < workers_; i++) {
        hists_[i].resize(info.bins, 0);
        qs_[i].resize(pts);
        in_box_[i].resize(pts);
        vec_bufs_[i].resize(pts);
    }
    
    // Initialize box output
    for (auto& box : boxes_) {
        box.counts.assign(info.bins, 0);
        box.centers = hist_centers(info);
    }
    
    // Loop over files
    for (size_t i = 0; i < files_.size(); i++) {
        std::cout << "Analyzed files " << i << "/" << files_.size() << std::endl;
        hist_from_file(files_[i], info);
    }
}

bool HistManager::hist_intersect(const io::SheetHeader& sheet, const HistBox& box) {
    for (int k = 0; k < 3; k++) {
        if (!intersect_1d(
            static_cast<double>(sheet.origin[k]),
            static_cast<double>(sheet.width[k]),
            box.origin[k], box.span[k],
            sheet.total_width
        )) {
            return false;
        }
    }
    return true;
}

bool HistManager::intersect_1d(double a, double a_span, double b, double b_span, double L) {
    return contains_1d(a, a_span, b, L) ||
           contains_1d(a, a_span, b + b_span, L) ||
           contains_1d(b, b_span, a, L) ||
           contains_1d(b, b_span, a + a_span, L);
}

bool HistManager::contains_1d(double a, double a_span, double b, double L) {
    if (b >= L) b -= L;
    return (b > a && a + a_span > b) ||
           (b < a && a + a_span - L > b);
}

void HistManager::hist_from_file(const std::string& file, const HistInfo& info) {
    // In a real implementation, read the sheet header and positions here:
    // io::read_sheet_header_at(file, hd_);
    // io::read_sheet_positions_at(file, xs_);
    
    std::vector<int> out(workers_);
    std::mutex out_mutex;
    
    for (auto& box : boxes_) {
        if (!hist_intersect(hd_, box)) continue;
        
        // Launch worker threads
        std::vector<std::thread> threads;
        for (int id = 0; id < workers_; id++) {
            threads.emplace_back([this, id, &box, &info, &out, &out_mutex]() {
                chan_histogram(id, box, info, out, out_mutex);
            });
        }
        
        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }
        
        // Merge worker histograms into box histogram
        for (int id = 0; id < workers_; id++) {
            for (int j = 0; j < info.bins; j++) {
                box.counts[j] += hists_[id][j];
            }
        }
    }
}

void HistManager::chan_histogram(int worker, HistBox& box, const HistInfo& info,
                                  std::vector<int>& out, std::mutex& out_mutex) {
    // Clear histogram buffer
    std::fill(hists_[worker].begin(), hists_[worker].end(), 0);
    
    int grid_width = static_cast<int>(hd_.grid_width);
    int seg_width = static_cast<int>(hd_.segment_width);
    auto& hist = hists_[worker];
    auto& qs = qs_[worker];
    auto& in_box = in_box_[worker];
    auto& vec_buf = vec_bufs_[worker];
    
    // Convert quantity to lowercase for comparison
    std::string lower_quantity = quantity_;
    for (auto& c : lower_quantity) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    
    if (lower_quantity == "density") {
        for (int z = 0; z < seg_width; z += skip_) {
            for (int y = 0; y < seg_width; y += skip_) {
                for (int x = 0; x < seg_width; x += skip_) {
                    int idx = x + y * grid_width + z * grid_width * grid_width;
                    
                    // Distribute work across workers
                    if (idx % (skip_ * skip_ * skip_ * workers_) != worker) {
                        continue;
                    }
                    
                    if (!cube_intersects(idx)) continue;
                    
                    for (int dir = 0; dir < geom::TETRA_DIR_COUNT; dir++) {
                        get_densities(idx, dir, box, vec_buf, qs, in_box);
                        histogram(qs, in_box, info, hist);
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("Non-implemented quantity: " + quantity_);
    }
    
    {
        std::lock_guard<std::mutex> lock(out_mutex);
        out[worker] = worker;
    }
}

bool HistManager::cube_intersects(int idx) const {
    int w = static_cast<int>(hd_.grid_width);
    int ix = idx % w;
    int iy = (idx / w) % w;
    int iz = idx / (w * w);
    
    // Sanity check
    if (ix < 0 || skip_ + ix >= w ||
        iy < 0 || skip_ + iy >= w ||
        iz < 0 || skip_ + iz >= w) {
        return false;
    }
    
    auto [origin, span] = cube_bounding_box(ix, iy, iz);
    
    for (int k = 0; k < 3; k++) {
        if (!intersect_1d(
            static_cast<double>(hd_.origin[k]),
            static_cast<double>(hd_.width[k]),
            static_cast<double>(origin[k]),
            static_cast<double>(span[k]),
            hd_.total_width
        )) {
            return false;
        }
    }
    return true;
}

std::pair<geom::Vec3f, geom::Vec3f> HistManager::cube_bounding_box(int ix, int iy, int iz) const {
    int jump = skip_;
    int w = static_cast<int>(hd_.grid_width);
    float L = static_cast<float>(hd_.total_width);
    
    geom::Vec3f origin = xs_[ix + iy * w + iz * w * w];
    geom::Vec3f span = {0.0f, 0.0f, 0.0f};
    
    for (int dz = 0; dz < jump; dz += jump) {
        for (int dy = 0; dy < jump; dy += jump) {
            for (int dx = 0; dx < jump; dx += jump) {
                int x = ix + dx;
                int y = iy + dy;
                int z = iz + dz;
                const geom::Vec3f& vec = xs_[x + y * w + z * w * w];
                
                for (int k = 0; k < 3; k++) {
                    float delta = vec[k] - origin[k];
                    if (delta > L / 2) {
                        delta -= L;
                    } else if (delta < -L / 2) {
                        delta += L;
                    }
                    
                    if (delta > 0 && delta > span[k]) {
                        span[k] = delta;
                    } else if (delta < 0) {
                        origin[k] -= delta;
                        if (origin[k] < 0) origin[k] += L;
                        span[k] += delta;
                    }
                }
            }
        }
    }
    
    return {origin, span};
}

void HistManager::get_densities(int i, int dir, const HistBox& box,
                                 std::vector<geom::Vec3f>& vec_buf,
                                 std::vector<double>& densities,
                                 std::vector<bool>& in_box_flags) const {
    float L = static_cast<float>(hd_.total_width);
    geom::Vec3f origin_f, span_f;
    for (int k = 0; k < 3; k++) {
        origin_f[k] = static_cast<float>(box.origin[k]);
        span_f[k] = static_cast<float>(box.span[k]);
    }
    
    // Initialize tetrahedron indices and tetrahedron
    geom::TetraIdxs idx_buf(static_cast<int64_t>(i), hd_.grid_width,
                            static_cast<int64_t>(skip_), dir);
    geom::Tetra tet;
    tet.init(xs_[idx_buf[0]], xs_[idx_buf[1]], xs_[idx_buf[2]], xs_[idx_buf[3]]);
    
    // Generate random buffer index and distribute points
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, unit_bufs_.size() - 1);
    size_t buf_idx = dist(rng);
    tet.distribute_tetra(unit_bufs_[buf_idx], vec_buf);
    
    for (size_t j = 0; j < vec_buf.size(); j++) {
        // Put vectors back into periodic box
        for (int k = 0; k < 3; k++) {
            if (vec_buf[j][k] >= L) vec_buf[j][k] -= L;
            if (vec_buf[j][k] < 0) vec_buf[j][k] += L;
        }
        
        // Check if point is in range
        if (!box.contains(vec_buf[j], hd_.total_width)) {
            densities[j] = -1;
            in_box_flags[j] = false;
            continue;
        }
        
        // Interpolate from density grid
        int grid_idx = grid_index(vec_buf[j]);
        if (grid_idx >= 0) {
            densities[j] = grid_[grid_idx];
            in_box_flags[j] = true;
        } else {
            densities[j] = -1;
            in_box_flags[j] = false;
        }
    }
}

int HistManager::grid_index(const geom::Vec3f& vec) const {
    std::array<int, 3> idx;
    double L = grid_hd_->cosmo.box_width;
    
    for (int k = 0; k < 3; k++) {
        double delta = static_cast<double>(vec[k]) - grid_hd_->loc.origin[k];
        if (delta < 0) delta += L;
        if (delta >= grid_hd_->loc.span[k]) delta -= L;
        
        if (delta < 0 || delta >= grid_hd_->loc.span[k]) return -1;
        idx[k] = static_cast<int>(delta / grid_hd_->loc.pixel_width);
    }
    
    return idx[0] + idx[1] * static_cast<int>(grid_hd_->loc.pixel_span[0]) +
           idx[2] * static_cast<int>(grid_hd_->loc.pixel_span[0] * grid_hd_->loc.pixel_span[1]);
}

void HistManager::histogram(const std::vector<double>& x,
                             const std::vector<bool>& ok,
                             const HistInfo& info,
                             std::vector<int>& counts) {
    double min_val = info.min;
    double max_val = info.max;
    double f_bins = static_cast<double>(info.bins);
    
    if (info.is_log()) {
        double log_min = std::log10(min_val);
        double log_max = std::log10(max_val);
        double dx = (log_max - log_min) / f_bins;
        
        for (size_t i = 0; i < x.size(); i++) {
            if (!ok[i]) continue;
            
            double idx = (std::log10(x[i]) - log_min) / dx;
            if (idx < 0 || idx >= f_bins) continue;
            
            counts[static_cast<int>(idx)]++;
        }
    } else {
        double dx = (max_val - min_val) / f_bins;
        
        for (size_t i = 0; i < x.size(); i++) {
            if (!ok[i]) continue;
            
            double idx = (x[i] - min_val) / dx;
            if (idx < 0 || idx >= f_bins) continue;
            
            counts[static_cast<int>(idx)]++;
        }
    }
}

} // namespace render
} // namespace tessera
