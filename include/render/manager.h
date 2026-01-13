#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <thread>
#include "geom/vec.h"
#include "geom/grid.h"
#include "density/quantity.h"
#include "density/buffer.h"
#include "density/interpolator.h"
#include "render/box.h"
#include "io/headers.h"

namespace tessera {
namespace render {

constexpr int UNIT_BUF_COUNT = 64;  // 1 << 6

/**
 * Workspace for a single worker thread.
 */
struct Workspace {
    std::unique_ptr<density::Buffer> buf;
    std::shared_ptr<density::Interpolator> intr;
    int low_x = 0;
    int high_x = 0;
};

/**
 * Internal renderer state for a single box.
 */
struct Renderer {
    Box* box = nullptr;
    std::unique_ptr<geom::GridLocation> g;
    geom::CellBounds cb;
    std::unique_ptr<Overlap> over;
    std::map<std::string, bool> valid_segs;

    bool requires_file(const std::string& file) const {
        auto it = valid_segs.find(file);
        return it != valid_segs.end() && it->second;
    }
};

/**
 * Rendering manager that coordinates density field computation.
 * 
 * This class manages the rendering of phase-space sheet data onto
 * density grids, handling file I/O, workload distribution, and
 * multi-threaded processing.
 */
class Manager {
public:
    /**
     * Create a new rendering manager.
     * 
     * @param files List of sheet segment files to process
     * @param boxes Rendering boxes defining output regions
     * @param log_flag Whether to enable logging
     * @param q Quantity to compute
     * @throws std::runtime_error if initialization fails
     */
    Manager(
        const std::vector<std::string>& files,
        std::vector<std::unique_ptr<Box>>& boxes,
        bool log_flag,
        density::Quantity q
    );

    ~Manager() = default;

    /// Enable or disable logging
    void set_log(bool flag) { log_ = flag; }

    /// Set subsampling factor (must be power of 2)
    void subsample(int subsample_length);

    /// Render density from all files
    void render_density();

    /// Render density from a single file
    void render_density_from_file(const std::string& file);

    /// Get number of workers
    int num_workers() const { return workers_; }

private:
    // Currently loaded sheet segment
    std::vector<geom::Vec3f> xs_;
    std::vector<geom::Vec3f> vs_;
    std::vector<geom::Vec3f> scaled_xs_;
    geom::CellBounds x_cb_;
    io::SheetHeader hd_;

    std::vector<Renderer> renderers_;
    int skip_ = 1;
    std::vector<std::vector<geom::Vec3f>> unit_bufs_;

    // I/O settings
    bool log_ = false;
    std::vector<std::string> files_;

    // Workspaces
    density::Quantity q_;
    int workers_ = 1;
    std::vector<Workspace> workspaces_;

    // Initialize unit buffers for Monte Carlo sampling
    void init_unit_bufs(int n_unit, int pts);

    // Load a sheet segment file
    void load_file(const std::string& file);

    // Scale positions for a renderer
    void scale_xs(Renderer& r);

    // Initialize workspaces for a renderer
    void init_workspaces(Renderer& r);

    // Calculate point value for a renderer
    double pt_val(const Renderer& r) const;

    // Worker interpolation function
    void chan_interpolate(int id, Renderer& r, std::vector<int>& out);

    // Check if x is a power of two
    static bool is_pow_two(int x) {
        while ((x & 1) == 0 && x > 0) x >>= 1;
        return x == 1;
    }
};

/**
 * Generate unit buffers for Monte Carlo sampling.
 */
std::vector<std::vector<geom::Vec3f>> generate_unit_bufs(int n_unit, int pts);

} // namespace render
} // namespace tessera
