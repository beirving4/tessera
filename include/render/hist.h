#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include "geom/vec.h"
#include "geom/tetra.h"
#include "io/headers.h"

namespace tessera {
namespace render {

/**
 * GridLocation contains spatial information about a grid.
 */
struct GridLocation {
    std::array<double, 3> origin = {0, 0, 0};
    std::array<double, 3> span = {0, 0, 0};
    std::array<int64_t, 3> pixel_origin = {0, 0, 0};
    std::array<int64_t, 3> pixel_span = {0, 0, 0};
    double pixel_width = 0.0;
};

/**
 * CosmoInfo contains cosmological parameters.
 */
struct CosmoInfo {
    double redshift = 0.0;
    double scale_factor = 0.0;
    double omega_m = 0.0;
    double omega_l = 0.0;
    double hubble = 0.0;
    double rho_mean = 0.0;
    double rho_critical = 0.0;
    double box_width = 0.0;
};

/**
 * RenderInfo contains rendering parameters.
 */
struct RenderInfo {
    int64_t particles = 0;
    int64_t total_pixels = 0;
    int64_t subsample_length = 0;
    int64_t min_projection_depth = 0;
    int64_t projection_axis = 0;
};

/**
 * TypeInfo contains grid type information.
 */
struct TypeInfo {
    int64_t header_size = 0;
    int64_t grid_type = 0;
    int64_t is_vector_grid = 0;
};

/**
 * GridHeader contains complete metadata for a density grid file.
 */
struct GridHeader {
    uint64_t endianness_version = 0;
    TypeInfo type;
    CosmoInfo cosmo;
    RenderInfo render;
    GridLocation loc;
    GridLocation vel;
};

/**
 * Read a GridHeader from a binary file.
 */
GridHeader read_grid_header(const std::string& filename);

/**
 * Read a scalar grid from a binary file.
 */
std::vector<double> read_grid(const std::string& filename);

/**
 * HistInfo contains histogram configuration parameters.
 */
struct HistInfo {
    double min = 0.0;
    double max = 0.0;
    int bins = 0;
    std::string scale = "linear";  // "linear" or "log"
    
    bool is_log() const;
};

/**
 * HistBox represents a spatial region with associated histogram data.
 */
struct HistBox {
    std::array<double, 3> origin = {0, 0, 0};
    std::array<double, 3> span = {0, 0, 0};
    
    std::vector<double> centers;  // Bin centers
    std::vector<int> counts;      // Histogram counts
    
    /**
     * Create a HistBox from box configuration.
     */
    static HistBox from_config(double x, double y, double z,
                               double x_width, double y_width, double z_width,
                               int bins);
    
    /**
     * Check if a point is contained within this box (with periodic boundaries).
     */
    bool contains(const geom::Vec3f& v, double L) const;
};

/**
 * HistManager manages parallel histogram computation from phase sheet data.
 * 
 * This class computes mass-weighted density histograms by sampling points
 * uniformly within tetrahedra and interpolating density values from a grid.
 */
class HistManager {
public:
    /**
     * Create a new histogram manager.
     * 
     * @param files List of sheet segment files to process
     * @param boxes Histogram boxes defining spatial regions
     * @param points Number of sample points per tetrahedron
     * @param quantity Name of quantity to histogram (e.g., "Density")
     * @param grid_file Path to density grid file for interpolation
     */
    HistManager(
        const std::vector<std::string>& files,
        std::vector<HistBox>& boxes,
        int points,
        const std::string& quantity,
        const std::string& grid_file
    );
    
    ~HistManager() = default;
    
    /**
     * Set subsampling factor (must be power of 2).
     */
    void subsample(int skip);
    
    /**
     * Compute histograms for all boxes using the given histogram info.
     */
    void hist(const HistInfo& info);
    
    /**
     * Process a single file and update histograms.
     */
    void hist_from_file(const std::string& file, const HistInfo& info);
    
    /**
     * Get number of workers.
     */
    int num_workers() const { return workers_; }
    
    /**
     * Get the histogram boxes (with computed results).
     */
    std::vector<HistBox>& boxes() { return boxes_; }
    const std::vector<HistBox>& boxes() const { return boxes_; }

private:
    // Sheet data
    std::vector<geom::Vec3f> xs_;
    io::SheetHeader hd_;
    std::vector<std::string> files_;
    
    // Histogram boxes
    std::vector<HistBox>& boxes_;
    std::vector<std::vector<geom::Vec3f>> unit_bufs_;
    
    // Per-worker buffers
    std::vector<std::vector<int>> hists_;
    std::vector<std::vector<double>> qs_;
    std::vector<std::vector<bool>> in_box_;
    std::vector<std::vector<geom::Vec3f>> vec_bufs_;
    
    int skip_ = 1;
    std::string quantity_;
    int workers_ = 1;
    
    // Grid data for density interpolation
    std::unique_ptr<GridHeader> grid_hd_;
    std::vector<double> grid_;
    
    // Generate histogram bin centers
    static std::vector<double> hist_centers(const HistInfo& info);
    
    // Check if sheet intersects with box
    static bool hist_intersect(const io::SheetHeader& sheet, const HistBox& box);
    
    // 1D containment check with periodic boundaries
    static bool contains_1d(double a, double a_span, double b, double L);
    
    // 1D intersection check with periodic boundaries
    static bool intersect_1d(double a, double a_span, double b, double b_span, double L);
    
    // Worker function for parallel histogram computation
    void chan_histogram(int worker, HistBox& box, const HistInfo& info,
                        std::vector<int>& out, std::mutex& out_mutex);
    
    // Check if cube at idx intersects with loaded sheet
    bool cube_intersects(int idx) const;
    
    // Get bounding box of a Lagrangian cube
    std::pair<geom::Vec3f, geom::Vec3f> cube_bounding_box(int ix, int iy, int iz) const;
    
    // Compute densities for points in a tetrahedron
    void get_densities(int i, int dir, const HistBox& box,
                       std::vector<geom::Vec3f>& vec_buf,
                       std::vector<double>& densities,
                       std::vector<bool>& in_box) const;
    
    // Get grid index for a point
    int grid_index(const geom::Vec3f& vec) const;
    
    // Add values to histogram
    static void histogram(const std::vector<double>& x,
                         const std::vector<bool>& ok,
                         const HistInfo& info,
                         std::vector<int>& counts);
    
    // Check if x is power of two
    static bool is_pow_two(int x) {
        while ((x & 1) == 0 && x > 0) x >>= 1;
        return x == 1;
    }
};

} // namespace render
} // namespace tessera
