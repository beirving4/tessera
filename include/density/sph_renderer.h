#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include <algorithm>

/**
 * @file sph_renderer.h
 * @brief High-performance SPH-based density field rendering
 *
 * This module provides SPH (Smoothed Particle Hydrodynamics) density rendering
 * as an alternative to phase-space tessellation. SPH rendering has no
 * tessellation artifacts and is particularly useful for late-time cosmological
 * simulations where tetrahedra become highly elongated.
 *
 * Key features:
 * - Multiple kernel functions (cubic spline, Wendland C2, quintic)
 * - SIMD-optimized kernel evaluation (AVX2)
 * - OpenMP parallelization
 * - Adaptive smoothing lengths
 * - Periodic boundary conditions
 * - Memory-efficient grid accumulation
 *
 * Algorithm:
 * 1. Build spatial hash grid for O(1) neighbor lookup
 * 2. For each grid cell, find particles within kernel radius
 * 3. Accumulate kernel-weighted contributions with SIMD
 * 4. Normalize by kernel volume for proper density units
 */

namespace tessera {
namespace density {

/**
 * SPH kernel type selection.
 */
enum class SPHKernel {
    CUBIC_SPLINE,    ///< M4 cubic spline (most common, compact support at 2h)
    WENDLAND_C2,     ///< Wendland C2 (smoother, compact support at 2h)
    WENDLAND_C4,     ///< Wendland C4 (very smooth, compact support at 2h)
    QUINTIC_SPLINE   ///< M6 quintic spline (high accuracy, compact support at 3h)
};

/**
 * Convert kernel enum to string.
 */
const char* sph_kernel_to_string(SPHKernel kernel);

/**
 * Parse kernel from string.
 */
SPHKernel sph_kernel_from_string(const char* str);

/**
 * Configuration for SPH density rendering.
 */
struct SPHConfig {
    // Grid configuration
    int output_cells = 256;         ///< Output grid cells per dimension
    double box_size = 0.0;          ///< Physical size of output region

    // Rendering region (sub-box within simulation)
    std::array<double, 3> center = {0, 0, 0};   ///< Center of rendering region
    double render_width = 0.0;      ///< Width of rendering region (0 = use box_size)

    // SPH parameters
    SPHKernel kernel = SPHKernel::CUBIC_SPLINE;  ///< Kernel function
    double smoothing_length = 0.0;  ///< Fixed smoothing length (0 = adaptive)
    int neighbors_target = 32;      ///< Target neighbors for adaptive smoothing
    double h_min = 0.0;             ///< Minimum smoothing length (0 = no limit)
    double h_max = 0.0;             ///< Maximum smoothing length (0 = no limit)

    // Simulation parameters
    double sim_box_size = 0.0;      ///< Full simulation box size (for periodicity)
    bool periodic = true;           ///< Use periodic boundary conditions
    double particle_mass = 1.0;     ///< Mass per particle

    // Performance
    int n_threads = 0;              ///< Number of threads (0 = auto)
    int projection_axis = 2;        ///< Axis to project along (0=x, 1=y, 2=z)
    double slice_thickness = 0.0;   ///< Slice thickness (0 = full projection)

    /**
     * Get compact support radius for the configured kernel.
     * This is the maximum distance at which the kernel is non-zero.
     */
    double compact_support_radius(double h) const;

    /**
     * Create default configuration for halo rendering.
     * @param halo_center Halo center position
     * @param box_width Physical width of rendering region
     * @param sim_box Full simulation box size
     */
    static SPHConfig for_halo(
        const std::array<double, 3>& halo_center,
        double box_width,
        double sim_box,
        int resolution = 256
    );
};

/**
 * Result of 2D SPH density projection.
 */
struct SPHResult2D {
    std::vector<double> density;    ///< Flattened (cells^2) surface density
    std::vector<int64_t> counts;    ///< Particles contributing to each cell
    int cells = 0;                  ///< Cells per dimension
    double cell_width = 0.0;        ///< Physical cell width
    int projection_axis = 2;        ///< Axis projected along
    double mean_density = 0.0;      ///< Mean surface density
    double total_mass = 0.0;        ///< Total mass in projection

    // Performance metrics
    double render_time_ms = 0.0;    ///< Rendering time in milliseconds
    int64_t n_particles = 0;        ///< Number of particles processed
    int64_t kernel_evals = 0;       ///< Total kernel evaluations

    // Extent information
    std::array<double, 2> extent_x = {0, 0};  ///< X extent [min, max]
    std::array<double, 2> extent_y = {0, 0};  ///< Y extent [min, max]
};

/**
 * Result of 3D SPH density field.
 */
struct SPHResult3D {
    std::vector<double> density;    ///< Flattened (cells^3) density
    std::vector<int64_t> counts;    ///< Particles contributing to each cell
    int cells = 0;                  ///< Cells per dimension
    double cell_width = 0.0;        ///< Physical cell width
    double mean_density = 0.0;      ///< Mean density
    double total_mass = 0.0;        ///< Total mass in volume

    // Performance metrics
    double render_time_ms = 0.0;
    int64_t n_particles = 0;
    int64_t kernel_evals = 0;
};

/**
 * Particle data for SPH rendering.
 *
 * Structure-of-Arrays format for efficient SIMD access.
 */
struct SPHParticles {
    std::vector<float> x, y, z;     ///< Positions
    std::vector<float> mass;        ///< Particle masses (optional, default = uniform)
    std::vector<float> h;           ///< Smoothing lengths (optional, computed if empty)

    /**
     * Number of particles.
     */
    size_t size() const { return x.size(); }

    /**
     * Check if empty.
     */
    bool empty() const { return x.empty(); }

    /**
     * Reserve memory.
     */
    void reserve(size_t n) {
        x.reserve(n);
        y.reserve(n);
        z.reserve(n);
    }

    /**
     * Clear all data.
     */
    void clear() {
        x.clear();
        y.clear();
        z.clear();
        mass.clear();
        h.clear();
    }

    /**
     * Add a particle.
     */
    void add(float px, float py, float pz, float m = 1.0f, float smooth = 0.0f) {
        x.push_back(px);
        y.push_back(py);
        z.push_back(pz);
        if (!mass.empty() || m != 1.0f) {
            if (mass.empty()) mass.resize(x.size() - 1, 1.0f);
            mass.push_back(m);
        }
        if (!h.empty() || smooth > 0.0f) {
            if (h.empty()) h.resize(x.size() - 1, 0.0f);
            h.push_back(smooth);
        }
    }
};

/**
 * Spatial hash grid for efficient neighbor finding.
 *
 * Divides space into cells and maintains lists of particles in each cell.
 * Provides O(1) average lookup for particles within a radius.
 */
class SpatialHashGrid {
public:
    /**
     * Build grid from particles.
     * @param particles Particle positions
     * @param cell_size Size of each grid cell (typically = smoothing length)
     * @param box_size Simulation box size (for periodic wrapping)
     * @param periodic Use periodic boundaries
     */
    SpatialHashGrid(
        const SPHParticles& particles,
        double cell_size,
        double box_size,
        bool periodic = true
    );

    /**
     * Find all particles within radius of a point.
     * @param x Query point x
     * @param y Query point y
     * @param z Query point z
     * @param radius Search radius
     * @param indices Output: particle indices within radius
     */
    void find_neighbors(
        double x, double y, double z,
        double radius,
        std::vector<int32_t>& indices
    ) const;

    /**
     * Find particles in a 2D projected column.
     * @param x Query point x (in projection plane)
     * @param y Query point y (in projection plane)
     * @param radius Search radius in projection plane
     * @param axis Projection axis (0=x, 1=y, 2=z)
     * @param indices Output: particle indices
     */
    void find_neighbors_2d(
        double x, double y,
        double radius,
        int axis,
        std::vector<int32_t>& indices
    ) const;

    /**
     * Get grid statistics.
     */
    int64_t n_cells() const { return static_cast<int64_t>(cells_.size()); }
    int64_t n_particles() const { return n_particles_; }
    double cell_size() const { return cell_size_; }

private:
    std::vector<std::vector<int32_t>> cells_;
    double cell_size_;
    double box_size_;
    double offset_;  // Offset for centered coordinates (box_size/2)
    int grid_dim_;
    bool periodic_;
    int64_t n_particles_;

    const SPHParticles* particles_;

    int64_t cell_index(int ix, int iy, int iz) const;
    void wrap_cell(int& ix, int& iy, int& iz) const;
};

/**
 * SPH kernel evaluator with SIMD optimization.
 */
class SPHKernelEvaluator {
public:
    explicit SPHKernelEvaluator(SPHKernel kernel);

    /**
     * Evaluate kernel W(r, h).
     * @param r Distance
     * @param h Smoothing length
     * @return Kernel value (normalized)
     */
    double evaluate(double r, double h) const;

    /**
     * Evaluate 2D projected kernel (line-of-sight integrated).
     * @param r_perp Perpendicular distance in projection plane
     * @param h Smoothing length
     * @return Surface density kernel value
     */
    double evaluate_2d(double r_perp, double h) const;

    /**
     * Batch evaluate kernel for SIMD (8 distances at once).
     * @param r Array of 8 distances
     * @param h Smoothing length
     * @param out Output array of 8 kernel values
     */
    void evaluate_batch(const float* r, float h, float* out) const;

    /**
     * Get compact support radius multiplier.
     * Kernel is zero for r > support_radius * h.
     */
    double support_radius() const { return support_radius_; }

    /**
     * Get kernel type.
     */
    SPHKernel kernel_type() const { return kernel_; }

private:
    SPHKernel kernel_;
    double support_radius_;
    double norm_1d_;
    double norm_2d_;
    double norm_3d_;

    void compute_normalization();
};

/**
 * Main SPH density renderer.
 */
class SPHRenderer {
public:
    /**
     * Construct renderer with configuration.
     */
    explicit SPHRenderer(const SPHConfig& config);

    /**
     * Render 2D density projection.
     *
     * Projects particle density along one axis using SPH kernel smoothing.
     * This is the primary function for generating halo density maps.
     *
     * @param particles Particle data (positions, optional masses/smoothing)
     * @return 2D density projection result
     */
    SPHResult2D render_2d(const SPHParticles& particles);

    /**
     * Render 2D density from raw arrays (convenience overload).
     *
     * @param x Particle x coordinates
     * @param y Particle y coordinates
     * @param z Particle z coordinates
     * @param n_particles Number of particles
     * @param masses Particle masses (nullptr = uniform mass from config)
     * @return 2D density projection result
     */
    SPHResult2D render_2d(
        const float* x,
        const float* y,
        const float* z,
        int64_t n_particles,
        const float* masses = nullptr
    );

    /**
     * Render 3D density field.
     *
     * @param particles Particle data
     * @return 3D density field result
     */
    SPHResult3D render_3d(const SPHParticles& particles);

    /**
     * Compute adaptive smoothing lengths.
     *
     * Estimates smoothing length for each particle based on local density
     * such that approximately neighbors_target particles are within kernel.
     *
     * @param particles Particle data (h will be filled)
     */
    void compute_smoothing_lengths(SPHParticles& particles);

    /**
     * Get configuration.
     */
    const SPHConfig& config() const { return config_; }

    /**
     * Get kernel evaluator.
     */
    const SPHKernelEvaluator& kernel() const { return kernel_; }

private:
    SPHConfig config_;
    SPHKernelEvaluator kernel_;

    // Internal rendering with SIMD
    void render_2d_cell_batch(
        const SPHParticles& particles,
        const SpatialHashGrid& grid,
        int cell_start,
        int cell_end,
        std::vector<double>& density,
        std::vector<int64_t>& counts,
        int64_t& kernel_evals
    );

    // Coordinate transformation
    void transform_to_render_coords(
        const SPHParticles& input,
        SPHParticles& output,
        double& render_width
    ) const;
};

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Render 2D SPH density projection.
 *
 * High-level function for rendering density around a halo or region.
 *
 * @param positions Particle positions (N x 3), row-major
 * @param n_particles Number of particles
 * @param center Center of rendering region
 * @param box_width Width of rendering region
 * @param output_cells Output grid resolution
 * @param sim_box_size Full simulation box size
 * @param projection_axis Axis to project along (0=x, 1=y, 2=z)
 * @param kernel SPH kernel type
 * @param smoothing_length Fixed smoothing length (0 = adaptive)
 * @param n_threads Number of threads (0 = auto)
 * @return 2D density result
 */
SPHResult2D render_sph_density_2d(
    const float* positions,
    int64_t n_particles,
    const std::array<double, 3>& center,
    double box_width,
    int output_cells,
    double sim_box_size,
    int projection_axis = 2,
    SPHKernel kernel = SPHKernel::CUBIC_SPLINE,
    double smoothing_length = 0.0,
    int n_threads = 0
);

/**
 * Double-precision version.
 */
SPHResult2D render_sph_density_2d(
    const double* positions,
    int64_t n_particles,
    const std::array<double, 3>& center,
    double box_width,
    int output_cells,
    double sim_box_size,
    int projection_axis = 2,
    SPHKernel kernel = SPHKernel::CUBIC_SPLINE,
    double smoothing_length = 0.0,
    int n_threads = 0
);

/**
 * Estimate smoothing length from mean inter-particle spacing.
 *
 * @param n_particles Number of particles
 * @param box_size Box size containing particles
 * @param neighbors_target Target number of neighbors
 * @return Estimated smoothing length
 */
double estimate_smoothing_length(
    int64_t n_particles,
    double box_size,
    int neighbors_target = 32
);

} // namespace density
} // namespace tessera
