#pragma once

/**
 * @file tetra_density.h
 * @brief High-performance tetrahedron-based density field computation
 * 
 * This module provides the core algorithm from gotetra for computing density
 * fields using phase-space tessellation. The key insight is that particles
 * in an N-body simulation start on a regular Lagrangian grid, and this
 * connectivity can be exploited to form tetrahedra that tile the simulation
 * volume.
 * 
 * Algorithm overview:
 * 1. Each Lagrangian cell (cube) is decomposed into 6 tetrahedra
 * 2. Monte Carlo samples are distributed uniformly within each tetrahedron
 * 3. Samples are deposited onto the output density grid
 * 4. The density naturally emerges: compressed tetrahedra deposit to fewer
 *    cells (high density), expanded tetrahedra spread across more cells
 *    (low density)
 */

#include <vector>
#include <array>
#include <cstdint>
#include <memory>

namespace tessera {
namespace density {

/**
 * Configuration for density field computation.
 */
struct TetraDensityConfig {
    int lagrangian_grid_size;   ///< Particles per dimension (N for N^3)
    int output_cells;           ///< Output grid cells per dimension
    double box_size;            ///< Physical simulation box size
    double particle_mass;       ///< Mass per particle
    int n_samples;              ///< Monte Carlo samples per tetrahedron
    int n_threads;              ///< Number of threads (0 = auto)
    bool periodic;              ///< Use periodic boundary conditions
    uint64_t seed;              ///< Random seed for reproducibility (0 = time-based)
    
    // Sub-box rendering (for high-resolution local density fields)
    bool subbox_enabled = false;        ///< If true, render only within sub-box
    std::array<double, 3> subbox_origin = {0, 0, 0};  ///< Sub-box origin (x, y, z)
    std::array<double, 3> subbox_width = {0, 0, 0};   ///< Sub-box dimensions (width in each dim)

    // gotetra compatibility mode
    bool gotetra_compatible = false;    ///< If true, use N+1 output cells like gotetra
                                        ///< This can improve quality at box boundaries
};

/**
 * Result of 3D density computation.
 */
struct TetraDensityResult3D {
    std::vector<double> density;    ///< Flattened (cells^3) density array
    int cells;                       ///< Cells per dimension
    double cell_width;               ///< Physical cell width
    double total_mass;               ///< Total mass deposited
    double mean_density;             ///< Mean density
    int64_t n_tetrahedra;           ///< Number of tetrahedra processed
};

/**
 * Result of 2D density computation (projection or slice).
 */
struct TetraDensityResult2D {
    std::vector<double> density;    ///< Flattened (cells^2) surface density
    int cells;                       ///< Cells per dimension
    double cell_width;               ///< Physical cell width
    int projection_axis;             ///< 0=x, 1=y, 2=z
    double slice_min;                ///< Slice start (if applicable)
    double slice_max;                ///< Slice end (if applicable)
    double mean_surface_density;     ///< Mean surface density
};

/**
 * Pre-generated unit tetrahedron samples for reuse across multiple
 * density computations. This avoids regenerating random samples each time.
 */
class UnitTetraSamples {
public:
    /**
     * Generate sample buffers.
     * 
     * @param n_buffers Number of independent sample buffers
     * @param n_samples Number of samples per buffer
     * @param seed Random seed (0 = time-based)
     */
    UnitTetraSamples(int n_buffers, int n_samples, uint64_t seed = 0);
    
    /// Get a sample buffer (barycentric coordinates: s, t, u where v=1-s-t-u)
    const std::vector<std::array<float, 4>>& get_buffer(int idx) const {
        return buffers_[idx % buffers_.size()];
    }
    
    int n_buffers() const { return static_cast<int>(buffers_.size()); }
    int n_samples() const { return n_samples_; }

private:
    std::vector<std::vector<std::array<float, 4>>> buffers_;
    int n_samples_;
};

/**
 * Generate tetrahedron corner indices for a Lagrangian grid.
 * 
 * Each Lagrangian cell is decomposed into 6 tetrahedra following the
 * gotetra convention. The indices reference into a flattened particle
 * array where particle i corresponds to Lagrangian position:
 *   ix = i % grid_size
 *   iy = (i / grid_size) % grid_size
 *   iz = i / (grid_size^2)
 * 
 * @param grid_size Particles per dimension
 * @param periodic If true, include boundary-wrapping tetrahedra
 * @return Vector of (n_tetra, 4) corner indices
 */
std::vector<std::array<int64_t, 4>> generate_tetra_indices(
    int grid_size, bool periodic = true);

/**
 * Compute a 3D density field using tetrahedron tessellation.
 * 
 * This is the main gotetra algorithm with OpenMP parallelization.
 * 
 * @param positions Particle positions, MUST be in Lagrangian order.
 *                  Shape: (grid_size^3, 3), row-major.
 * @param config Computation configuration
 * @param samples Optional pre-generated samples (if nullptr, generates new ones)
 * @return TetraDensityResult3D containing the density grid
 */
TetraDensityResult3D compute_tetra_density_3d(
    const double* positions,
    const TetraDensityConfig& config,
    const UnitTetraSamples* samples = nullptr
);

/**
 * Compute a 3D density field (float input version).
 */
TetraDensityResult3D compute_tetra_density_3d(
    const float* positions,
    const TetraDensityConfig& config,
    const UnitTetraSamples* samples = nullptr
);

/**
 * Compute a 2D projected density field (column density).
 * 
 * Projects all mass along one axis to create a 2D density map.
 * 
 * @param positions Particle positions in Lagrangian order
 * @param config Computation configuration
 * @param projection_axis Axis to project along (0=x, 1=y, 2=z)
 * @param samples Optional pre-generated samples
 * @return TetraDensityResult2D containing the surface density
 */
TetraDensityResult2D compute_tetra_density_2d_projection(
    const double* positions,
    const TetraDensityConfig& config,
    int projection_axis,
    const UnitTetraSamples* samples = nullptr
);

/**
 * Compute a 2D slice density field.
 * 
 * Extracts and projects a thin slice of the volume.
 * 
 * @param positions Particle positions in Lagrangian order
 * @param config Computation configuration
 * @param projection_axis Axis perpendicular to slice (0=x, 1=y, 2=z)
 * @param slice_min Start of slice along projection axis
 * @param slice_max End of slice along projection axis
 * @param samples Optional pre-generated samples
 * @return TetraDensityResult2D containing the slice surface density
 */
TetraDensityResult2D compute_tetra_density_2d_slice(
    const double* positions,
    const TetraDensityConfig& config,
    int projection_axis,
    double slice_min,
    double slice_max,
    const UnitTetraSamples* samples = nullptr
);

/**
 * Sort particles from arbitrary order to Lagrangian order.
 * 
 * In GADGET-style simulations, particle IDs encode the original Lagrangian
 * grid position. This function reorders particles so that particle i is at
 * Lagrangian position (i % N, (i/N) % N, i/N^2).
 * 
 * @param positions Input positions (n_particles, 3)
 * @param particle_ids Particle IDs (n_particles,)
 * @param n_particles Total number of particles
 * @param grid_size Lagrangian grid size (N for N^3 particles)
 * @param sorted_positions Output array, must be pre-allocated (n_particles, 3)
 * @param id_offset Offset to subtract from IDs (typically 1 for 1-indexed)
 */
void sort_by_lagrangian_id(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double* sorted_positions,
    int64_t id_offset = 1
);

/**
 * Sort particles (float version).
 */
void sort_by_lagrangian_id(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    float* sorted_positions,
    int64_t id_offset = 1
);

} // namespace density
} // namespace tessera
