/**
 * @file cic_density.h
 * @brief Cloud-in-Cell (CIC) density estimation on a regular grid
 *
 * Computes density by depositing particle mass onto a regular grid using
 * trilinear (CIC) interpolation weights, then optionally samples the grid
 * back at particle positions for per-particle density values.
 *
 * The CIC kernel assigns each particle's mass to the 8 nearest grid cells
 * with weights proportional to the overlap volume, making it the standard
 * Eulerian density estimator for cosmological simulations.
 *
 * Reference: Klypin et al. 2018, MNRAS 478, 4602
 *            "Density Distribution of the Cosmological Matter Field"
 */

#pragma once

#include <cstdint>
#include <vector>

namespace tessera {
namespace density {

/**
 * Configuration for CIC density computation.
 */
struct CICDensityConfig {
    double box_size = 0.0;          ///< Physical box size (periodic cube)
    int output_cells = 256;         ///< Grid resolution (cells per dimension)
    double particle_mass = 1.0;     ///< Mass per particle
    int n_threads = 0;              ///< Number of OpenMP threads (0 = auto, capped at 8)
    bool periodic = true;           ///< Use periodic boundary conditions
    bool sample_at_particles = true; ///< Sample grid density back at particle positions
};

/**
 * Result of CIC density computation.
 */
struct CICDensityResult {
    std::vector<double> grid_density;      ///< 3D density grid (output_cells^3), row-major [z][y][x]
    std::vector<double> particle_density;  ///< Per-particle density from grid interpolation (N elements)
    double mean_density = 0.0;             ///< Mean grid density = total_mass / box_volume
    double cell_width = 0.0;               ///< Physical cell width = box_size / output_cells
    int output_cells = 0;                  ///< Grid resolution used
    double total_time_ms = 0.0;            ///< Total computation wall time in milliseconds
    double grid_time_ms = 0.0;             ///< Grid deposition time in milliseconds
    double sample_time_ms = 0.0;           ///< Particle sampling time in milliseconds
    int64_t n_particles = 0;               ///< Number of particles processed
};

/**
 * Compute CIC density on a regular grid.
 *
 * Algorithm:
 * 1. For each particle, compute cell coordinates and fractional offsets
 * 2. Deposit mass to 8 neighboring cells with trilinear weights
 * 3. Divide by cell volume to get density
 * 4. Optionally sample grid density at particle positions via trilinear interpolation
 *
 * Uses thread-local grids for OpenMP parallelism (no atomics).
 *
 * @param positions Particle positions, (N, 3) C-contiguous float64
 * @param n_particles Number of particles
 * @param config Configuration parameters
 * @return CICDensityResult with grid density and optional per-particle density
 */
CICDensityResult compute_cic_density(
    const double* positions,
    int64_t n_particles,
    const CICDensityConfig& config);

} // namespace density
} // namespace tessera
