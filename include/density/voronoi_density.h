/**
 * @file voronoi_density.h
 * @brief Voronoi Tessellation Field Estimator (VTFE) density computation
 *
 * Computes per-particle density using Voronoi cell volumes in Eulerian space:
 *   1 + delta_i = V_mean / V_i
 *
 * This resolves densities up to 10^4-10^6 for N=256^3, matching the ORIGAMI
 * paper (Falck et al.) results. Uses voro++ with periodic boundaries and
 * OpenMP parallelism for performance at scale (N=1024^3).
 *
 * Reference: Falck, Neyrinck, & Szalay 2012, ApJ 754, 126
 */

#pragma once

#include <cstdint>
#include <vector>

#ifdef TESSERA_HAS_VORO

namespace tessera {
namespace density {

/**
 * Configuration for Voronoi density computation.
 */
struct VoronoiDensityConfig {
    double box_size = 0.0;        ///< Physical box size (periodic cube)
    int n_threads = 0;            ///< Number of OpenMP threads (0 = auto)
    bool compute_volumes = true;  ///< Return raw Voronoi volumes in result
};

/**
 * Result of Voronoi density computation.
 */
struct VoronoiDensityResult {
    std::vector<double> density;  ///< Per-particle density: mean_density * V_mean / V_i (N elements)
    std::vector<double> volumes;  ///< Raw Voronoi cell volumes V_i (N elements, if compute_volumes)
    double mean_density = 0.0;    ///< Total mass / box volume (assuming unit particle mass)
    double mean_volume = 0.0;     ///< V_mean = box_volume / N
    double total_time_ms = 0.0;   ///< Computation wall time in milliseconds
    int64_t n_particles = 0;      ///< Number of particles processed
};

/**
 * Compute per-particle density using Voronoi cell volumes.
 *
 * Algorithm:
 * 1. Build periodic Voronoi tessellation using voro++
 * 2. Compute cell volume V_i for each particle (OpenMP parallel)
 * 3. density_i = mean_density * V_mean / V_i
 *
 * where V_mean = box_size^3 / N and mean_density = N / box_size^3
 * (assuming unit particle mass).
 *
 * @param positions Particle positions, (N, 3) C-contiguous float64
 * @param n_particles Number of particles
 * @param config Configuration parameters
 * @return VoronoiDensityResult with per-particle densities and volumes
 */
VoronoiDensityResult compute_voronoi_density(
    const double* positions,
    int64_t n_particles,
    const VoronoiDensityConfig& config);

} // namespace density
} // namespace tessera

#endif // TESSERA_HAS_VORO
