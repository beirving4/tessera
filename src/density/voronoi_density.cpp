/**
 * @file voronoi_density.cpp
 * @brief Voronoi Tessellation Field Estimator (VTFE) density computation
 *
 * Uses voro++ with periodic containers. Cell computation is parallelized
 * over the container's internal block structure.
 */

#ifdef TESSERA_HAS_VORO

#include "density/voronoi_density.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "voro++.hh"

namespace tessera {
namespace density {

VoronoiDensityResult compute_voronoi_density(
    const double* positions,
    int64_t n_particles,
    const VoronoiDensityConfig& config)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    if (n_particles <= 0) {
        throw std::runtime_error("compute_voronoi_density: n_particles must be > 0");
    }
    if (config.box_size <= 0.0) {
        throw std::runtime_error("compute_voronoi_density: box_size must be > 0");
    }

    const double L = config.box_size;
    const double box_vol = L * L * L;
    const double mean_vol = box_vol / static_cast<double>(n_particles);
    const double mean_density = static_cast<double>(n_particles) / box_vol;

    // Determine thread count
    int n_threads = config.n_threads;
#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = std::min(omp_get_max_threads(), 8);
    }
#else
    n_threads = 1;
#endif
    (void)n_threads;  // Suppress unused warning when OpenMP disabled

    // Container grid: ~5 particles per cell is optimal for voro++
    int n_grid = std::max(1, static_cast<int>(std::round(std::cbrt(
        static_cast<double>(n_particles) / 5.0))));
    int init_mem = 8;  // Initial memory per cell

    // Create periodic container and add particles
    voro::container_periodic con(L, 0.0, L, 0.0, 0.0, L,
                                  n_grid, n_grid, n_grid, init_mem);

    for (int64_t i = 0; i < n_particles; ++i) {
        con.put(static_cast<int>(i),
                positions[3 * i],
                positions[3 * i + 1],
                positions[3 * i + 2]);
    }

    // Compute Voronoi volumes
    VoronoiDensityResult result;
    result.n_particles = n_particles;
    result.mean_volume = mean_vol;
    result.mean_density = mean_density;

    std::vector<double> volumes(n_particles, 0.0);

    // Collect (ijk, q, id) tuples from the container's loop iterator,
    // then compute cells in parallel using compute_cell(c, ijk, q).
    struct CellLoc { int ijk; int q; int id; };
    std::vector<CellLoc> locs;
    locs.reserve(n_particles);

    voro::c_loop_all_periodic cl(con);
    if (cl.start()) do {
        locs.push_back({cl.ijk, cl.q, cl.pid()});
    } while (cl.inc());

    int64_t n_locs = static_cast<int64_t>(locs.size());

    // Parallel Voronoi cell computation.
    // compute_cell(c, ijk, q) is safe to call from multiple threads as long as
    // each thread uses its own voronoicell object and the container is read-only.
    #pragma omp parallel num_threads(n_threads)
    {
        voro::voronoicell c;
        #pragma omp for schedule(dynamic, 64)
        for (int64_t idx = 0; idx < n_locs; ++idx) {
            const auto& loc = locs[idx];
            if (con.compute_cell(c, loc.ijk, loc.q)) {
                volumes[loc.id] = c.volume();
            }
        }
    }

    // Compute density: density_i = mean_density * mean_vol / volumes_i
    result.density.resize(n_particles);
    for (int64_t i = 0; i < n_particles; ++i) {
        if (volumes[i] > 0.0) {
            result.density[i] = mean_density * mean_vol / volumes[i];
        } else {
            result.density[i] = 0.0;
        }
    }

    if (config.compute_volumes) {
        result.volumes = std::move(volumes);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

} // namespace density
} // namespace tessera

#endif // TESSERA_HAS_VORO
