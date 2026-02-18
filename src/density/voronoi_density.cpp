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
#include <random>

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

    // Determine thread count.  Per-thread voro_compute objects use ~108 KB
    // each (mask + queue), so even 64 threads costs only ~7 MB.
    int n_threads = config.n_threads;
#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
#else
    n_threads = 1;
#endif
    (void)n_threads;  // Suppress unused warning when OpenMP disabled

    // Container grid: ~5 particles per cell is optimal for voro++
    int n_grid = std::max(1, static_cast<int>(std::round(std::cbrt(
        static_cast<double>(n_particles) / 5.0))));
    int init_mem = 8;  // Initial memory per cell

    // Create periodic container and add particles.
    // Use put(particle_order&, ...) to bypass voro++'s hard-coded duplicate
    // detection (squared-distance < 1e-10), which would call exit(1) on the
    // co-located particles that naturally arise from shell-crossing in N-body
    // simulations.  A tiny random perturbation (~1e-8 * mean_spacing) is still
    // applied to break exact geometric degeneracies in the Voronoi construction.
    voro::container_periodic con(L, 0.0, L, 0.0, 0.0, L,
                                  n_grid, n_grid, n_grid, init_mem);

    const double mean_spacing = L / std::cbrt(static_cast<double>(n_particles));
    const double eps = 1e-8 * mean_spacing;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> jitter(-eps, eps);

    voro::particle_order po(static_cast<int>(n_particles));
    for (int64_t i = 0; i < n_particles; ++i) {
        con.put(po, static_cast<int>(i),
                positions[3 * i]     + jitter(rng),
                positions[3 * i + 1] + jitter(rng),
                positions[3 * i + 2] + jitter(rng));
    }

    // Pre-build all periodic images so that create_periodic_image() becomes a
    // no-op during parallel compute_cell() calls (the img[] flags are already set).
    con.create_all_images();

    // Compute Voronoi volumes
    VoronoiDensityResult result;
    result.n_particles = n_particles;
    result.mean_volume = mean_vol;
    result.mean_density = mean_density;

    std::vector<double> volumes(n_particles, 0.0);

    // Collect (ijk, q, id, ci, cj, ck) tuples from the container's loop
    // iterator, then compute cells in parallel via per-thread voro_compute objects.
    struct CellLoc { int ijk; int q; int id; int ci, cj, ck; };
    std::vector<CellLoc> locs;
    locs.reserve(n_particles);

    voro::c_loop_all_periodic cl(con);
    if (cl.start()) do {
        locs.push_back({cl.ijk, cl.q, cl.pid(), cl.i, cl.j, cl.k});
    } while (cl.inc());

    int64_t n_locs = static_cast<int64_t>(locs.size());

    // Dimensions for per-thread voro_compute objects (matches container_prd.cc:76).
    int hx = 2 * con.nx + 1;
    int hy = 2 * con.ey + 1;
    int hz = 2 * con.ez + 1;

    // Parallel Voronoi cell computation.
    // Each thread gets its own voro_compute (owns private mask[]/qu[] buffers)
    // to avoid data races on the container's internal voro_compute member.
    #pragma omp parallel num_threads(n_threads)
    {
        voro::voro_compute<voro::container_periodic> thread_vc(con, hx, hy, hz);
        voro::voronoicell c;
        #pragma omp for schedule(dynamic, 64)
        for (int64_t idx = 0; idx < n_locs; ++idx) {
            const auto& loc = locs[idx];
            if (thread_vc.compute_cell(c, loc.ijk, loc.q, loc.ci, loc.cj, loc.ck)) {
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
