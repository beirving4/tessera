/**
 * @file cic_density.cpp
 * @brief Cloud-in-Cell (CIC) density estimation implementation
 *
 * Uses thread-local grid accumulation (same pattern as tetra_density.cpp)
 * to avoid atomics. Each thread deposits into its own grid, then grids
 * are reduced into the final result.
 */

#include "density/cic_density.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tessera {
namespace density {

CICDensityResult compute_cic_density(
    const double* positions,
    int64_t n_particles,
    const CICDensityConfig& config)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    if (n_particles <= 0) {
        throw std::runtime_error("compute_cic_density: n_particles must be > 0");
    }
    if (config.box_size <= 0.0) {
        throw std::runtime_error("compute_cic_density: box_size must be > 0");
    }
    if (config.output_cells <= 0) {
        throw std::runtime_error("compute_cic_density: output_cells must be > 0");
    }

    const int nc = config.output_cells;
    const double L = config.box_size;
    const double cell_width = L / nc;
    const double inv_cell_width = 1.0 / cell_width;
    const double cell_volume = cell_width * cell_width * cell_width;
    const int64_t grid_size = static_cast<int64_t>(nc) * nc * nc;

    // Determine thread count
    int n_threads = config.n_threads;
#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = std::min(omp_get_max_threads(), 8);
    }
#else
    n_threads = 1;
#endif

    CICDensityResult result;
    result.n_particles = n_particles;
    result.output_cells = nc;
    result.cell_width = cell_width;
    result.mean_density = static_cast<double>(n_particles) * config.particle_mass /
                          (L * L * L);

    // === Grid deposition with thread-local buffers ===
    auto t_grid_start = std::chrono::high_resolution_clock::now();

    // Allocate thread-local grids
    std::vector<std::vector<double>> thread_grids(n_threads,
        std::vector<double>(grid_size, 0.0));

    #pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        double* local_grid = thread_grids[tid].data();

        #pragma omp for schedule(static)
        for (int64_t p = 0; p < n_particles; ++p) {
            double px = positions[p * 3 + 0];
            double py = positions[p * 3 + 1];
            double pz = positions[p * 3 + 2];

            // Convert to cell coordinates
            double cx = px * inv_cell_width;
            double cy = py * inv_cell_width;
            double cz = pz * inv_cell_width;

            // Lower-left cell index
            int ix = static_cast<int>(std::floor(cx));
            int iy = static_cast<int>(std::floor(cy));
            int iz = static_cast<int>(std::floor(cz));

            // Fractional offsets within cell [0, 1)
            double dx = cx - ix;
            double dy = cy - iy;
            double dz = cz - iz;

            // Trilinear weights for the 8 neighboring cells
            double wx[2] = {1.0 - dx, dx};
            double wy[2] = {1.0 - dy, dy};
            double wz[2] = {1.0 - dz, dz};

            // Deposit to 8 cells
            for (int di = 0; di < 2; ++di) {
                int ci = ix + di;
                if (config.periodic) {
                    ci = ((ci % nc) + nc) % nc;
                } else if (ci < 0 || ci >= nc) {
                    continue;
                }

                for (int dj = 0; dj < 2; ++dj) {
                    int cj = iy + dj;
                    if (config.periodic) {
                        cj = ((cj % nc) + nc) % nc;
                    } else if (cj < 0 || cj >= nc) {
                        continue;
                    }

                    for (int dk = 0; dk < 2; ++dk) {
                        int ck = iz + dk;
                        if (config.periodic) {
                            ck = ((ck % nc) + nc) % nc;
                        } else if (ck < 0 || ck >= nc) {
                            continue;
                        }

                        double w = wx[di] * wy[dj] * wz[dk] * config.particle_mass;
                        int64_t idx = static_cast<int64_t>(ci) * nc * nc +
                                      static_cast<int64_t>(cj) * nc +
                                      static_cast<int64_t>(ck);
                        local_grid[idx] += w;
                    }
                }
            }
        }
    }

    // Reduce thread-local grids into result
    result.grid_density.resize(grid_size, 0.0);

    // Parallel reduction across threads
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < grid_size; ++i) {
        double sum = 0.0;
        for (int t = 0; t < n_threads; ++t) {
            sum += thread_grids[t][i];
        }
        // Convert mass to density by dividing by cell volume
        result.grid_density[i] = sum / cell_volume;
    }

    // Free thread-local grids
    thread_grids.clear();
    thread_grids.shrink_to_fit();

    auto t_grid_end = std::chrono::high_resolution_clock::now();
    result.grid_time_ms = std::chrono::duration<double, std::milli>(
        t_grid_end - t_grid_start).count();

    // === Optional: Sample density at particle positions ===
    if (config.sample_at_particles) {
        auto t_sample_start = std::chrono::high_resolution_clock::now();

        result.particle_density.resize(n_particles);

        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t p = 0; p < n_particles; ++p) {
            double px = positions[p * 3 + 0];
            double py = positions[p * 3 + 1];
            double pz = positions[p * 3 + 2];

            // Cell-centered coordinates: shift by half a cell so interpolation
            // is centered on cell centers rather than cell corners
            double cx = px * inv_cell_width - 0.5;
            double cy = py * inv_cell_width - 0.5;
            double cz = pz * inv_cell_width - 0.5;

            int ix = static_cast<int>(std::floor(cx));
            int iy = static_cast<int>(std::floor(cy));
            int iz = static_cast<int>(std::floor(cz));

            double dx = cx - ix;
            double dy = cy - iy;
            double dz = cz - iz;

            double wx[2] = {1.0 - dx, dx};
            double wy[2] = {1.0 - dy, dy};
            double wz[2] = {1.0 - dz, dz};

            double density_val = 0.0;
            for (int di = 0; di < 2; ++di) {
                int ci = ix + di;
                if (config.periodic) {
                    ci = ((ci % nc) + nc) % nc;
                } else if (ci < 0 || ci >= nc) {
                    continue;
                }

                for (int dj = 0; dj < 2; ++dj) {
                    int cj = iy + dj;
                    if (config.periodic) {
                        cj = ((cj % nc) + nc) % nc;
                    } else if (cj < 0 || cj >= nc) {
                        continue;
                    }

                    for (int dk = 0; dk < 2; ++dk) {
                        int ck = iz + dk;
                        if (config.periodic) {
                            ck = ((ck % nc) + nc) % nc;
                        } else if (ck < 0 || ck >= nc) {
                            continue;
                        }

                        int64_t idx = static_cast<int64_t>(ci) * nc * nc +
                                      static_cast<int64_t>(cj) * nc +
                                      static_cast<int64_t>(ck);
                        density_val += wx[di] * wy[dj] * wz[dk] *
                                       result.grid_density[idx];
                    }
                }
            }
            result.particle_density[p] = density_val;
        }

        auto t_sample_end = std::chrono::high_resolution_clock::now();
        result.sample_time_ms = std::chrono::duration<double, std::milli>(
            t_sample_end - t_sample_start).count();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    return result;
}

} // namespace density
} // namespace tessera
