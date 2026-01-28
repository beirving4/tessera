/**
 * @file origami.cpp
 * @brief ORIGAMI morphology classification implementation
 *
 * This is a C++ port of the original origamitag.c from Falck, Neyrinck, & Szalay.
 * The algorithm is preserved exactly, including the diagonal checks for robustness.
 */

#include "origami/origami.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tessera {
namespace origami {

namespace {

/**
 * Check if parallel execution should be used.
 * Returns true only if OpenMP is available AND n_threads != 1.
 * This allows:
 * - Cluster (n_threads=0 or >1): parallel execution
 * - macOS workaround (n_threads=1): serial execution
 * - Non-OpenMP builds: serial execution
 */
inline bool use_parallel([[maybe_unused]] int n_threads) {
#ifdef _OPENMP
    return n_threads != 1;
#else
    return false;
#endif
}

// Helper: safe modulo that handles negative values (periodic boundaries)
inline int goodmod(int a, int b) {
    if (a >= b) return a - b;
    if (a < 0) return a + b;
    return a;
}

// Helper: return 1 if h < 0, else 0 (used for alternating search direction)
inline int isneg(int h) {
    return (h < 0) ? 1 : 0;
}

// Helper: convert (x, y, z) grid indices to linear index
inline int64_t par(int x, int y, int z, int ng) {
    return static_cast<int64_t>(x) + (static_cast<int64_t>(y) + static_cast<int64_t>(z) * ng) * ng;
}

/**
 * Core ORIGAMI algorithm - templated for float/double positions.
 *
 * Ported exactly from origamitag.c, including:
 * - Cartesian axis checks (x, y, z)
 * - Diagonal checks (6 directions for robustness)
 *
 * Optimized with bit-packed crossing flags:
 * - Bit 0: x crossed, Bit 1: y crossed, Bit 2: z crossed
 * - Bit 3-4: m0 diagonals, Bit 5-6: m1 diagonals, Bit 7-8: m2 diagonals
 */
template<typename T>
OrigamiResult compute_morphology_impl(
    const T* positions,
    const OrigamiConfig& config)
{
    const int np1d = config.lagrangian_grid_size;
    const double boxsize = config.box_size;
    const int nsplit = config.n_split;

    if (np1d <= 0) {
        throw std::runtime_error("lagrangian_grid_size must be positive");
    }
    if (boxsize <= 0) {
        throw std::runtime_error("box_size must be positive");
    }

    const int64_t np = static_cast<int64_t>(np1d) * np1d * np1d;
    const int ng4 = np1d / 4;  // Search range limit

    // Use native type T for arithmetic (avoids float->double->float conversions)
    const T box_t = static_cast<T>(boxsize);
    const T b2 = box_t / static_cast<T>(2);
    const T negb2 = -b2;

    // Bit flags for crossing detection (replaces 4 separate uint8_t arrays)
    // Total: 9 bits packed into uint16_t (saves 50% memory: 2N vs 4N bytes)
    constexpr uint16_t BIT_X = 1 << 0;      // x-axis crossed
    constexpr uint16_t BIT_Y = 1 << 1;      // y-axis crossed
    constexpr uint16_t BIT_Z = 1 << 2;      // z-axis crossed
    constexpr uint16_t BIT_M0_D1 = 1 << 3;  // m0 diagonal 1 (y+h, z+h)
    constexpr uint16_t BIT_M0_D2 = 1 << 4;  // m0 diagonal 2 (y+h, z-h)
    constexpr uint16_t BIT_M1_D1 = 1 << 5;  // m1 diagonal 1 (x+h, z+h)
    constexpr uint16_t BIT_M1_D2 = 1 << 6;  // m1 diagonal 2 (x+h, z-h)
    constexpr uint16_t BIT_M2_D1 = 1 << 7;  // m2 diagonal 1 (x+h, y+h)
    constexpr uint16_t BIT_M2_D2 = 1 << 8;  // m2 diagonal 2 (x+h, y-h)

    // Single bit-packed array (0 = no crossings detected yet)
    std::vector<uint16_t> flags(np, 0);

    // Set number of threads
#ifdef _OPENMP
    if (config.n_threads > 0) {
        omp_set_num_threads(config.n_threads);
    }
#endif

    // Main loop: parallelized over spatial subdomains
    // Dynamic scheduling helps with load imbalance when early-termination varies
    const bool skip_diagonals = config.cartesian_only;
    #pragma omp parallel for default(none) schedule(dynamic, 1) \
        shared(positions, flags) \
        firstprivate(np1d, ng4, box_t, negb2, b2, nsplit, skip_diagonals)
    for (int s = 0; s < nsplit * nsplit * nsplit; ++s) {
        // Compute subdomain bounds
        int zstart = s / (nsplit * nsplit);
        int zend = zstart + 1;
        int ystart = (s - zstart * nsplit * nsplit) / nsplit;
        int yend = ystart + 1;
        int xstart = (s - zstart * nsplit * nsplit - ystart * nsplit);
        int xend = xstart + 1;

        // Scale to particle indices
        zstart *= (np1d / nsplit);
        ystart *= (np1d / nsplit);
        xstart *= (np1d / nsplit);
        zend *= (np1d / nsplit);
        yend *= (np1d / nsplit);
        xend *= (np1d / nsplit);

        // Process all particles in this subdomain
        for (int x = xstart; x < xend; ++x) {
            for (int y = ystart; y < yend; ++y) {
                for (int z = zstart; z < zend; ++z) {
                    int64_t i = par(x, y, z, np1d);

                    // Prefetch next particle's position data (z+1 or y+1 or x+1)
                    if (z + 1 < zend) {
                        int64_t i_next = par(x, y, z + 1, np1d);
                        __builtin_prefetch(&positions[i_next * 3], 0, 1);
                    }

                    // Cache current particle's position
                    const T px = positions[i * 3 + 0];
                    const T py = positions[i * 3 + 1];
                    const T pz = positions[i * 3 + 2];

                    // ========================================
                    // Cartesian axis checks (optimized loop pattern)
                    // ========================================

                    // X-direction: check positive then negative offsets
                    // Using alternating pattern for early termination
                    for (int h = 1; h < ng4; ++h) {
                        // Check positive offset
                        int64_t i2p = par(goodmod(x + h, np1d), y, z, np1d);
                        T dxp = positions[i2p * 3 + 0] - px;
                        if (dxp < negb2) dxp += box_t;
                        if (dxp > b2) dxp -= box_t;
                        if (dxp < static_cast<T>(0)) {
                            flags[i] |= BIT_X;
                            break;
                        }
                        // Check negative offset
                        int64_t i2n = par(goodmod(x - h, np1d), y, z, np1d);
                        T dxn = positions[i2n * 3 + 0] - px;
                        if (dxn < negb2) dxn += box_t;
                        if (dxn > b2) dxn -= box_t;
                        if (dxn > static_cast<T>(0)) {
                            flags[i] |= BIT_X;
                            break;
                        }
                    }

                    // Y-direction
                    for (int h = 1; h < ng4; ++h) {
                        int64_t i2p = par(x, goodmod(y + h, np1d), z, np1d);
                        T dxp = positions[i2p * 3 + 1] - py;
                        if (dxp < negb2) dxp += box_t;
                        if (dxp > b2) dxp -= box_t;
                        if (dxp < static_cast<T>(0)) {
                            flags[i] |= BIT_Y;
                            break;
                        }
                        int64_t i2n = par(x, goodmod(y - h, np1d), z, np1d);
                        T dxn = positions[i2n * 3 + 1] - py;
                        if (dxn < negb2) dxn += box_t;
                        if (dxn > b2) dxn -= box_t;
                        if (dxn > static_cast<T>(0)) {
                            flags[i] |= BIT_Y;
                            break;
                        }
                    }

                    // Z-direction
                    for (int h = 1; h < ng4; ++h) {
                        int64_t i2p = par(x, y, goodmod(z + h, np1d), np1d);
                        T dxp = positions[i2p * 3 + 2] - pz;
                        if (dxp < negb2) dxp += box_t;
                        if (dxp > b2) dxp -= box_t;
                        if (dxp < static_cast<T>(0)) {
                            flags[i] |= BIT_Z;
                            break;
                        }
                        int64_t i2n = par(x, y, goodmod(z - h, np1d), np1d);
                        T dxn = positions[i2n * 3 + 2] - pz;
                        if (dxn < negb2) dxn += box_t;
                        if (dxn > b2) dxn -= box_t;
                        if (dxn > static_cast<T>(0)) {
                            flags[i] |= BIT_Z;
                            break;
                        }
                    }

                    // ========================================
                    // Diagonal checks (for robustness)
                    // Skip if cartesian_only mode is enabled
                    // ========================================

                    if (!skip_diagonals) {

                    // Diagonals in YZ plane (x constant)
                    // Direction: (y+h, z+h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(x, goodmod(y + h, np1d), goodmod(z + h, np1d), np1d);
                        T d1 = positions[i2 * 3 + 1] - py;
                        T d2 = positions[i2 * 3 + 2] - pz;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 + d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M0_D1;
                            break;
                        }
                    }

                    // Direction: (y+h, z-h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(x, goodmod(y + h, np1d), goodmod(z - h, np1d), np1d);
                        T d1 = positions[i2 * 3 + 1] - py;
                        T d2 = positions[i2 * 3 + 2] - pz;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 - d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M0_D2;
                            break;
                        }
                    }

                    // Diagonals in XZ plane (y constant)
                    // Direction: (x+h, z+h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(goodmod(x + h, np1d), y, goodmod(z + h, np1d), np1d);
                        T d1 = positions[i2 * 3 + 0] - px;
                        T d2 = positions[i2 * 3 + 2] - pz;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 + d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M1_D1;
                            break;
                        }
                    }

                    // Direction: (x+h, z-h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(goodmod(x + h, np1d), y, goodmod(z - h, np1d), np1d);
                        T d1 = positions[i2 * 3 + 0] - px;
                        T d2 = positions[i2 * 3 + 2] - pz;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 - d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M1_D2;
                            break;
                        }
                    }

                    // Diagonals in XY plane (z constant)
                    // Direction: (x+h, y+h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(goodmod(x + h, np1d), goodmod(y + h, np1d), z, np1d);
                        T d1 = positions[i2 * 3 + 0] - px;
                        T d2 = positions[i2 * 3 + 1] - py;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 + d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M2_D1;
                            break;
                        }
                    }

                    // Direction: (x+h, y-h)
                    for (int h = 1; h < ng4; h = -h + isneg(h)) {
                        int64_t i2 = par(goodmod(x + h, np1d), goodmod(y - h, np1d), z, np1d);
                        T d1 = positions[i2 * 3 + 0] - px;
                        T d2 = positions[i2 * 3 + 1] - py;
                        if (d1 < negb2) d1 += box_t;
                        if (d1 > b2) d1 -= box_t;
                        if (d2 < negb2) d2 += box_t;
                        if (d2 > b2) d2 -= box_t;
                        if ((d1 - d2) * static_cast<T>(h) < static_cast<T>(0)) {
                            flags[i] |= BIT_M2_D2;
                            break;
                        }
                    }

                    } // end if (!config.cartesian_only)
                }
            }
        }
    }

    // ========================================
    // Compute final morphology and statistics
    // ========================================

    OrigamiResult result;
    result.morphology.resize(np);
    result.n_void = 0;
    result.n_wall = 0;
    result.n_filament = 0;
    result.n_halo = 0;

    // Helper lambda to count bits (crossings) from flags
    const bool cartesian_only = config.cartesian_only;
    auto count_crossings = [cartesian_only](uint16_t f) -> uint8_t {
        // Count Cartesian crossings: bits 0, 1, 2
        uint8_t mn = ((f & BIT_X) != 0) + ((f & BIT_Y) != 0) + ((f & BIT_Z) != 0);
        if (cartesian_only) return mn;
        // m0n: x-cross + m0 diagonals (bits 0, 3, 4)
        uint8_t m0n = ((f & BIT_X) != 0) + ((f & BIT_M0_D1) != 0) + ((f & BIT_M0_D2) != 0);
        // m1n: y-cross + m1 diagonals (bits 1, 5, 6)
        uint8_t m1n = ((f & BIT_Y) != 0) + ((f & BIT_M1_D1) != 0) + ((f & BIT_M1_D2) != 0);
        // m2n: z-cross + m2 diagonals (bits 2, 7, 8)
        uint8_t m2n = ((f & BIT_Z) != 0) + ((f & BIT_M2_D1) != 0) + ((f & BIT_M2_D2) != 0);
        return std::max({mn, m0n, m1n, m2n});
    };

    if (use_parallel(config.n_threads)) {
        // Parallel version with reduction
        int64_t count_void = 0, count_wall = 0, count_filament = 0, count_halo = 0;

        #pragma omp parallel for reduction(+:count_void, count_wall, count_filament, count_halo)
        for (int64_t i = 0; i < np; ++i) {
            uint8_t final_morph = count_crossings(flags[i]);
            result.morphology[i] = final_morph;

            // Count particles per class
            switch (final_morph) {
                case 0: ++count_void; break;
                case 1: ++count_wall; break;
                case 2: ++count_filament; break;
                case 3: ++count_halo; break;
            }
        }

        result.n_void = count_void;
        result.n_wall = count_wall;
        result.n_filament = count_filament;
        result.n_halo = count_halo;
    } else {
        // Serial version
        for (int64_t i = 0; i < np; ++i) {
            uint8_t final_morph = count_crossings(flags[i]);
            result.morphology[i] = final_morph;

            // Count particles per class
            switch (final_morph) {
                case 0: ++result.n_void; break;
                case 1: ++result.n_wall; break;
                case 2: ++result.n_filament; break;
                case 3: ++result.n_halo; break;
            }
        }
    }

    // Compute mass fractions
    double inv_np = 1.0 / static_cast<double>(np);
    result.f_void = static_cast<double>(result.n_void) * inv_np;
    result.f_wall = static_cast<double>(result.n_wall) * inv_np;
    result.f_filament = static_cast<double>(result.n_filament) * inv_np;
    result.f_halo = static_cast<double>(result.n_halo) * inv_np;

    // Detect linear regime
    // In the linear regime, no shell-crossing has occurred, so ORIGAMI
    // classifies (nearly) all particles as voids. The classification is
    // not physically meaningful in this case.
    result.linear_regime_threshold = config.linear_regime_threshold;
    result.is_linear_regime = (result.f_void >= config.linear_regime_threshold);

    return result;
}

} // anonymous namespace

// Public API implementations

OrigamiResult compute_morphology(const double* positions, const OrigamiConfig& config) {
    return compute_morphology_impl(positions, config);
}

OrigamiResult compute_morphology(const float* positions, const OrigamiConfig& config) {
    return compute_morphology_impl(positions, config);
}

void sample_density_at_particles(
    const double* density_grid,
    int grid_cells,
    const double* positions,
    int64_t n_particles,
    double box_size,
    double* particle_density)
{
    const double cell_width = box_size / grid_cells;
    const double inv_cell_width = 1.0 / cell_width;

    #pragma omp parallel for
    for (int64_t i = 0; i < n_particles; ++i) {
        // Get particle position
        double px = positions[i * 3 + 0];
        double py = positions[i * 3 + 1];
        double pz = positions[i * 3 + 2];

        // Wrap to box
        while (px < 0) px += box_size;
        while (px >= box_size) px -= box_size;
        while (py < 0) py += box_size;
        while (py >= box_size) py -= box_size;
        while (pz < 0) pz += box_size;
        while (pz >= box_size) pz -= box_size;

        // Get cell indices and fractional positions
        double fx = px * inv_cell_width;
        double fy = py * inv_cell_width;
        double fz = pz * inv_cell_width;

        int ix0 = static_cast<int>(fx);
        int iy0 = static_cast<int>(fy);
        int iz0 = static_cast<int>(fz);

        // Fractional position within cell
        double dx = fx - ix0;
        double dy = fy - iy0;
        double dz = fz - iz0;

        // Neighbor indices (with periodic wrapping)
        int ix1 = (ix0 + 1) % grid_cells;
        int iy1 = (iy0 + 1) % grid_cells;
        int iz1 = (iz0 + 1) % grid_cells;

        // Clamp base indices
        ix0 = ix0 % grid_cells;
        iy0 = iy0 % grid_cells;
        iz0 = iz0 % grid_cells;

        // Trilinear interpolation
        // Grid is [z, y, x] ordered
        auto idx = [grid_cells](int z, int y, int x) {
            return static_cast<int64_t>(z) * grid_cells * grid_cells +
                   static_cast<int64_t>(y) * grid_cells +
                   static_cast<int64_t>(x);
        };

        double c000 = density_grid[idx(iz0, iy0, ix0)];
        double c001 = density_grid[idx(iz0, iy0, ix1)];
        double c010 = density_grid[idx(iz0, iy1, ix0)];
        double c011 = density_grid[idx(iz0, iy1, ix1)];
        double c100 = density_grid[idx(iz1, iy0, ix0)];
        double c101 = density_grid[idx(iz1, iy0, ix1)];
        double c110 = density_grid[idx(iz1, iy1, ix0)];
        double c111 = density_grid[idx(iz1, iy1, ix1)];

        double c00 = c000 * (1 - dx) + c001 * dx;
        double c01 = c010 * (1 - dx) + c011 * dx;
        double c10 = c100 * (1 - dx) + c101 * dx;
        double c11 = c110 * (1 - dx) + c111 * dx;

        double c0 = c00 * (1 - dy) + c01 * dy;
        double c1 = c10 * (1 - dy) + c11 * dy;

        particle_density[i] = c0 * (1 - dz) + c1 * dz;
    }
}

void deposit_morphology_to_grid(
    const uint8_t* morphology,
    const double* positions,
    int64_t n_particles,
    double box_size,
    int grid_cells,
    OrigamiResult& result)
{
    const double cell_width = box_size / grid_cells;
    const double inv_cell_width = 1.0 / cell_width;
    const int64_t n_cells_total = static_cast<int64_t>(grid_cells) * grid_cells * grid_cells;

    // Initialize grids
    result.grid_cells = grid_cells;
    result.morphology_grid.resize(n_cells_total, 0);
    result.void_fraction_grid.resize(n_cells_total, 0.0f);
    result.wall_fraction_grid.resize(n_cells_total, 0.0f);
    result.filament_fraction_grid.resize(n_cells_total, 0.0f);
    result.halo_fraction_grid.resize(n_cells_total, 0.0f);

    // Count particles per cell per class
    std::vector<int> count_void(n_cells_total, 0);
    std::vector<int> count_wall(n_cells_total, 0);
    std::vector<int> count_filament(n_cells_total, 0);
    std::vector<int> count_halo(n_cells_total, 0);

#ifdef _OPENMP
    // Parallel version with thread-local count arrays
    int max_threads = omp_get_max_threads();

    // Allocate thread-local count arrays
    std::vector<std::vector<int>> tl_void(max_threads);
    std::vector<std::vector<int>> tl_wall(max_threads);
    std::vector<std::vector<int>> tl_filament(max_threads);
    std::vector<std::vector<int>> tl_halo(max_threads);

    for (int t = 0; t < max_threads; ++t) {
        tl_void[t].resize(n_cells_total, 0);
        tl_wall[t].resize(n_cells_total, 0);
        tl_filament[t].resize(n_cells_total, 0);
        tl_halo[t].resize(n_cells_total, 0);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for (int64_t i = 0; i < n_particles; ++i) {
            double px = positions[i * 3 + 0];
            double py = positions[i * 3 + 1];
            double pz = positions[i * 3 + 2];

            // Wrap to box
            while (px < 0) px += box_size;
            while (px >= box_size) px -= box_size;
            while (py < 0) py += box_size;
            while (py >= box_size) py -= box_size;
            while (pz < 0) pz += box_size;
            while (pz >= box_size) pz -= box_size;

            // Get cell index
            int ix = static_cast<int>(px * inv_cell_width) % grid_cells;
            int iy = static_cast<int>(py * inv_cell_width) % grid_cells;
            int iz = static_cast<int>(pz * inv_cell_width) % grid_cells;

            // Grid is [z, y, x] ordered
            int64_t cell_idx = static_cast<int64_t>(iz) * grid_cells * grid_cells +
                               static_cast<int64_t>(iy) * grid_cells +
                               static_cast<int64_t>(ix);

            // Increment thread-local count for this class
            switch (morphology[i]) {
                case 0: ++tl_void[tid][cell_idx]; break;
                case 1: ++tl_wall[tid][cell_idx]; break;
                case 2: ++tl_filament[tid][cell_idx]; break;
                case 3: ++tl_halo[tid][cell_idx]; break;
            }
        }
    }

    // Reduce thread-local counts into global counts
    #pragma omp parallel for
    for (int64_t c = 0; c < n_cells_total; ++c) {
        for (int t = 0; t < max_threads; ++t) {
            count_void[c] += tl_void[t][c];
            count_wall[c] += tl_wall[t][c];
            count_filament[c] += tl_filament[t][c];
            count_halo[c] += tl_halo[t][c];
        }
    }
#else
    // Serial version (no OpenMP)
    for (int64_t i = 0; i < n_particles; ++i) {
        double px = positions[i * 3 + 0];
        double py = positions[i * 3 + 1];
        double pz = positions[i * 3 + 2];

        // Wrap to box
        while (px < 0) px += box_size;
        while (px >= box_size) px -= box_size;
        while (py < 0) py += box_size;
        while (py >= box_size) py -= box_size;
        while (pz < 0) pz += box_size;
        while (pz >= box_size) pz -= box_size;

        // Get cell index
        int ix = static_cast<int>(px * inv_cell_width) % grid_cells;
        int iy = static_cast<int>(py * inv_cell_width) % grid_cells;
        int iz = static_cast<int>(pz * inv_cell_width) % grid_cells;

        // Grid is [z, y, x] ordered
        int64_t cell_idx = static_cast<int64_t>(iz) * grid_cells * grid_cells +
                           static_cast<int64_t>(iy) * grid_cells +
                           static_cast<int64_t>(ix);

        // Increment count for this class
        switch (morphology[i]) {
            case 0: ++count_void[cell_idx]; break;
            case 1: ++count_wall[cell_idx]; break;
            case 2: ++count_filament[cell_idx]; break;
            case 3: ++count_halo[cell_idx]; break;
        }
    }
#endif

    // Compute fractions and dominant class
    #pragma omp parallel for
    for (int64_t c = 0; c < n_cells_total; ++c) {
        int total = count_void[c] + count_wall[c] + count_filament[c] + count_halo[c];

        if (total > 0) {
            float inv_total = 1.0f / static_cast<float>(total);
            result.void_fraction_grid[c] = static_cast<float>(count_void[c]) * inv_total;
            result.wall_fraction_grid[c] = static_cast<float>(count_wall[c]) * inv_total;
            result.filament_fraction_grid[c] = static_cast<float>(count_filament[c]) * inv_total;
            result.halo_fraction_grid[c] = static_cast<float>(count_halo[c]) * inv_total;

            // Find dominant class
            int max_count = count_void[c];
            uint8_t dominant = 0;
            if (count_wall[c] > max_count) { max_count = count_wall[c]; dominant = 1; }
            if (count_filament[c] > max_count) { max_count = count_filament[c]; dominant = 2; }
            if (count_halo[c] > max_count) { dominant = 3; }
            result.morphology_grid[c] = dominant;
        }
    }

    // Compute volume fractions based on dominant class per cell
    int64_t vol_void = 0, vol_wall = 0, vol_filament = 0, vol_halo = 0;

    #pragma omp parallel for reduction(+:vol_void, vol_wall, vol_filament, vol_halo)
    for (int64_t c = 0; c < n_cells_total; ++c) {
        switch (result.morphology_grid[c]) {
            case 0: ++vol_void; break;
            case 1: ++vol_wall; break;
            case 2: ++vol_filament; break;
            case 3: ++vol_halo; break;
        }
    }

    double inv_n_cells = 1.0 / static_cast<double>(n_cells_total);
    result.v_void = static_cast<double>(vol_void) * inv_n_cells;
    result.v_wall = static_cast<double>(vol_wall) * inv_n_cells;
    result.v_filament = static_cast<double>(vol_filament) * inv_n_cells;
    result.v_halo = static_cast<double>(vol_halo) * inv_n_cells;
}

} // namespace origami
} // namespace tessera
