/**
 * @file tetra_density.cpp
 * @brief Implementation of tetrahedron-based density field computation
 * 
 * This implements the core gotetra algorithm with OpenMP parallelization
 * for high-performance density field computation from N-body simulations.
 */

#include "density/tetra_density.h"
#include "geom/tetra.h"
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asymptotic_tetra {
namespace density {

// Direction lookup table for tetrahedron configurations (from gotetra)
// Each Lagrangian cell is split into 6 tetrahedra
// Corner order: [DIRS[dir][0], DIRS[dir][1], (1,1,1), (0,0,0)]
static const int64_t DIRS[6][2][3] = {
    {{1, 0, 0}, {1, 1, 0}},
    {{1, 0, 0}, {1, 0, 1}},
    {{0, 1, 0}, {1, 1, 0}},
    {{0, 0, 1}, {1, 0, 1}},
    {{0, 1, 0}, {0, 1, 1}},
    {{0, 0, 1}, {0, 1, 1}},
};

// =============================================================================
// UnitTetraSamples implementation
// =============================================================================

/**
 * Apply Rocchini-Cignoni transformation to map unit cube point to unit tetrahedron.
 * Returns barycentric coordinates (s, t, u, v) where v = 1-s-t-u.
 */
static std::array<float, 4> rocchini_cignoni_transform(float x, float y, float z) {
    float s = x, t = y, u = z;
    
    // Fold to get uniform distribution in unit tetrahedron
    if (s + t > 1.0f) {
        s = 1.0f - s;
        t = 1.0f - t;
    }
    
    if (t + u > 1.0f) {
        float old_t = t;
        t = 1.0f - u;
        u = 1.0f - s - old_t;
    } else if (s + t + u > 1.0f) {
        float old_s = s;
        s = 1.0f - t - u;
        u = old_s + t + u - 1.0f;
    }
    
    float v = 1.0f - s - t - u;
    return {s, t, u, v};
}

UnitTetraSamples::UnitTetraSamples(int n_buffers, int n_samples, uint64_t seed)
    : n_samples_(n_samples)
{
    // Initialize random generator
    if (seed == 0) {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    buffers_.resize(n_buffers);
    for (int i = 0; i < n_buffers; ++i) {
        buffers_[i].resize(n_samples);
        for (int j = 0; j < n_samples; ++j) {
            float x = dist(gen);
            float y = dist(gen);
            float z = dist(gen);
            buffers_[i][j] = rocchini_cignoni_transform(x, y, z);
        }
    }
}

// =============================================================================
// Tetrahedron index generation
// =============================================================================

std::vector<std::array<int64_t, 4>> generate_tetra_indices(int grid_size, bool periodic) {
    int n_cells_per_dim = periodic ? grid_size : grid_size - 1;
    int64_t n_cells = static_cast<int64_t>(n_cells_per_dim) * n_cells_per_dim * n_cells_per_dim;
    int64_t n_tetra = n_cells * 6;
    
    std::vector<std::array<int64_t, 4>> indices(n_tetra);
    
    auto idx_3d_to_1d = [grid_size](int64_t x, int64_t y, int64_t z) -> int64_t {
        // Apply periodic wrapping
        x = ((x % grid_size) + grid_size) % grid_size;
        y = ((y % grid_size) + grid_size) % grid_size;
        z = ((z % grid_size) + grid_size) % grid_size;
        return x + y * grid_size + z * static_cast<int64_t>(grid_size) * grid_size;
    };
    
    int64_t tetra_i = 0;
    for (int iz = 0; iz < n_cells_per_dim; ++iz) {
        for (int iy = 0; iy < n_cells_per_dim; ++iy) {
            for (int ix = 0; ix < n_cells_per_dim; ++ix) {
                for (int dir = 0; dir < 6; ++dir) {
                    // Corner 0: origin + DIRS[dir][0]
                    indices[tetra_i][0] = idx_3d_to_1d(
                        ix + DIRS[dir][0][0],
                        iy + DIRS[dir][0][1],
                        iz + DIRS[dir][0][2]
                    );
                    // Corner 1: origin + DIRS[dir][1]
                    indices[tetra_i][1] = idx_3d_to_1d(
                        ix + DIRS[dir][1][0],
                        iy + DIRS[dir][1][1],
                        iz + DIRS[dir][1][2]
                    );
                    // Corner 2: origin + (1,1,1)
                    indices[tetra_i][2] = idx_3d_to_1d(ix + 1, iy + 1, iz + 1);
                    // Corner 3: origin
                    indices[tetra_i][3] = idx_3d_to_1d(ix, iy, iz);
                    
                    tetra_i++;
                }
            }
        }
    }
    
    return indices;
}

// =============================================================================
// Sorting by Lagrangian ID
// =============================================================================

void sort_by_lagrangian_id(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double* sorted_positions,
    int64_t id_offset)
{
    int64_t expected_n = static_cast<int64_t>(grid_size) * grid_size * grid_size;
    if (n_particles != expected_n) {
        throw std::runtime_error("Particle count doesn't match grid_size^3");
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n_particles; ++i) {
        int64_t lag_idx = particle_ids[i] - id_offset;
        if (lag_idx < 0 || lag_idx >= n_particles) {
            continue;  // Skip invalid IDs
        }
        sorted_positions[lag_idx * 3 + 0] = positions[i * 3 + 0];
        sorted_positions[lag_idx * 3 + 1] = positions[i * 3 + 1];
        sorted_positions[lag_idx * 3 + 2] = positions[i * 3 + 2];
    }
}

void sort_by_lagrangian_id(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    float* sorted_positions,
    int64_t id_offset)
{
    int64_t expected_n = static_cast<int64_t>(grid_size) * grid_size * grid_size;
    if (n_particles != expected_n) {
        throw std::runtime_error("Particle count doesn't match grid_size^3");
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n_particles; ++i) {
        int64_t lag_idx = particle_ids[i] - id_offset;
        if (lag_idx < 0 || lag_idx >= n_particles) {
            continue;
        }
        sorted_positions[lag_idx * 3 + 0] = positions[i * 3 + 0];
        sorted_positions[lag_idx * 3 + 1] = positions[i * 3 + 1];
        sorted_positions[lag_idx * 3 + 2] = positions[i * 3 + 2];
    }
}

// =============================================================================
// Core density computation
// =============================================================================

/**
 * Transform barycentric coordinates to physical position within a tetrahedron.
 * 
 * This follows gotetra's distribute_tetra:
 * 1. Compute barycenter of tetrahedron
 * 2. Compute displacement vectors from barycenter to each corner
 * 3. Position = barycenter + s*d0 + t*d1 + u*d2 + v*d3
 */
template<typename T>
static inline void barycentric_to_physical(
    const T corners[4][3],        // 4 corner positions
    const std::array<float, 4>& bary,  // Barycentric coords (s,t,u,v)
    double box_size,
    double& out_x, double& out_y, double& out_z)
{
    // Compute barycenter
    double bary_x = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0;
    double bary_y = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0;
    double bary_z = (corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2]) / 4.0;
    
    // Compute displacements from barycenter to corners
    double d[4][3];
    for (int i = 0; i < 4; ++i) {
        d[i][0] = corners[i][0] - bary_x;
        d[i][1] = corners[i][1] - bary_y;
        d[i][2] = corners[i][2] - bary_z;
    }
    
    // Compute position: bary + s*d0 + t*d1 + u*d2 + v*d3
    float s = bary[0], t = bary[1], u = bary[2], v = bary[3];
    out_x = bary_x + s * d[0][0] + t * d[1][0] + u * d[2][0] + v * d[3][0];
    out_y = bary_y + s * d[0][1] + t * d[1][1] + u * d[2][1] + v * d[3][1];
    out_z = bary_z + s * d[0][2] + t * d[1][2] + u * d[2][2] + v * d[3][2];
    
    // Apply periodic boundary conditions
    if (out_x < 0) out_x += box_size;
    else if (out_x >= box_size) out_x -= box_size;
    if (out_y < 0) out_y += box_size;
    else if (out_y >= box_size) out_y -= box_size;
    if (out_z < 0) out_z += box_size;
    else if (out_z >= box_size) out_z -= box_size;
}

/**
 * Unwrap corners relative to corner 0 to handle periodic boundaries.
 * Tetrahedra near box boundaries may have corners on opposite sides.
 */
template<typename T>
static inline void unwrap_corners(
    const T* positions,
    const std::array<int64_t, 4>& idx,
    double box_size,
    double corners[4][3])
{
    double half_box = box_size / 2.0;
    
    // Reference corner (corner 3 = origin in gotetra convention)
    corners[0][0] = positions[idx[0] * 3 + 0];
    corners[0][1] = positions[idx[0] * 3 + 1];
    corners[0][2] = positions[idx[0] * 3 + 2];
    
    // Unwrap other corners relative to corner 0
    for (int c = 1; c < 4; ++c) {
        for (int d = 0; d < 3; ++d) {
            double pos = positions[idx[c] * 3 + d];
            double diff = pos - corners[0][d];
            
            if (diff > half_box) {
                pos -= box_size;
            } else if (diff < -half_box) {
                pos += box_size;
            }
            corners[c][d] = pos;
        }
    }
}

/**
 * Core 3D density computation with OpenMP parallelization.
 */
template<typename T>
static TetraDensityResult3D compute_density_3d_impl(
    const T* positions,
    const TetraDensityConfig& config,
    const UnitTetraSamples* samples_ptr)
{
    int grid_size = config.lagrangian_grid_size;
    int output_cells = config.output_cells;
    double box_size = config.box_size;
    double particle_mass = config.particle_mass;
    int n_samples = config.n_samples;
    
    // Verify particle count
    int64_t n_particles = static_cast<int64_t>(grid_size) * grid_size * grid_size;
    
    // Set up threading
    int n_threads = config.n_threads;
#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    omp_set_num_threads(n_threads);
#else
    n_threads = 1;
#endif
    
    // Generate or use provided samples
    std::unique_ptr<UnitTetraSamples> local_samples;
    const UnitTetraSamples* samples;
    if (samples_ptr) {
        samples = samples_ptr;
    } else {
        local_samples = std::make_unique<UnitTetraSamples>(64, n_samples, config.seed);
        samples = local_samples.get();
    }
    
    // Generate tetrahedron indices
    auto tetra_indices = generate_tetra_indices(grid_size, config.periodic);
    int64_t n_tetra = static_cast<int64_t>(tetra_indices.size());
    
    // Compute mass per sample point
    // Each Lagrangian cell contributes 6 tetrahedra
    // Each tetrahedron gets 1/6 of a particle's mass
    // Spread over n_samples points
    double mass_per_sample = particle_mass / static_cast<double>(n_samples) / 6.0;
    
    // Cell dimensions
    double cell_width = box_size / output_cells;
    double inv_cell_width = 1.0 / cell_width;
    int64_t cells_sq = static_cast<int64_t>(output_cells) * output_cells;
    int64_t cells_cu = cells_sq * output_cells;
    
    // Thread-local density grids (avoid atomics)
    std::vector<std::vector<double>> thread_grids(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        thread_grids[t].resize(cells_cu, 0.0);
    }
    
    // Main parallel loop over tetrahedra
    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        double* local_density = thread_grids[tid].data();
        
        // Per-thread random buffer selection
        std::mt19937 local_gen(config.seed + tid + 1);
        std::uniform_int_distribution<int> buf_dist(0, samples->n_buffers() - 1);
        
        double corners[4][3];
        
        #pragma omp for schedule(dynamic, 256)
        for (int64_t ti = 0; ti < n_tetra; ++ti) {
            const auto& idx = tetra_indices[ti];
            
            // Get and unwrap corner positions
            unwrap_corners(positions, idx, box_size, corners);
            
            // Get sample buffer for this tetrahedron
            int buf_idx = buf_dist(local_gen);
            const auto& sample_buf = samples->get_buffer(buf_idx);
            
            // Distribute samples and deposit onto grid
            for (int si = 0; si < n_samples; ++si) {
                double px, py, pz;
                barycentric_to_physical(corners, sample_buf[si], box_size, px, py, pz);
                
                // Convert to grid indices (with periodic wrapping)
                int ix = static_cast<int>(px * inv_cell_width);
                int iy = static_cast<int>(py * inv_cell_width);
                int iz = static_cast<int>(pz * inv_cell_width);
                
                // Ensure indices are in bounds (handle edge cases)
                ix = std::max(0, std::min(output_cells - 1, ix));
                iy = std::max(0, std::min(output_cells - 1, iy));
                iz = std::max(0, std::min(output_cells - 1, iz));
                
                int64_t cell_idx = ix + iy * output_cells + iz * cells_sq;
                local_density[cell_idx] += mass_per_sample;
            }
        }
    }
    
    // Reduce thread-local grids
    std::vector<double> density(cells_cu, 0.0);
    for (int t = 0; t < n_threads; ++t) {
        for (int64_t i = 0; i < cells_cu; ++i) {
            density[i] += thread_grids[t][i];
        }
    }
    
    // Convert mass to density (mass per unit volume)
    double cell_volume = cell_width * cell_width * cell_width;
    double inv_cell_volume = 1.0 / cell_volume;
    for (int64_t i = 0; i < cells_cu; ++i) {
        density[i] *= inv_cell_volume;
    }
    
    // Compute statistics
    double total_mass = std::accumulate(density.begin(), density.end(), 0.0) * cell_volume;
    double mean_density = total_mass / (box_size * box_size * box_size);
    
    TetraDensityResult3D result;
    result.density = std::move(density);
    result.cells = output_cells;
    result.cell_width = cell_width;
    result.total_mass = total_mass;
    result.mean_density = mean_density;
    result.n_tetrahedra = n_tetra;
    
    return result;
}

// Explicit instantiations
TetraDensityResult3D compute_tetra_density_3d(
    const double* positions,
    const TetraDensityConfig& config,
    const UnitTetraSamples* samples)
{
    return compute_density_3d_impl(positions, config, samples);
}

TetraDensityResult3D compute_tetra_density_3d(
    const float* positions,
    const TetraDensityConfig& config,
    const UnitTetraSamples* samples)
{
    return compute_density_3d_impl(positions, config, samples);
}

// =============================================================================
// 2D projection/slice implementations
// =============================================================================

TetraDensityResult2D compute_tetra_density_2d_projection(
    const double* positions,
    const TetraDensityConfig& config,
    int projection_axis,
    const UnitTetraSamples* samples_ptr)
{
    // First compute full 3D density
    auto result_3d = compute_tetra_density_3d(positions, config, samples_ptr);
    
    int cells = config.output_cells;
    int64_t cells_sq = static_cast<int64_t>(cells) * cells;
    double cell_width = config.box_size / cells;
    
    // Project along specified axis
    std::vector<double> density_2d(cells_sq, 0.0);
    
    // Determine projection indices
    // axis=0 (x): sum over x, keep (y,z) -> output indexed by (y,z)
    // axis=1 (y): sum over y, keep (x,z) -> output indexed by (x,z)  
    // axis=2 (z): sum over z, keep (x,y) -> output indexed by (x,y)
    
    #pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < cells; ++i1) {
        for (int i2 = 0; i2 < cells; ++i2) {
            double sum = 0.0;
            for (int ip = 0; ip < cells; ++ip) {
                int64_t idx_3d;
                if (projection_axis == 0) {
                    // Project x: sum over x, output (y=i1, z=i2)
                    idx_3d = ip + i1 * cells + i2 * cells_sq;
                } else if (projection_axis == 1) {
                    // Project y: sum over y, output (x=i1, z=i2)
                    idx_3d = i1 + ip * cells + i2 * cells_sq;
                } else {
                    // Project z: sum over z, output (x=i1, y=i2)
                    idx_3d = i1 + i2 * cells + ip * cells_sq;
                }
                sum += result_3d.density[idx_3d];
            }
            // Multiply by cell_width to convert to surface density
            density_2d[i1 + i2 * cells] = sum * cell_width;
        }
    }
    
    // Compute mean surface density
    double total_surface_mass = std::accumulate(density_2d.begin(), density_2d.end(), 0.0);
    double mean_surface_density = total_surface_mass / cells_sq;
    
    TetraDensityResult2D result;
    result.density = std::move(density_2d);
    result.cells = cells;
    result.cell_width = cell_width;
    result.projection_axis = projection_axis;
    result.slice_min = 0.0;
    result.slice_max = config.box_size;
    result.mean_surface_density = mean_surface_density;
    
    return result;
}

TetraDensityResult2D compute_tetra_density_2d_slice(
    const double* positions,
    const TetraDensityConfig& config,
    int projection_axis,
    double slice_min,
    double slice_max,
    const UnitTetraSamples* samples_ptr)
{
    // First compute full 3D density
    auto result_3d = compute_tetra_density_3d(positions, config, samples_ptr);
    
    int cells = config.output_cells;
    int64_t cells_sq = static_cast<int64_t>(cells) * cells;
    double cell_width = config.box_size / cells;
    
    // Find cell indices for slice bounds
    int idx_min = std::max(0, static_cast<int>(slice_min / cell_width));
    int idx_max = std::min(cells, static_cast<int>(std::ceil(slice_max / cell_width)));
    
    // Project slice along specified axis
    std::vector<double> density_2d(cells_sq, 0.0);
    
    #pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < cells; ++i1) {
        for (int i2 = 0; i2 < cells; ++i2) {
            double sum = 0.0;
            for (int ip = idx_min; ip < idx_max; ++ip) {
                int64_t idx_3d;
                if (projection_axis == 0) {
                    idx_3d = ip + i1 * cells + i2 * cells_sq;
                } else if (projection_axis == 1) {
                    idx_3d = i1 + ip * cells + i2 * cells_sq;
                } else {
                    idx_3d = i1 + i2 * cells + ip * cells_sq;
                }
                sum += result_3d.density[idx_3d];
            }
            density_2d[i1 + i2 * cells] = sum * cell_width;
        }
    }
    
    double total = std::accumulate(density_2d.begin(), density_2d.end(), 0.0);
    double mean_surface_density = total / cells_sq;
    
    TetraDensityResult2D result;
    result.density = std::move(density_2d);
    result.cells = cells;
    result.cell_width = cell_width;
    result.projection_axis = projection_axis;
    result.slice_min = slice_min;
    result.slice_max = slice_max;
    result.mean_surface_density = mean_surface_density;
    
    return result;
}

} // namespace density
} // namespace asymptotic_tetra
