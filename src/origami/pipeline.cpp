/**
 * @file pipeline.cpp
 * @brief Implementation of the unified ORIGAMI pipeline
 */

#include "origami/pipeline.h"
#include "density/tetra_density.h"
#include "density/cic_density.h"
#include "stats/histogram.h"
#ifdef TESSERA_HAS_VORO
#include "density/voronoi_density.h"
#endif

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tessera {
namespace origami {

namespace {

/**
 * Compute periodic minimum image difference.
 */
inline double periodic_diff(double a, double b, double box_size) {
    double diff = a - b;
    if (diff > box_size / 2.0) diff -= box_size;
    if (diff < -box_size / 2.0) diff += box_size;
    return diff;
}

/**
 * Compute target index for z-major to x-major conversion.
 */
inline int64_t zmajor_to_xmajor(int64_t id, int64_t grid_size) {
    int64_t z = id % grid_size;
    int64_t y = (id / grid_size) % grid_size;
    int64_t x = id / (grid_size * grid_size);
    return x + y * grid_size + z * grid_size * grid_size;
}

/**
 * Get high-resolution timer in milliseconds.
 */
inline double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // anonymous namespace

// ============================================================================
// detect_id_ordering - Template implementation
// ============================================================================

template<typename T>
IdOrdering detect_id_ordering_impl(
    const T* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size)
{
    // Build a lookup from ID to array index
    std::unordered_map<int64_t, int64_t> id_to_idx;
    id_to_idx.reserve(std::min(n_particles, int64_t(100000)));

    // Only need a sample for detection
    for (int64_t i = 0; i < n_particles && id_to_idx.size() < 100000; ++i) {
        id_to_idx[particle_ids[i]] = i;
    }

    auto get_pos = [&](int64_t pid, double* out) -> bool {
        auto it = id_to_idx.find(pid);
        if (it == id_to_idx.end()) return false;
        int64_t idx = it->second;
        out[0] = static_cast<double>(positions[idx * 3 + 0]);
        out[1] = static_cast<double>(positions[idx * 3 + 1]);
        out[2] = static_cast<double>(positions[idx * 3 + 2]);
        return true;
    };

    double x_major_score = 0.0;
    double z_major_score = 0.0;
    int n_valid = 0;

    double spacing = box_size / grid_size;

    // Test several ID pairs spread across the domain
    std::vector<int64_t> test_ids = {
        1,
        static_cast<int64_t>(grid_size) / 4,
        static_cast<int64_t>(grid_size) / 2,
        static_cast<int64_t>(grid_size) * 10,
        static_cast<int64_t>(grid_size) * 100
    };

    for (int64_t base_id : test_ids) {
        if (base_id < 1 || base_id >= n_particles) continue;

        double p1[3], p2[3];
        if (!get_pos(base_id, p1)) continue;
        if (!get_pos(base_id + 1, p2)) continue;

        // Compute periodic displacement
        double diff[3];
        diff[0] = periodic_diff(p2[0], p1[0], box_size);
        diff[1] = periodic_diff(p2[1], p1[1], box_size);
        diff[2] = periodic_diff(p2[2], p1[2], box_size);

        // For z-major: expect diff ~= (0, 0, spacing)
        // For x-major: expect diff ~= (spacing, 0, 0)
        double z_major_expected[3] = {0.0, 0.0, spacing};
        double x_major_expected[3] = {spacing, 0.0, 0.0};

        double z_err = 0.0, x_err = 0.0;
        for (int d = 0; d < 3; ++d) {
            z_err += (diff[d] - z_major_expected[d]) * (diff[d] - z_major_expected[d]);
            x_err += (diff[d] - x_major_expected[d]) * (diff[d] - x_major_expected[d]);
        }
        z_major_score += std::sqrt(z_err);
        x_major_score += std::sqrt(x_err);
        ++n_valid;
    }

    // Also check y-layer transitions for additional robustness
    for (int64_t base_id : test_ids) {
        int64_t next_y_id = base_id + grid_size;
        if (next_y_id >= n_particles) continue;

        double p1[3], p2[3];
        if (!get_pos(base_id, p1)) continue;
        if (!get_pos(next_y_id, p2)) continue;

        double diff[3];
        diff[0] = periodic_diff(p2[0], p1[0], box_size);
        diff[1] = periodic_diff(p2[1], p1[1], box_size);
        diff[2] = periodic_diff(p2[2], p1[2], box_size);

        // z-major: x should stay near 0
        // x-major: z should stay near 0
        z_major_score += std::abs(diff[0]);
        x_major_score += std::abs(diff[2]);
    }

    if (n_valid == 0) {
        return IdOrdering::ZMajor;  // Default fallback
    }

    return (x_major_score < z_major_score) ? IdOrdering::XMajor : IdOrdering::ZMajor;
}

// Explicit instantiations
IdOrdering detect_id_ordering(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size)
{
    return detect_id_ordering_impl(positions, particle_ids, n_particles, grid_size, box_size);
}

IdOrdering detect_id_ordering(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size)
{
    return detect_id_ordering_impl(positions, particle_ids, n_particles, grid_size, box_size);
}

// ============================================================================
// validate_particle_ids
// ============================================================================

void validate_particle_ids(
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    int64_t id_offset)
{
    int64_t expected_n = static_cast<int64_t>(grid_size) * grid_size * grid_size;

    if (n_particles != expected_n) {
        throw std::runtime_error(
            "Particle count " + std::to_string(n_particles) +
            " doesn't match grid_size^3 = " + std::to_string(expected_n));
    }

    int64_t min_id = id_offset;
    int64_t max_id = id_offset + expected_n - 1;

    // Parallel range check
    bool has_invalid = false;

    #pragma omp parallel for reduction(||:has_invalid)
    for (int64_t i = 0; i < n_particles; ++i) {
        if (particle_ids[i] < min_id || particle_ids[i] > max_id) {
            has_invalid = true;
        }
    }

    if (has_invalid) {
        throw std::runtime_error(
            "Particle IDs out of expected range [" + std::to_string(min_id) +
            ", " + std::to_string(max_id) + "]");
    }

    // Parallel uniqueness check using atomic flags
    // For very large N, this uses ~1GB for the seen array
    std::vector<std::atomic<bool>> seen(expected_n);
    for (auto& a : seen) a.store(false, std::memory_order_relaxed);

    bool has_duplicate = false;

    #pragma omp parallel for reduction(||:has_duplicate)
    for (int64_t i = 0; i < n_particles; ++i) {
        int64_t idx = particle_ids[i] - id_offset;
        bool was_seen = seen[idx].exchange(true, std::memory_order_relaxed);
        if (was_seen) {
            has_duplicate = true;
        }
    }

    if (has_duplicate) {
        throw std::runtime_error("Duplicate particle IDs found");
    }
}

// ============================================================================
// sort_to_lagrangian - Out-of-place version
// ============================================================================

template<typename T>
void sort_to_lagrangian_impl(
    const T* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset,
    T* sorted_output)
{
    int64_t gs = static_cast<int64_t>(grid_size);

    #pragma omp parallel for
    for (int64_t i = 0; i < n_particles; ++i) {
        int64_t id = particle_ids[i] - id_offset;
        int64_t target_idx;

        if (ordering == IdOrdering::ZMajor) {
            // z-major to x-major conversion
            target_idx = zmajor_to_xmajor(id, gs);
        } else {
            // x-major: direct mapping
            target_idx = id;
        }

        // Copy position to target location
        sorted_output[target_idx * 3 + 0] = positions[i * 3 + 0];
        sorted_output[target_idx * 3 + 1] = positions[i * 3 + 1];
        sorted_output[target_idx * 3 + 2] = positions[i * 3 + 2];
    }
}

void sort_to_lagrangian(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset,
    double* sorted_output)
{
    sort_to_lagrangian_impl(positions, particle_ids, n_particles, grid_size,
                            ordering, id_offset, sorted_output);
}

void sort_to_lagrangian(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset,
    float* sorted_output)
{
    sort_to_lagrangian_impl(positions, particle_ids, n_particles, grid_size,
                            ordering, id_offset, sorted_output);
}

// ============================================================================
// sort_to_lagrangian_inplace - In-place version using cycle decomposition
// ============================================================================

template<typename T>
void sort_to_lagrangian_inplace_impl(
    T* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset)
{
    int64_t gs = static_cast<int64_t>(grid_size);

    // Phase 1: Build inverse permutation array
    // inv_perm[target_idx] = source_idx
    // For N=1024^3, this is ~8GB
    std::vector<int64_t> inv_perm(n_particles);

    #pragma omp parallel for
    for (int64_t i = 0; i < n_particles; ++i) {
        int64_t id = particle_ids[i] - id_offset;
        int64_t target_idx;

        if (ordering == IdOrdering::ZMajor) {
            target_idx = zmajor_to_xmajor(id, gs);
        } else {
            target_idx = id;
        }

        inv_perm[target_idx] = i;
    }

    // Phase 2: Apply permutation in-place using cycle decomposition
    // For N=1024^3, visited array is ~1GB
    std::vector<bool> visited(n_particles, false);

    for (int64_t start = 0; start < n_particles; ++start) {
        if (visited[start]) continue;

        // Check if this is a fixed point
        if (inv_perm[start] == start) {
            visited[start] = true;
            continue;
        }

        // Follow the cycle starting at 'start'
        int64_t current = start;
        T temp[3];
        temp[0] = positions[start * 3 + 0];
        temp[1] = positions[start * 3 + 1];
        temp[2] = positions[start * 3 + 2];

        while (!visited[current]) {
            visited[current] = true;
            int64_t src = inv_perm[current];

            if (src == start) {
                // End of cycle - place the saved element
                positions[current * 3 + 0] = temp[0];
                positions[current * 3 + 1] = temp[1];
                positions[current * 3 + 2] = temp[2];
                break;
            }

            // Move element from src to current
            positions[current * 3 + 0] = positions[src * 3 + 0];
            positions[current * 3 + 1] = positions[src * 3 + 1];
            positions[current * 3 + 2] = positions[src * 3 + 2];

            current = src;
        }
    }
}

void sort_to_lagrangian_inplace(
    double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset)
{
    sort_to_lagrangian_inplace_impl(positions, particle_ids, n_particles,
                                     grid_size, ordering, id_offset);
}

void sort_to_lagrangian_inplace(
    float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset)
{
    sort_to_lagrangian_inplace_impl(positions, particle_ids, n_particles,
                                     grid_size, ordering, id_offset);
}

// ============================================================================
// sort_particles_to_lagrangian - Convenience function with auto-detection
// ============================================================================

template<typename T>
std::vector<T> sort_particles_to_lagrangian_impl(
    const T* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size,
    IdOrdering* detected_ordering,
    int64_t* detected_offset)
{
    // Step 1: Detect ID ordering
    IdOrdering ordering = detect_id_ordering(positions, particle_ids, n_particles,
                                              grid_size, box_size);
    if (detected_ordering) {
        *detected_ordering = ordering;
    }

    // Step 2: Detect ID offset (minimum particle ID)
    int64_t id_offset = particle_ids[0];
    for (int64_t i = 1; i < n_particles; ++i) {
        if (particle_ids[i] < id_offset) {
            id_offset = particle_ids[i];
        }
    }
    if (detected_offset) {
        *detected_offset = id_offset;
    }

    // Step 3: Allocate output and sort
    std::vector<T> sorted_output(n_particles * 3);
    sort_to_lagrangian_impl(positions, particle_ids, n_particles, grid_size,
                            ordering, id_offset, sorted_output.data());

    return sorted_output;
}

std::vector<double> sort_particles_to_lagrangian(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size,
    IdOrdering* detected_ordering,
    int64_t* detected_offset)
{
    return sort_particles_to_lagrangian_impl(positions, particle_ids, n_particles,
                                              grid_size, box_size,
                                              detected_ordering, detected_offset);
}

std::vector<float> sort_particles_to_lagrangian(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size,
    IdOrdering* detected_ordering,
    int64_t* detected_offset)
{
    return sort_particles_to_lagrangian_impl(positions, particle_ids, n_particles,
                                              grid_size, box_size,
                                              detected_ordering, detected_offset);
}

// ============================================================================
// run_pipeline - Main unified entry point
// ============================================================================

template<typename T>
OrigamiPipelineResult run_pipeline_impl(
    T* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config)
{
    OrigamiPipelineResult result;
    double start_time = get_time_ms();

    int64_t n_particles = static_cast<int64_t>(config.lagrangian_grid_size) *
                          config.lagrangian_grid_size * config.lagrangian_grid_size;

    // === Validation ===
    if (config.lagrangian_grid_size <= 0) {
        throw std::runtime_error("lagrangian_grid_size must be positive");
    }
    if (config.box_size <= 0) {
        throw std::runtime_error("box_size must be positive");
    }

    if (config.validate_ids) {
        validate_particle_ids(particle_ids, n_particles,
                              config.lagrangian_grid_size, config.id_offset);
    }

    // === ID Ordering Detection ===
    double t0 = get_time_ms();
    IdOrdering ordering = config.id_ordering;
    if (ordering == IdOrdering::Auto) {
        ordering = detect_id_ordering(positions, particle_ids, n_particles,
                                       config.lagrangian_grid_size, config.box_size);
    }
    result.detected_ordering = ordering;
    result.detection_time_ms = get_time_ms() - t0;

    // === Sorting ===
    t0 = get_time_ms();
    std::vector<double> sorted_positions_storage;
    double* sorted_ptr;

    if (config.positions_already_sorted) {
        // Use positions directly (cast to double if needed)
        if constexpr (std::is_same_v<T, double>) {
            sorted_ptr = positions;
        } else {
            // Need to convert float to double for ORIGAMI
            sorted_positions_storage.resize(n_particles * 3);
            #pragma omp parallel for
            for (int64_t i = 0; i < n_particles * 3; ++i) {
                sorted_positions_storage[i] = static_cast<double>(positions[i]);
            }
            sorted_ptr = sorted_positions_storage.data();
        }
        result.sorting_performed = false;
    } else if (config.sort_in_place) {
        // Sort in-place
        sort_to_lagrangian_inplace_impl(positions, particle_ids, n_particles,
                                         config.lagrangian_grid_size, ordering,
                                         config.id_offset);
        result.sorting_performed = true;

        // Convert to double for ORIGAMI if needed
        if constexpr (std::is_same_v<T, double>) {
            sorted_ptr = positions;
        } else {
            sorted_positions_storage.resize(n_particles * 3);
            #pragma omp parallel for
            for (int64_t i = 0; i < n_particles * 3; ++i) {
                sorted_positions_storage[i] = static_cast<double>(positions[i]);
            }
            sorted_ptr = sorted_positions_storage.data();
        }
    } else {
        // Out-of-place sort (allocate new array)
        sorted_positions_storage.resize(n_particles * 3);

        if constexpr (std::is_same_v<T, double>) {
            // Direct double-to-double sorting
            sort_to_lagrangian_impl(positions, particle_ids, n_particles,
                                     config.lagrangian_grid_size, ordering,
                                     config.id_offset, sorted_positions_storage.data());
        } else {
            // Float input: convert while sorting
            #pragma omp parallel for
            for (int64_t i = 0; i < n_particles; ++i) {
                int64_t id = particle_ids[i] - config.id_offset;
                int64_t gs = static_cast<int64_t>(config.lagrangian_grid_size);
                int64_t target_idx;

                if (ordering == IdOrdering::ZMajor) {
                    target_idx = zmajor_to_xmajor(id, gs);
                } else {
                    target_idx = id;
                }

                // Convert float to double while placing in sorted order
                sorted_positions_storage[target_idx * 3 + 0] = static_cast<double>(positions[i * 3 + 0]);
                sorted_positions_storage[target_idx * 3 + 1] = static_cast<double>(positions[i * 3 + 1]);
                sorted_positions_storage[target_idx * 3 + 2] = static_cast<double>(positions[i * 3 + 2]);
            }
        }

        sorted_ptr = sorted_positions_storage.data();
        result.sorting_performed = true;
    }
    result.sorting_time_ms = get_time_ms() - t0;

    // === ORIGAMI Morphology ===
    t0 = get_time_ms();
    OrigamiConfig morph_config;
    morph_config.lagrangian_grid_size = config.lagrangian_grid_size;
    morph_config.box_size = config.box_size;
    morph_config.n_threads = config.n_threads;
    morph_config.n_split = config.n_split;
    morph_config.linear_regime_threshold = config.linear_regime_threshold;
    morph_config.cartesian_only = config.cartesian_only;

    OrigamiResult morph_result = compute_morphology(sorted_ptr, morph_config);

    // Copy morphology results
    result.morphology = std::move(morph_result.morphology);
    result.n_void = morph_result.n_void;
    result.n_wall = morph_result.n_wall;
    result.n_filament = morph_result.n_filament;
    result.n_halo = morph_result.n_halo;
    result.f_void = morph_result.f_void;
    result.f_wall = morph_result.f_wall;
    result.f_filament = morph_result.f_filament;
    result.f_halo = morph_result.f_halo;
    result.is_linear_regime = morph_result.is_linear_regime;
    result.linear_regime_threshold = morph_result.linear_regime_threshold;

    result.morphology_time_ms = get_time_ms() - t0;

    // === Optional: Density Field ===
    if (config.use_cic_density) {
        // Use CIC (Cloud-in-Cell) grid density: fastest method, grid-smoothed.
        // Does NOT require Lagrangian sorting — works on Eulerian positions.
        t0 = get_time_ms();

        density::CICDensityConfig cic_config;
        cic_config.box_size = config.box_size;
        cic_config.output_cells = config.cic_output_cells;
        cic_config.particle_mass = config.particle_mass;
        cic_config.n_threads = config.n_threads;
        cic_config.periodic = config.density_periodic;
        cic_config.sample_at_particles = true;

        auto cic_result = density::compute_cic_density(
            sorted_ptr, n_particles, cic_config);

        result.particle_density = std::move(cic_result.particle_density);
        result.density_3d = std::move(cic_result.grid_density);
        result.mean_density = cic_result.mean_density;
        result.density_cells = cic_result.output_cells;
        result.density_cell_width = cic_result.cell_width;

        result.density_time_ms = get_time_ms() - t0;
        result.sampling_time_ms = 0.0;

    } else
#ifdef TESSERA_HAS_VORO
    if (config.use_voronoi_density) {
        // Use Voronoi (VTFE) density: density_i = mean_density * V_mean / V_i
        // Does NOT require Lagrangian sorting — works on Eulerian positions.
        // Resolves high-density tail up to 10^4-10^6.
        t0 = get_time_ms();

        density::VoronoiDensityConfig voro_config;
        voro_config.box_size = config.box_size;
        voro_config.n_threads = config.n_threads;
        voro_config.compute_volumes = false;  // Only need density

        auto voro_result = density::compute_voronoi_density(
            sorted_ptr, n_particles, voro_config);

        result.particle_density = std::move(voro_result.density);
        result.mean_density = voro_result.mean_density;
        result.density_cells = 0;
        result.density_cell_width = 0.0;

        result.density_time_ms = get_time_ms() - t0;
        result.sampling_time_ms = 0.0;

    } else
#endif
    if (config.use_direct_particle_density) {
        // Use direct particle density method (memory-efficient, no 3D grid)
        // This computes density directly at particle positions from tetrahedra volumes
        t0 = get_time_ms();

        density::ParticleDensityConfig direct_config;
        direct_config.lagrangian_grid_size = config.lagrangian_grid_size;
        direct_config.box_size = config.box_size;
        direct_config.particle_mass = config.particle_mass;
        direct_config.n_threads = config.n_threads;
        direct_config.periodic = config.density_periodic;

        auto direct_result = density::compute_particle_density(sorted_ptr, direct_config);

        result.particle_density = std::move(direct_result.density);
        result.mean_density = direct_result.mean_density;
        // Note: density_3d is left empty when using direct method
        result.density_cells = 0;
        result.density_cell_width = 0.0;

        result.density_time_ms = get_time_ms() - t0;
        result.sampling_time_ms = 0.0;  // No separate sampling step needed

    } else if (config.density_output_cells > 0) {
        // Use grid-based Monte Carlo density (default method)
        t0 = get_time_ms();

        density::TetraDensityConfig density_config;
        density_config.lagrangian_grid_size = config.lagrangian_grid_size;
        density_config.output_cells = config.density_output_cells;
        density_config.box_size = config.box_size;
        density_config.particle_mass = config.particle_mass;
        density_config.n_samples = config.density_n_samples;
        density_config.n_threads = config.n_threads;
        density_config.periodic = config.density_periodic;
        density_config.seed = config.seed;

        auto density_result = density::compute_tetra_density_3d(sorted_ptr, density_config);

        result.density_3d = std::move(density_result.density);
        result.density_cells = config.density_output_cells;
        result.density_cell_width = config.box_size / config.density_output_cells;
        result.mean_density = density_result.mean_density;

        result.density_time_ms = get_time_ms() - t0;

        // === Optional: Sample Density at Particles ===
        if (config.sample_density_at_particles && !result.density_3d.empty()) {
            t0 = get_time_ms();

            result.particle_density.resize(n_particles);
            sample_density_at_particles(
                result.density_3d.data(),
                config.density_output_cells,
                sorted_ptr,
                n_particles,
                config.box_size,
                result.particle_density.data()
            );

            result.sampling_time_ms = get_time_ms() - t0;
        }
    }

    // === Optional: Grid Deposition ===
    if (config.grid_cells > 0) {
        t0 = get_time_ms();

        // Create a temporary OrigamiResult for the grid deposition function
        OrigamiResult grid_result;
        grid_result.morphology = result.morphology;  // Copy morphology

        deposit_morphology_to_grid(
            grid_result.morphology.data(),
            sorted_ptr,
            n_particles,
            config.box_size,
            config.grid_cells,
            grid_result
        );

        // Move grid results
        result.morphology_grid = std::move(grid_result.morphology_grid);
        result.void_fraction_grid = std::move(grid_result.void_fraction_grid);
        result.wall_fraction_grid = std::move(grid_result.wall_fraction_grid);
        result.filament_fraction_grid = std::move(grid_result.filament_fraction_grid);
        result.halo_fraction_grid = std::move(grid_result.halo_fraction_grid);
        result.morph_grid_cells = config.grid_cells;
        result.v_void = grid_result.v_void;
        result.v_wall = grid_result.v_wall;
        result.v_filament = grid_result.v_filament;
        result.v_halo = grid_result.v_halo;

        result.grid_time_ms = get_time_ms() - t0;
    }

    // === Optional: PDF Computation ===
    if (config.pdf_n_bins > 0 && !result.particle_density.empty()) {
        t0 = get_time_ms();

        // Compute overdensity: (1 + delta) = density / mean_density
        // IMPORTANT: Use mean of particle densities, not grid mean
        // The grid mean and particle mean can differ significantly
        std::vector<double> overdensity(n_particles);

        // Compute mean from particle densities
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int64_t i = 0; i < n_particles; ++i) {
            sum += result.particle_density[i];
        }
        double mean_rho = sum / static_cast<double>(n_particles);
        result.rho_M = mean_rho;

        #pragma omp parallel for
        for (int64_t i = 0; i < n_particles; ++i) {
            overdensity[i] = result.particle_density[i] / mean_rho;
        }

        // Compute overdensity statistics
        double od_sum = 0.0;
        #pragma omp parallel for reduction(+:od_sum)
        for (int64_t i = 0; i < n_particles; ++i) {
            od_sum += overdensity[i];
        }
        result.overdensity_mean = od_sum / n_particles;

        // Determine bin range
        double range_min = config.pdf_range_min;
        double range_max = config.pdf_range_max;

        if (range_min <= 0 || range_max <= range_min) {
            // Auto-detect range from positive overdensity values only
            double data_min = std::numeric_limits<double>::max();
            double data_max = 0.0;
            #pragma omp parallel for reduction(min:data_min) reduction(max:data_max)
            for (int64_t i = 0; i < n_particles; ++i) {
                if (overdensity[i] > 0) {
                    data_min = std::min(data_min, overdensity[i]);
                    data_max = std::max(data_max, overdensity[i]);
                }
            }
            range_min = std::max(data_min, 1e-3);  // Avoid zero for log bins
            range_max = data_max * 1.001;          // Small padding
        }

        result.pdf_n_bins = config.pdf_n_bins;

        if (config.pdf_jackknife) {
            // Use jackknife conditional histogram
            result.pdf_jackknife_enabled = true;
            result.pdf_jackknife_n_subboxes = config.pdf_jackknife_subboxes *
                                               config.pdf_jackknife_subboxes *
                                               config.pdf_jackknife_subboxes;

            auto jk_result = stats::compute_jackknife_conditional_histogram(
                sorted_ptr,
                overdensity.data(),
                result.morphology.data(),
                n_particles,
                config.box_size,
                config.pdf_n_bins,
                range_min,
                range_max,
                config.pdf_log_bins,
                config.pdf_jackknife_subboxes,
                config.n_threads
            );

            // Copy bin info
            result.pdf_bin_edges = std::move(jk_result.all.bin_edges);
            result.pdf_bin_centers = std::move(jk_result.all.bin_centers);

            // Copy global PDFs
            result.pdf_all = std::move(jk_result.all.global_pdf);
            result.pdf_void = std::move(jk_result.void_class.global_pdf);
            result.pdf_wall = std::move(jk_result.wall_class.global_pdf);
            result.pdf_filament = std::move(jk_result.filament_class.global_pdf);
            result.pdf_halo = std::move(jk_result.halo_class.global_pdf);

            // Copy histogram counts
            result.hist_all = std::move(jk_result.all.global_counts);
            result.hist_void = std::move(jk_result.void_class.global_counts);
            result.hist_wall = std::move(jk_result.wall_class.global_counts);
            result.hist_filament = std::move(jk_result.filament_class.global_counts);
            result.hist_halo = std::move(jk_result.halo_class.global_counts);

            // Copy jackknife errors
            result.pdf_all_error = std::move(jk_result.all.pdf_error);
            result.pdf_void_error = std::move(jk_result.void_class.pdf_error);
            result.pdf_wall_error = std::move(jk_result.wall_class.pdf_error);
            result.pdf_filament_error = std::move(jk_result.filament_class.pdf_error);
            result.pdf_halo_error = std::move(jk_result.halo_class.pdf_error);

        } else {
            // Simple histogram without jackknife
            result.pdf_jackknife_enabled = false;

            // Compute histogram for all particles
            stats::HistogramResult hist_all;
            if (config.pdf_log_bins) {
                hist_all = stats::histogram_log(
                    overdensity.data(), n_particles,
                    config.pdf_n_bins, range_min, range_max,
                    config.n_threads
                );
            } else {
                hist_all = stats::histogram(
                    overdensity.data(), n_particles,
                    config.pdf_n_bins, range_min, range_max,
                    config.n_threads
                );
            }

            // IMPORTANT: Call histogram_to_pdf BEFORE moving bin_edges
            // (histogram_to_pdf needs bin_edges to compute bin widths)
            result.pdf_all = stats::histogram_to_pdf(hist_all);
            result.pdf_bin_edges = std::move(hist_all.bin_edges);
            result.pdf_bin_centers = stats::bin_centers(result.pdf_bin_edges, config.pdf_log_bins);
            result.hist_all = std::move(hist_all.counts);

            // Per-class histograms
            // Extract overdensity values for each class
            // Pre-allocate using known counts for efficiency
            std::vector<double> od_void;
            std::vector<double> od_wall;
            std::vector<double> od_filament;
            std::vector<double> od_halo;
            od_void.reserve(result.n_void);
            od_wall.reserve(result.n_wall);
            od_filament.reserve(result.n_filament);
            od_halo.reserve(result.n_halo);

            for (int64_t i = 0; i < n_particles; ++i) {
                switch (result.morphology[i]) {
                    case 0: od_void.push_back(overdensity[i]); break;
                    case 1: od_wall.push_back(overdensity[i]); break;
                    case 2: od_filament.push_back(overdensity[i]); break;
                    case 3: od_halo.push_back(overdensity[i]); break;
                }
            }

            // Compute per-class histograms using same bins
            auto compute_class_pdf = [&](const std::vector<double>& od_class) {
                if (od_class.empty()) {
                    return std::make_pair(
                        std::vector<double>(config.pdf_n_bins, 0.0),
                        std::vector<int64_t>(config.pdf_n_bins, 0)
                    );
                }
                stats::HistogramResult hist;
                if (config.pdf_log_bins) {
                    hist = stats::histogram_log(
                        od_class.data(), od_class.size(),
                        config.pdf_n_bins, range_min, range_max,
                        config.n_threads
                    );
                } else {
                    hist = stats::histogram(
                        od_class.data(), od_class.size(),
                        config.pdf_n_bins, range_min, range_max,
                        config.n_threads
                    );
                }
                return std::make_pair(
                    stats::histogram_to_pdf(hist),
                    std::move(hist.counts)
                );
            };

            auto [pdf_v, hist_v] = compute_class_pdf(od_void);
            auto [pdf_w, hist_w] = compute_class_pdf(od_wall);
            auto [pdf_f, hist_f] = compute_class_pdf(od_filament);
            auto [pdf_h, hist_h] = compute_class_pdf(od_halo);

            result.pdf_void = std::move(pdf_v);
            result.pdf_wall = std::move(pdf_w);
            result.pdf_filament = std::move(pdf_f);
            result.pdf_halo = std::move(pdf_h);

            result.hist_void = std::move(hist_v);
            result.hist_wall = std::move(hist_w);
            result.hist_filament = std::move(hist_f);
            result.hist_halo = std::move(hist_h);
        }

        // === Per-class overdensity statistics ===
        // Compute mean, stddev, median, and percentiles for each class
        {
            // Gather per-class overdensity vectors
            std::vector<double> od_all_vec(overdensity.begin(), overdensity.end());
            std::vector<double> od_cls[4];  // void, wall, filament, halo
            od_cls[0].reserve(result.n_void);
            od_cls[1].reserve(result.n_wall);
            od_cls[2].reserve(result.n_filament);
            od_cls[3].reserve(result.n_halo);

            for (int64_t i = 0; i < n_particles; ++i) {
                od_cls[result.morphology[i]].push_back(overdensity[i]);
            }

            // Helper to compute stats for a vector
            auto compute_stats = [](std::vector<double>& v, int idx,
                                    OrigamiPipelineResult& res) {
                if (v.empty()) return;
                int64_t n = static_cast<int64_t>(v.size());

                // Mean
                double sum = 0.0;
                for (int64_t i = 0; i < n; ++i) sum += v[i];
                double mean = sum / n;
                res.od_mean[idx] = mean;

                // Stddev
                double sum_sq = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    double d = v[i] - mean;
                    sum_sq += d * d;
                }
                res.od_stddev[idx] = std::sqrt(sum_sq / n);

                // Sort for quantiles
                std::sort(v.begin(), v.end());
                auto percentile = [&v, n](double p) -> double {
                    double idx_f = p * (n - 1);
                    int64_t lo = static_cast<int64_t>(idx_f);
                    int64_t hi = std::min(lo + 1, n - 1);
                    double frac = idx_f - lo;
                    return v[lo] * (1.0 - frac) + v[hi] * frac;
                };

                res.od_median[idx] = percentile(0.50);
                res.od_p5[idx]    = percentile(0.05);
                res.od_p10[idx]   = percentile(0.10);
                res.od_p25[idx]   = percentile(0.25);
                res.od_p75[idx]   = percentile(0.75);
                res.od_p90[idx]   = percentile(0.90);
                res.od_p95[idx]   = percentile(0.95);
                res.od_p99[idx]   = percentile(0.99);
            };

            // Index 0 = all, 1 = void, 2 = wall, 3 = filament, 4 = halo
            compute_stats(od_all_vec, 0, result);
            for (int c = 0; c < 4; ++c) {
                compute_stats(od_cls[c], c + 1, result);
            }
        }

        // === Optional: Volume-weighted PDF from grid cells ===
        if (config.pdf_volume_weighted && !result.density_3d.empty()) {
            int64_t n_cells = static_cast<int64_t>(result.density_3d.size());

            // Compute grid mean (rho_V = mean of density_3d)
            double grid_sum = 0.0;
            #pragma omp parallel for reduction(+:grid_sum)
            for (int64_t i = 0; i < n_cells; ++i) {
                grid_sum += result.density_3d[i];
            }
            double grid_mean = grid_sum / static_cast<double>(n_cells);

            // Compute grid overdensity: (1+delta) = density / grid_mean
            std::vector<double> grid_overdensity(n_cells);
            #pragma omp parallel for
            for (int64_t i = 0; i < n_cells; ++i) {
                grid_overdensity[i] = result.density_3d[i] / grid_mean;
            }

            // Histogram using same bin range as mass-weighted PDF
            stats::HistogramResult hist_vol;
            if (config.pdf_log_bins) {
                hist_vol = stats::histogram_log(
                    grid_overdensity.data(), n_cells,
                    config.pdf_n_bins, range_min, range_max,
                    config.n_threads
                );
            } else {
                hist_vol = stats::histogram(
                    grid_overdensity.data(), n_cells,
                    config.pdf_n_bins, range_min, range_max,
                    config.n_threads
                );
            }

            result.pdf_all_volume_weighted = stats::histogram_to_pdf(hist_vol);
            result.hist_all_volume_weighted = std::move(hist_vol.counts);
        }

        result.pdf_time_ms = get_time_ms() - t0;
    }

    result.total_time_ms = get_time_ms() - start_time;
    return result;
}

// Explicit instantiations
OrigamiPipelineResult run_pipeline(
    double* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config)
{
    return run_pipeline_impl(positions, particle_ids, config);
}

OrigamiPipelineResult run_pipeline(
    float* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config)
{
    return run_pipeline_impl(positions, particle_ids, config);
}

OrigamiPipelineResult run_pipeline_safe(
    const double* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config)
{
    // Create a mutable copy and use that
    int64_t n_particles = static_cast<int64_t>(config.lagrangian_grid_size) *
                          config.lagrangian_grid_size * config.lagrangian_grid_size;

    std::vector<double> positions_copy(positions, positions + n_particles * 3);

    // Use in-place sorting on the copy
    OrigamiPipelineConfig config_copy = config;
    config_copy.sort_in_place = true;

    return run_pipeline_impl(positions_copy.data(), particle_ids, config_copy);
}

} // namespace origami
} // namespace tessera
