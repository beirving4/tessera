/**
 * @file pipeline.h
 * @brief Unified ORIGAMI pipeline for complete morphology analysis
 *
 * This module provides a single entry point for the complete ORIGAMI workflow:
 * - Particle ID ordering detection
 * - Lagrangian position sorting
 * - Morphology classification
 * - Optional density field computation
 * - Optional grid deposition
 *
 * Designed for memory efficiency with large simulations (N >= 1024^3).
 */

#pragma once

#include "origami/origami.h"
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

namespace tessera {
namespace origami {

/**
 * ID ordering convention for Lagrangian grid mapping.
 *
 * GADGET-4 uses z-major: ID = 1 + z + y*N + x*N^2
 * Some codes use x-major: ID = 1 + x + y*N + z*N^2
 */
enum class IdOrdering : uint8_t {
    XMajor = 0,   ///< ID = 1 + x + y*N + z*N^2 (x varies fastest)
    ZMajor = 1,   ///< ID = 1 + z + y*N + x*N^2 (z varies fastest, GADGET-4 default)
    Auto   = 255  ///< Automatically detect from particle data
};

/**
 * Configuration for the unified ORIGAMI pipeline.
 */
struct OrigamiPipelineConfig {
    // === Required parameters ===
    int lagrangian_grid_size = 0;       ///< Particles per dimension (N for N^3)
    double box_size = 0.0;              ///< Physical box size

    // === ID handling ===
    IdOrdering id_ordering = IdOrdering::Auto;  ///< How particle IDs encode positions
    int64_t id_offset = 1;              ///< Value to subtract from IDs (1 for GADGET-4)

    // === Sorting options ===
    bool sort_in_place = true;          ///< Sort positions in-place (saves ~24GB for N=1024^3)
    bool positions_already_sorted = false;  ///< Skip sorting if already in Lagrangian order

    // === ORIGAMI computation ===
    int n_split = 2;                    ///< Domain decomposition for parallelization
    double linear_regime_threshold = 0.99;  ///< Void fraction threshold for linear regime

    // === Optional: Density computation (set > 0 to enable) ===
    int density_output_cells = 0;       ///< 0 = skip, >0 = compute density at this resolution
    int density_n_samples = 100;        ///< Monte Carlo samples per tetrahedron
    double particle_mass = 1.0;         ///< Particle mass for density normalization
    bool density_periodic = true;       ///< Periodic boundaries for density

    // === Optional: Grid deposition (set > 0 to enable) ===
    int grid_cells = 0;                 ///< 0 = skip, >0 = deposit morphology at this resolution

    // === Optional: Density sampling at particles ===
    bool sample_density_at_particles = true;  ///< Sample density field at particle positions

    // === Threading ===
    int n_threads = 0;                  ///< 0 = auto-detect
    uint64_t seed = 0;                  ///< Random seed for density (0 = time-based)

    // === Validation ===
    bool validate_ids = true;           ///< Validate particle ID range and uniqueness

    /**
     * Default constructor with sensible defaults.
     */
    OrigamiPipelineConfig() = default;

    /**
     * Constructor with required parameters.
     */
    OrigamiPipelineConfig(int grid_size, double box)
        : lagrangian_grid_size(grid_size), box_size(box) {}
};

/**
 * Complete result from the ORIGAMI pipeline.
 */
struct OrigamiPipelineResult {
    // === Core ORIGAMI outputs (always computed) ===
    std::vector<uint8_t> morphology;    ///< Per-particle morphology class (0-3)

    // Particle counts
    int64_t n_void = 0;
    int64_t n_wall = 0;
    int64_t n_filament = 0;
    int64_t n_halo = 0;

    // Mass fractions
    double f_void = 0.0;
    double f_wall = 0.0;
    double f_filament = 0.0;
    double f_halo = 0.0;

    // Linear regime detection
    bool is_linear_regime = false;
    double linear_regime_threshold = 0.99;

    // === ID ordering detection result ===
    IdOrdering detected_ordering = IdOrdering::Auto;

    // === Optional: Density field (if density_output_cells > 0) ===
    std::vector<double> density_3d;     ///< 3D density field [cells^3]
    int density_cells = 0;              ///< Cells per dimension
    double density_cell_width = 0.0;    ///< Physical cell width
    double mean_density = 0.0;          ///< Mean density value

    // === Optional: Per-particle density (if sample_density_at_particles) ===
    std::vector<double> particle_density;

    // === Optional: Grid-based outputs (if grid_cells > 0) ===
    std::vector<uint8_t> morphology_grid;       ///< Dominant class per cell
    std::vector<float> void_fraction_grid;
    std::vector<float> wall_fraction_grid;
    std::vector<float> filament_fraction_grid;
    std::vector<float> halo_fraction_grid;
    int morph_grid_cells = 0;

    // Volume fractions (from grid)
    double v_void = 0.0;
    double v_wall = 0.0;
    double v_filament = 0.0;
    double v_halo = 0.0;

    // === Diagnostics ===
    bool sorting_performed = false;
    double detection_time_ms = 0.0;
    double sorting_time_ms = 0.0;
    double morphology_time_ms = 0.0;
    double density_time_ms = 0.0;
    double sampling_time_ms = 0.0;
    double grid_time_ms = 0.0;
    double total_time_ms = 0.0;
};

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * Detect ID ordering from particle positions and IDs.
 *
 * Uses displacement correlations between neighboring IDs to robustly
 * detect the ordering even when particles have moved significantly
 * and wrapped around periodic boundaries.
 *
 * @param positions Particle positions (N*3 contiguous), any order
 * @param particle_ids Particle IDs (N)
 * @param n_particles Number of particles
 * @param grid_size Lagrangian grid size (N for N^3)
 * @param box_size Simulation box size
 * @return Detected ordering (XMajor or ZMajor)
 */
IdOrdering detect_id_ordering(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size
);

/**
 * Float version of detect_id_ordering.
 */
IdOrdering detect_id_ordering(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    double box_size
);

/**
 * Validate particle IDs for ORIGAMI computation.
 *
 * Checks:
 * - ID range is [id_offset, id_offset + N^3 - 1]
 * - All IDs are unique
 *
 * @param particle_ids Particle IDs (N)
 * @param n_particles Number of particles
 * @param grid_size Lagrangian grid size
 * @param id_offset Value subtracted from IDs
 * @throws std::runtime_error if validation fails
 */
void validate_particle_ids(
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    int64_t id_offset
);

/**
 * Sort particles to x-major Lagrangian order (in-place).
 *
 * Modifies the positions array in-place using cycle decomposition
 * to minimize memory usage. For N=1024^3, this saves ~24GB compared
 * to creating a copy.
 *
 * @param positions Particle positions (N*3) - WILL BE MODIFIED
 * @param particle_ids Particle IDs (N)
 * @param n_particles Number of particles
 * @param grid_size Lagrangian grid size
 * @param ordering ID ordering convention
 * @param id_offset Value to subtract from IDs
 */
void sort_to_lagrangian_inplace(
    double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset = 1
);

/**
 * Float version of sort_to_lagrangian_inplace.
 */
void sort_to_lagrangian_inplace(
    float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset = 1
);

/**
 * Sort particles to x-major Lagrangian order (out-of-place).
 *
 * Creates a new sorted array, leaving input unchanged.
 * Uses more memory but is safer for debugging.
 *
 * @param positions Input positions (N*3) - not modified
 * @param particle_ids Particle IDs (N)
 * @param n_particles Number of particles
 * @param grid_size Lagrangian grid size
 * @param ordering ID ordering convention
 * @param id_offset Value to subtract from IDs
 * @param sorted_output Output array (N*3) - must be pre-allocated
 */
void sort_to_lagrangian(
    const double* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset,
    double* sorted_output
);

/**
 * Float version of sort_to_lagrangian.
 */
void sort_to_lagrangian(
    const float* positions,
    const int64_t* particle_ids,
    int64_t n_particles,
    int grid_size,
    IdOrdering ordering,
    int64_t id_offset,
    float* sorted_output
);

/**
 * Run the complete ORIGAMI analysis pipeline.
 *
 * This is the main entry point for ORIGAMI analysis. It handles:
 * 1. ID ordering detection (if ordering=Auto)
 * 2. ID validation (if validate_ids=true)
 * 3. Sorting to Lagrangian order (in-place if sort_in_place=true)
 * 4. ORIGAMI morphology classification
 * 5. Optional density field computation
 * 6. Optional density sampling at particles
 * 7. Optional grid deposition
 *
 * WARNING: If sort_in_place=true (default), the positions array WILL BE MODIFIED.
 *
 * @param positions Particle positions (N*3) - may be modified if sort_in_place=true
 * @param particle_ids Particle IDs (N)
 * @param config Pipeline configuration
 * @return Complete result with all requested outputs
 */
OrigamiPipelineResult run_pipeline(
    double* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config
);

/**
 * Float version of run_pipeline.
 */
OrigamiPipelineResult run_pipeline(
    float* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config
);

/**
 * Const version of run_pipeline (always copies positions).
 *
 * This version is safe but uses more memory. The positions array
 * will not be modified, but an internal copy will be made for sorting.
 */
OrigamiPipelineResult run_pipeline_safe(
    const double* positions,
    const int64_t* particle_ids,
    const OrigamiPipelineConfig& config
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get string name for IdOrdering enum.
 */
inline const char* id_ordering_name(IdOrdering ordering) {
    switch (ordering) {
        case IdOrdering::XMajor: return "x-major";
        case IdOrdering::ZMajor: return "z-major";
        case IdOrdering::Auto:   return "auto";
        default: return "unknown";
    }
}

/**
 * Parse IdOrdering from string.
 */
inline IdOrdering id_ordering_from_string(const std::string& s) {
    if (s == "x-major" || s == "xmajor" || s == "x") return IdOrdering::XMajor;
    if (s == "z-major" || s == "zmajor" || s == "z") return IdOrdering::ZMajor;
    if (s == "auto") return IdOrdering::Auto;
    return IdOrdering::Auto;
}

} // namespace origami
} // namespace tessera
