#pragma once

/**
 * @file origami.h
 * @brief ORIGAMI morphology classification for cosmic web structure
 *
 * This module implements the ORIGAMI (Order-Reversing, Inverted Gravity,
 * And Multi-streaming Identification) algorithm for classifying particles
 * in cosmological simulations by their collapse dimensionality.
 *
 * The classification is based on shell-crossing detection:
 * - 0 crossings: Void (single-stream, uncollapsed)
 * - 1 crossing:  Wall/Sheet (1D collapse)
 * - 2 crossings: Filament (2D collapse)
 * - 3 crossings: Halo/Knot (3D collapse)
 *
 * Reference: Falck, Neyrinck, & Szalay 2012, ApJ 754, 126
 *
 * Algorithm overview:
 * 1. For each particle, check along Cartesian axes and diagonals
 * 2. Shell-crossing detected when coordinate difference sign != search direction
 * 3. Count unique axis crossings to determine collapse dimensionality
 * 4. Uses prime encoding (2, 3, 5) to track which axes have crossed
 */

#include <vector>
#include <array>
#include <cstdint>

namespace asymptotic_tetra {
namespace origami {

/**
 * Morphology class enumeration.
 */
enum class MorphologyClass : uint8_t {
    Void = 0,      ///< No shell crossings (single-stream)
    Wall = 1,      ///< 1 shell crossing (sheet/wall)
    Filament = 2,  ///< 2 shell crossings (filament)
    Halo = 3       ///< 3 shell crossings (halo/knot)
};

/**
 * Configuration for ORIGAMI morphology computation.
 */
struct OrigamiConfig {
    int lagrangian_grid_size;   ///< Particles per dimension (N for N^3)
    double box_size;            ///< Physical simulation box size
    int n_threads;              ///< Number of OpenMP threads (0 = auto)
    int n_split;                ///< Domain decomposition: split into n_split^3 subcubes
                                ///< Higher values = better parallelization, default: 2

    OrigamiConfig()
        : lagrangian_grid_size(0)
        , box_size(0.0)
        , n_threads(0)
        , n_split(2)
    {}
};

/**
 * Result of ORIGAMI morphology computation.
 */
struct OrigamiResult {
    std::vector<uint8_t> morphology;  ///< Per-particle morphology class (0-3)

    // Particle counts per class
    int64_t n_void;      ///< Number of void particles
    int64_t n_wall;      ///< Number of wall particles
    int64_t n_filament;  ///< Number of filament particles
    int64_t n_halo;      ///< Number of halo particles

    // Mass fractions (assuming equal mass particles)
    double f_void;       ///< Void mass fraction
    double f_wall;       ///< Wall mass fraction
    double f_filament;   ///< Filament mass fraction
    double f_halo;       ///< Halo mass fraction

    // Volume fractions (computed from grid, filled by deposit_to_grid)
    double v_void;       ///< Void volume fraction
    double v_wall;       ///< Wall volume fraction
    double v_filament;   ///< Filament volume fraction
    double v_halo;       ///< Halo volume fraction

    // Per-particle density (optional, filled by sample_density_at_particles)
    std::vector<double> particle_density;

    // Grid-based outputs (optional, filled by deposit_to_grid)
    std::vector<uint8_t> morphology_grid;      ///< Dominant class per cell [z,y,x]
    std::vector<float> void_fraction_grid;     ///< Void fraction per cell
    std::vector<float> wall_fraction_grid;     ///< Wall fraction per cell
    std::vector<float> filament_fraction_grid; ///< Filament fraction per cell
    std::vector<float> halo_fraction_grid;     ///< Halo fraction per cell
    int grid_cells;                             ///< Grid cells per dimension
};

/**
 * Compute ORIGAMI morphology classification for all particles.
 *
 * This implements the exact algorithm from Falck, Neyrinck, & Szalay (2012),
 * including both Cartesian axis checks and diagonal checks for robustness.
 *
 * @param positions Particle positions in Lagrangian order, shape (N^3, 3).
 *                  Use sort_by_lagrangian_id() if particles are not sorted.
 * @param config Configuration parameters
 * @return OrigamiResult containing per-particle morphology and statistics
 */
OrigamiResult compute_morphology(
    const double* positions,
    const OrigamiConfig& config
);

/**
 * Compute morphology (float input version).
 */
OrigamiResult compute_morphology(
    const float* positions,
    const OrigamiConfig& config
);

/**
 * Sample a 3D density field at particle positions.
 *
 * Uses trilinear interpolation to get density values at each particle's
 * Eulerian position. Useful for computing conditional PDFs P(1+delta | class).
 *
 * @param density_grid Flattened 3D density grid, shape (cells^3,), ordered [z,y,x]
 * @param grid_cells Number of cells per dimension
 * @param positions Particle positions, shape (N, 3)
 * @param n_particles Number of particles
 * @param box_size Simulation box size
 * @param particle_density Output array, size n_particles (must be pre-allocated)
 */
void sample_density_at_particles(
    const double* density_grid,
    int grid_cells,
    const double* positions,
    int64_t n_particles,
    double box_size,
    double* particle_density
);

/**
 * Deposit morphology classifications onto a 3D grid.
 *
 * Creates grid-based representations of the morphology field for visualization.
 * Each cell gets the dominant class and per-class fractions based on particles
 * within that cell.
 *
 * @param morphology Per-particle morphology array
 * @param positions Particle positions, shape (N, 3)
 * @param n_particles Number of particles
 * @param box_size Simulation box size
 * @param grid_cells Output grid cells per dimension
 * @param result OrigamiResult to fill with grid outputs
 */
void deposit_morphology_to_grid(
    const uint8_t* morphology,
    const double* positions,
    int64_t n_particles,
    double box_size,
    int grid_cells,
    OrigamiResult& result
);

/**
 * Get string name for a morphology class.
 */
inline const char* morphology_name(MorphologyClass cls) {
    switch (cls) {
        case MorphologyClass::Void: return "Void";
        case MorphologyClass::Wall: return "Wall";
        case MorphologyClass::Filament: return "Filament";
        case MorphologyClass::Halo: return "Halo";
        default: return "Unknown";
    }
}

/**
 * Get string name for a morphology class by value.
 */
inline const char* morphology_name(uint8_t cls) {
    return morphology_name(static_cast<MorphologyClass>(cls));
}

} // namespace origami
} // namespace asymptotic_tetra
