#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include "geom/vec.h"

namespace asymptotic_tetra {
namespace io {

/**
 * GADGET-4 snapshot header from HDF5 attributes.
 */
struct Gadget4Header {
    double box_size = 0.0;
    double time = 0.0;
    double redshift = 0.0;
    std::array<double, 6> mass_table = {0, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> num_part_this_file = {0, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> num_part_total = {0, 0, 0, 0, 0, 0};
    int num_files_per_snapshot = 1;
    
    // Git info for provenance
    std::string git_commit;
    std::string git_date;
    
    /**
     * Get total particle count across all types.
     */
    uint64_t total_particles() const;
    
    /**
     * Get particle count for a specific type.
     */
    uint64_t particles_of_type(int type) const;
};

/**
 * Particle data from a GADGET-4 snapshot.
 * Template to support different particle types.
 */
struct Gadget4Particles {
    std::vector<std::array<float, 3>> coordinates;
    std::vector<std::array<float, 3>> velocities;
    std::vector<uint64_t> particle_ids;
    
    // Optional fields (may be empty depending on simulation)
    std::vector<std::array<float, 3>> accelerations;
    std::vector<float> potential;
    std::vector<float> subfind_density;
    std::vector<float> subfind_hsml;
    std::vector<float> subfind_vel_disp;
    
    /**
     * Number of particles.
     */
    size_t size() const { return particle_ids.size(); }
    
    /**
     * Check if has coordinates.
     */
    bool has_coordinates() const { return !coordinates.empty(); }
    
    /**
     * Check if has velocities.
     */
    bool has_velocities() const { return !velocities.empty(); }
};

/**
 * Complete GADGET-4 snapshot data.
 */
struct Gadget4Snapshot {
    Gadget4Header header;
    
    // Particle data by type (index 0-5)
    // Type 0: Gas, Type 1: DM, Type 2: Disk, Type 3: Bulge, Type 4: Stars, Type 5: BH
    std::array<Gadget4Particles, 6> particles;
    
    /**
     * Get dark matter particles (type 1).
     */
    const Gadget4Particles& dark_matter() const { return particles[1]; }
    Gadget4Particles& dark_matter() { return particles[1]; }
    
    /**
     * Get gas particles (type 0).
     */
    const Gadget4Particles& gas() const { return particles[0]; }
    Gadget4Particles& gas() { return particles[0]; }
};

/**
 * FOF Group (halo) data from GADGET-4 halo catalog.
 */
struct Gadget4Group {
    std::array<float, 3> pos = {0, 0, 0};      // Center of mass position
    std::array<float, 3> vel = {0, 0, 0};      // Center of mass velocity
    float mass = 0.0f;                          // Total mass (FOF)
    int32_t len = 0;                            // Number of particles
    std::array<float, 6> mass_type = {0, 0, 0, 0, 0, 0};  // Mass by particle type
    std::array<int32_t, 6> len_type = {0, 0, 0, 0, 0, 0}; // Particles by type
    
    // Spherical overdensity properties
    float m_crit200 = 0.0f;    // M200 critical
    float r_crit200 = 0.0f;    // R200 critical
    float m_crit500 = 0.0f;    // M500 critical
    float r_crit500 = 0.0f;    // R500 critical
    float m_mean200 = 0.0f;    // M200 mean
    float r_mean200 = 0.0f;    // R200 mean
    float m_tophat200 = 0.0f;  // M_tophat
    float r_tophat200 = 0.0f;  // R_tophat
    
    // Subhalo info
    int64_t first_sub = -1;    // Index of first subhalo
    int32_t n_subs = 0;        // Number of subhalos
    
    float ascale = 0.0f;       // Scale factor when group was identified
};

/**
 * Subhalo data from GADGET-4 halo catalog.
 */
struct Gadget4Subhalo {
    std::array<float, 3> pos = {0, 0, 0};      // Position (most bound particle)
    std::array<float, 3> vel = {0, 0, 0};      // Velocity
    std::array<float, 3> cm = {0, 0, 0};       // Center of mass
    std::array<float, 3> spin = {0, 0, 0};     // Angular momentum
    
    float mass = 0.0f;                          // Total bound mass
    int32_t len = 0;                            // Number of bound particles
    std::array<float, 6> mass_type = {0, 0, 0, 0, 0, 0};
    std::array<int32_t, 6> len_type = {0, 0, 0, 0, 0, 0};
    
    float halfmass_rad = 0.0f;                  // Half-mass radius
    std::array<float, 6> halfmass_rad_type = {0, 0, 0, 0, 0, 0};
    
    float vmax = 0.0f;                          // Maximum circular velocity
    float vmax_rad = 0.0f;                      // Radius of Vmax
    float vel_disp = 0.0f;                      // Velocity dispersion
    
    int64_t group_nr = -1;                      // Parent group index
    int32_t rank_in_gr = 0;                     // Rank within parent group
    int32_t parent_rank = 0;                    // Parent subhalo rank (for subsubhalos)
    uint64_t id_mostbound = 0;                  // ID of most bound particle
};

/**
 * GADGET-4 FOF/Subfind halo catalog header.
 */
struct Gadget4HaloCatalogHeader {
    double box_size = 0.0;
    double time = 0.0;
    double redshift = 0.0;
    
    int64_t n_groups_this_file = 0;
    int64_t n_groups_total = 0;
    int64_t n_subhalos_this_file = 0;
    int64_t n_subhalos_total = 0;
    int64_t n_ids_this_file = 0;
    int64_t n_ids_total = 0;
    int num_files = 1;
    
    std::string git_commit;
    std::string git_date;
};

/**
 * Complete GADGET-4 halo catalog.
 */
struct Gadget4HaloCatalog {
    Gadget4HaloCatalogHeader header;
    std::vector<Gadget4Group> groups;
    std::vector<Gadget4Subhalo> subhalos;
    std::vector<uint64_t> particle_ids;  // IDs of particles in groups
    
    /**
     * Number of groups (halos).
     */
    size_t num_groups() const { return groups.size(); }
    
    /**
     * Number of subhalos.
     */
    size_t num_subhalos() const { return subhalos.size(); }
    
    /**
     * Get subhalos belonging to a specific group.
     */
    std::vector<const Gadget4Subhalo*> subhalos_of_group(size_t group_idx) const;
};

// =============================================================================
// Read functions
// =============================================================================

/**
 * Read GADGET-4 snapshot header only.
 * @param filename Path to HDF5 snapshot file
 * @return Snapshot header
 */
Gadget4Header read_gadget4_header(const std::string& filename);

/**
 * Read complete GADGET-4 snapshot.
 * @param filename Path to HDF5 snapshot file
 * @param particle_types Bitmask of particle types to read (default: all)
 * @param read_velocities Whether to read velocity data
 * @param read_ids Whether to read particle IDs
 * @return Complete snapshot data
 */
Gadget4Snapshot read_gadget4_snapshot(
    const std::string& filename,
    uint32_t particle_types = 0x3F,  // All 6 types
    bool read_velocities = true,
    bool read_ids = true
);

/**
 * Read only positions from a GADGET-4 snapshot for a specific particle type.
 * @param filename Path to HDF5 snapshot file
 * @param particle_type Particle type (0-5)
 * @return Vector of positions
 */
std::vector<std::array<float, 3>> read_gadget4_positions(
    const std::string& filename,
    int particle_type = 1
);

/**
 * Read GADGET-4 halo catalog header only.
 * @param filename Path to HDF5 halo catalog file
 * @return Halo catalog header
 */
Gadget4HaloCatalogHeader read_gadget4_halo_header(const std::string& filename);

/**
 * Read complete GADGET-4 halo catalog.
 * @param filename Path to HDF5 halo catalog file
 * @param read_ids Whether to read particle IDs
 * @return Complete halo catalog
 */
Gadget4HaloCatalog read_gadget4_halo_catalog(
    const std::string& filename,
    bool read_ids = false
);

/**
 * Read only FOF groups from GADGET-4 halo catalog.
 * @param filename Path to HDF5 halo catalog file
 * @return Vector of groups
 */
std::vector<Gadget4Group> read_gadget4_groups(const std::string& filename);

/**
 * Read only subhalos from GADGET-4 halo catalog.
 * @param filename Path to HDF5 halo catalog file
 * @return Vector of subhalos
 */
std::vector<Gadget4Subhalo> read_gadget4_subhalos(const std::string& filename);

} // namespace io
} // namespace asymptotic_tetra
