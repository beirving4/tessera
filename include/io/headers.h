#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "geom/vec.h"
#include "geom/grid.h"

namespace tessera {
namespace io {

/**
 * CosmologyHeader contains information about the simulation's cosmology.
 */
struct CosmologyHeader {
    double z = 0.0;          // Redshift
    double omega_m = 0.0;    // Matter density parameter
    double omega_l = 0.0;    // Dark energy density parameter
    double h100 = 0.0;       // Hubble parameter in units of 100 km/s/Mpc
};

/**
 * CatalogHeader contains metadata about a particle catalog.
 */
struct CatalogHeader {
    CosmologyHeader cosmo;

    double mass = 0.0;           // Mass of one particle
    int64_t count = 0;           // Number of particles in catalog
    int64_t total_count = 0;     // Number of particles in all catalogs
    int64_t count_width = 0;     // Number of particles "on one side"

    int64_t idx = 0;             // Index of catalog
    int64_t grid_width = 0;      // Number of grid cells "on one side"
    double width = 0.0;          // Width of the catalog's bounding box
    double total_width = 0.0;    // Width of the simulation box
};

/**
 * SheetHeader contains metadata for phase sheet files.
 * Binary layout must match the Go version for compatibility.
 */
#pragma pack(push, 1)
struct SheetHeader {
    CosmologyHeader cosmo;
    int64_t count = 0;
    int64_t count_width = 0;
    int64_t segment_width = 0;
    int64_t grid_width = 0;      // = segment_width + 1
    int64_t grid_count = 0;
    int64_t idx = 0;
    int64_t cells = 0;

    double mass = 0.0;
    double total_width = 0.0;

    geom::Vec3f origin;
    geom::Vec3f width;
    geom::Vec3f velocity_origin;
    geom::Vec3f velocity_width;

    /**
     * Get cell bounds for this sheet at given cell resolution.
     */
    geom::CellBounds cell_bounds(int cells) const;
};
#pragma pack(pop)

/**
 * GadgetHeader is the header format used by Gadget-2 simulation outputs.
 */
#pragma pack(push, 1)
struct GadgetHeader {
    uint32_t n_part[6];
    double mass[6];
    double time = 0.0;
    double redshift = 0.0;
    int32_t flag_sfr = 0;
    int32_t flag_feedback = 0;
    uint32_t n_part_total[6];
    int32_t flag_cooling = 0;
    int32_t num_files = 0;
    double box_size = 0.0;
    double omega0 = 0.0;
    double omega_lambda = 0.0;
    double hubble_param = 0.0;
    int32_t flag_stellar_age = 0;
    int32_t hash_tab_size = 0;
    uint8_t padding[88];

    /**
     * Convert to standardized CatalogHeader.
     */
    CatalogHeader standardize() const;

    /**
     * Wrap a distance value into periodic box.
     */
    double wrap_distance(double x) const;
};
#pragma pack(pop)

/**
 * Endianness flag values.
 */
constexpr int32_t ENDIANNESS_LITTLE = 0;
constexpr int32_t ENDIANNESS_BIG = -1;

/**
 * ByteOrder specifies endianness for binary I/O.
 */
enum class ByteOrder {
    Little,
    Big
};

/**
 * Check if the system uses little-endian byte order.
 */
bool is_system_little_endian();

/**
 * Get the system's native byte order.
 */
ByteOrder system_byte_order();

} // namespace io
} // namespace tessera
