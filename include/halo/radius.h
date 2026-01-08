#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "io/headers.h"
#include "cosmo/cosmo.h"

namespace asymptotic_tetra {
namespace halo {

/**
 * Radius type enum for different halo radius definitions.
 */
enum class RadiusType {
    RVirial,   // Virial radius (Bryan & Norman 1998 overdensity)
    R200c,     // Radius enclosing 200x critical density
    R200m,     // Radius enclosing 200x mean density
    R500c,     // Radius enclosing 500x critical density
    R2500c     // Radius enclosing 2500x critical density
};

/**
 * Parse a radius type from a string.
 * 
 * @param s String like "200m", "r200m", "vir", "rvir", "200c", etc.
 * @return Pair of (RadiusType, success)
 */
inline std::pair<RadiusType, bool> radius_from_string(const std::string& s) {
    std::string lower = s;
    for (auto& c : lower) c = std::tolower(c);
    
    if (lower == "200m" || lower == "r200m") {
        return {RadiusType::R200m, true};
    } else if (lower == "vir" || lower == "rvir") {
        return {RadiusType::RVirial, true};
    } else if (lower == "200c" || lower == "r200c") {
        return {RadiusType::R200c, true};
    } else if (lower == "500c" || lower == "r500c") {
        return {RadiusType::R500c, true};
    } else if (lower == "2500c" || lower == "r2500c") {
        return {RadiusType::R2500c, true};
    }
    return {RadiusType::RVirial, false};
}

/**
 * Convert radius type to string.
 */
inline std::string radius_to_string(RadiusType r) {
    switch (r) {
        case RadiusType::R200m: return "R200m";
        case RadiusType::R200c: return "R200c";
        case RadiusType::R500c: return "R500c";
        case RadiusType::R2500c: return "R2500c";
        case RadiusType::RVirial: return "RVir";
    }
    throw std::runtime_error("Unknown radius type");
}

/**
 * Get the reference density for a given radius type.
 * 
 * @param r Radius type
 * @param cosmo Cosmology header
 * @return Reference density in cosmological units
 */
inline double get_reference_density(RadiusType r, const io::CosmologyHeader& cosmo) {
    double H0 = cosmo.h100 * 100.0;
    
    switch (r) {
        case RadiusType::RVirial:
            // Bryan & Norman 1998 virial overdensity (~178 for flat LCDM at z=0)
            return 177.653 * cosmo::rho_critical(H0, cosmo.omega_m, cosmo.omega_l, cosmo.z);
        case RadiusType::R200c:
            return 200.0 * cosmo::rho_critical(H0, cosmo.omega_m, cosmo.omega_l, cosmo.z);
        case RadiusType::R200m:
            return 200.0 * cosmo::rho_average(H0, cosmo.omega_m, cosmo.omega_l, cosmo.z);
        case RadiusType::R500c:
            return 500.0 * cosmo::rho_critical(H0, cosmo.omega_m, cosmo.omega_l, cosmo.z);
        case RadiusType::R2500c:
            return 2500.0 * cosmo::rho_critical(H0, cosmo.omega_m, cosmo.omega_l, cosmo.z);
    }
    throw std::runtime_error("Unknown radius type");
}

/**
 * Convert masses to radii for a given radius definition.
 * 
 * @param r Radius type
 * @param cosmo Cosmology header
 * @param masses Input masses [M_sun/h]
 * @param radii Output radii [Mpc/h] (must be pre-allocated)
 */
inline void mass_to_radius(RadiusType r, const io::CosmologyHeader& cosmo,
                          const std::vector<double>& masses, std::vector<double>& radii) {
    double rho = get_reference_density(r, cosmo);
    double a = 1.0 / (1.0 + cosmo.z);
    double factor = rho * 4.0 * M_PI / 3.0;
    
    for (size_t i = 0; i < masses.size(); ++i) {
        radii[i] = std::pow(masses[i] / factor, 1.0/3.0) / a;
    }
}

/**
 * Convert radii to masses for a given radius definition.
 * 
 * @param r Radius type
 * @param cosmo Cosmology header
 * @param radii Input radii [Mpc/h]
 * @param masses Output masses [M_sun/h] (must be pre-allocated)
 */
inline void radius_to_mass(RadiusType r, const io::CosmologyHeader& cosmo,
                          const std::vector<double>& radii, std::vector<double>& masses) {
    double rho = get_reference_density(r, cosmo);
    double a = 1.0 / (1.0 + cosmo.z);
    double factor = rho * 4.0 * M_PI / 3.0;
    
    for (size_t i = 0; i < radii.size(); ++i) {
        double r_phys = radii[i] * a;
        masses[i] = factor * r_phys * r_phys * r_phys;
    }
}

/**
 * Get the Rockstar catalog column index for a given radius type.
 * 
 * @param r Radius type
 * @return Column index in Rockstar ASCII output
 */
inline int rockstar_column(RadiusType r) {
    switch (r) {
        case RadiusType::RVirial: return 11;  // Rvir column
        case RadiusType::R200c: return 37;
        case RadiusType::R200m: return 36;
        case RadiusType::R500c: return 38;
        case RadiusType::R2500c: return 39;
    }
    throw std::runtime_error("Unknown radius type");
}

/**
 * Check if Rockstar stores mass (true) or radius (false) for this type.
 * 
 * @param r Radius type
 * @return true if Rockstar stores mass, false if it stores radius
 */
inline bool rockstar_is_mass(RadiusType r) {
    switch (r) {
        case RadiusType::RVirial: return false;  // Rockstar stores Rvir directly
        case RadiusType::R200c:
        case RadiusType::R200m:
        case RadiusType::R500c:
        case RadiusType::R2500c:
            return true;  // Rockstar stores M200c, M200m, etc.
    }
    throw std::runtime_error("Unknown radius type");
}

} // namespace halo
} // namespace asymptotic_tetra
