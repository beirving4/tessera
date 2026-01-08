#pragma once

#include <string>
#include <optional>

namespace asymptotic_tetra {
namespace density {

/**
 * Quantity enumerates the types of physical quantities that can be computed.
 */
enum class Quantity : int64_t {
    Density = 0,
    DensityGradient = 1,
    Velocity = 2,
    VelocityDivergence = 3,
    VelocityCurl = 4,
    EndQuantity = 5
};

/**
 * Convert a Quantity to its string representation.
 */
inline std::string quantity_to_string(Quantity q) {
    switch (q) {
        case Quantity::Density: return "Density";
        case Quantity::DensityGradient: return "DensityGradient";
        case Quantity::Velocity: return "Velocity";
        case Quantity::VelocityDivergence: return "VelocityDivergence";
        case Quantity::VelocityCurl: return "VelocityCurl";
        default: return "Unknown";
    }
}

/**
 * Parse a string to a Quantity.
 */
inline std::optional<Quantity> quantity_from_string(const std::string& str) {
    if (str == "Density") return Quantity::Density;
    if (str == "DensityGradient") return Quantity::DensityGradient;
    if (str == "Velocity") return Quantity::Velocity;
    if (str == "VelocityDivergence") return Quantity::VelocityDivergence;
    if (str == "VelocityCurl") return Quantity::VelocityCurl;
    return std::nullopt;
}

/**
 * Check if a quantity requires velocity data.
 */
inline bool requires_velocity(Quantity q) {
    switch (q) {
        case Quantity::Density:
        case Quantity::DensityGradient:
            return false;
        case Quantity::Velocity:
        case Quantity::VelocityDivergence:
        case Quantity::VelocityCurl:
            return true;
        default:
            return false;
    }
}

/**
 * Check if a quantity can be projected.
 */
inline bool can_project(Quantity q) {
    switch (q) {
        case Quantity::Density:
        case Quantity::Velocity:
            return true;
        case Quantity::DensityGradient:
        case Quantity::VelocityDivergence:
        case Quantity::VelocityCurl:
            return false;
        default:
            return false;
    }
}

/**
 * Check if a quantity is a vector quantity.
 */
inline bool is_vector(Quantity q) {
    return !(q == Quantity::Density || q == Quantity::VelocityDivergence);
}

} // namespace density
} // namespace asymptotic_tetra
