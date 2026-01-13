#pragma once

#include <vector>
#include <array>
#include "grid.h"

namespace tessera {
namespace geom {

/**
 * DerivOp specifies how derivative results are combined with output.
 */
enum class DerivOp {
    None,      // Replace output
    Add,       // Add to output
    Subtract   // Subtract from output
};

/**
 * DerivOptions configures derivative computation.
 */
struct DerivOptions {
    bool periodic = false;
    DerivOp op = DerivOp::None;
    int order = 4;  // 2 or 4

    DerivOptions() = default;
    DerivOptions(bool p, DerivOp o, int ord) : periodic(p), op(o), order(ord) {}
};

extern const DerivOptions DERIV_OPTIONS_DEFAULT;

/**
 * Compute the derivative of a scalar field along an axis.
 */
void deriv(const GridLocation& g, const std::vector<float>& vals,
           std::vector<float>& out, int axis, const DerivOptions& opt);

/**
 * Compute the gradient of a scalar field.
 */
void gradient(const GridLocation& g, const std::vector<float>& vals,
              std::array<std::vector<float>, 3>& out, const DerivOptions& opt);

/**
 * Compute the divergence of a vector field.
 */
void divergence(const GridLocation& g, const std::array<std::vector<float>, 3>& vecs,
                std::vector<float>& out, const DerivOptions& opt);

/**
 * Compute the curl of a vector field.
 */
void curl(const GridLocation& g, const std::array<std::vector<float>, 3>& vecs,
          std::array<std::vector<float>, 3>& out, const DerivOptions& opt);

} // namespace geom
} // namespace tessera
