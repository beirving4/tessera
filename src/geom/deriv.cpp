#include "geom/deriv.h"
#include <stdexcept>
#include <cmath>

namespace tessera {
namespace geom {

const DerivOptions DERIV_OPTIONS_DEFAULT = {false, DerivOp::None, 4};

// Helper functions for 2nd order derivatives
inline float deriv2_flat(const std::vector<float>& vals, int idx, int di, float denom) {
    return (vals[idx + di] - vals[idx - di]) / denom;
}

inline float deriv2_edge(const std::vector<float>& vals, int idx, int di, int pos, float denom) {
    if (pos == +1) {
        return (-3.0f * vals[idx] + 4.0f * vals[idx + di] - vals[idx + 2*di]) / denom;
    } else { // pos == -1
        return (-3.0f * vals[idx] + 4.0f * vals[idx - di] - vals[idx - 2*di]) / (-denom);
    }
}

inline float deriv2_wrap(const std::vector<float>& vals, int idx, int di, int pos, int width, float denom) {
    int lo, hi;
    if (pos == +1) {
        lo = idx + (width - 1) * di;
        hi = idx + di;
    } else { // pos == -1
        hi = idx - (width - 1) * di;
        lo = idx - di;
    }
    return (vals[hi] - vals[lo]) / denom;
}

// Helper functions for 4th order derivatives
inline float deriv4_flat(const std::vector<float>& vals, int idx, int di, float denom) {
    return (-vals[idx + 2*di] + 8.0f * vals[idx + di] 
            - 8.0f * vals[idx - di] + vals[idx - 2*di]) / denom;
}

inline float deriv4_edge(const std::vector<float>& vals, int idx, int di, int pos, float denom) {
    switch (pos) {
        case +1:
            return (-3.0f * vals[idx + 4*di] + 16.0f * vals[idx + 3*di] 
                    - 36.0f * vals[idx + 2*di] + 48.0f * vals[idx + di] 
                    - 25.0f * vals[idx]) / denom;
        case +2:
            return (-3.0f * vals[idx - di] - 10.0f * vals[idx] 
                    + 18.0f * vals[idx + di] - 6.0f * vals[idx + 2*di] 
                    + vals[idx + 3*di]) / denom;
        case -2:
            return (-3.0f * vals[idx + di] - 10.0f * vals[idx] 
                    + 18.0f * vals[idx - di] - 6.0f * vals[idx - 2*di] 
                    + vals[idx - 3*di]) / (-denom);
        case -1:
            return (-3.0f * vals[idx - 4*di] + 16.0f * vals[idx - 3*di] 
                    - 36.0f * vals[idx - 2*di] + 48.0f * vals[idx - di] 
                    - 25.0f * vals[idx]) / (-denom);
    }
    return 0.0f;
}

inline float deriv4_wrap(const std::vector<float>& vals, int idx, int di, int pos, int width, float denom) {
    int im2, im1, ip1, ip2;
    switch (pos) {
        case +1:
            im2 = idx + (width - 2) * di;
            im1 = idx + (width - 1) * di;
            ip1 = idx + di;
            ip2 = idx + 2 * di;
            break;
        case +2:
            im2 = idx + (width - 2) * di;
            im1 = idx - di;
            ip1 = idx + di;
            ip2 = idx + 2 * di;
            break;
        case -2:
            ip2 = idx - (width - 2) * di;
            ip1 = idx + di;
            im1 = idx - di;
            im2 = idx - 2 * di;
            break;
        case -1:
            ip2 = idx - (width - 2) * di;
            ip1 = idx - (width - 1) * di;
            im1 = idx - di;
            im2 = idx - 2 * di;
            break;
        default:
            return 0.0f;
    }
    return (-vals[ip2] + 8.0f * vals[ip1] - 8.0f * vals[im1] + vals[im2]) / denom;
}

void deriv(const GridLocation& g, const std::vector<float>& vals,
           std::vector<float>& out, int axis, const DerivOptions& opt) {
    if (axis > 2 || axis < 0) {
        throw std::runtime_error("Unrecognized axis");
    }
    if (g.bounds.width[axis] < 3) {
        throw std::runtime_error("Width of array must be at least 3 for derivative");
    }
    if (opt.order != 2 && opt.order != 4) {
        throw std::runtime_error("Grid.deriv() can only compute 2nd and 4th order derivatives");
    }

    std::array<int, 3> l_bounds = {0, 0, 0};
    std::array<int, 3> u_bounds = g.bounds.width;
    l_bounds[axis] += opt.order / 2;
    u_bounds[axis] -= opt.order / 2;
    
    float dx = static_cast<float>(g.box_width / static_cast<double>(g.cells));
    float denom = (opt.order == 2) ? (dx * 2.0f) : (dx * 12.0f);

    int di = 1;
    if (axis == 1) di = g.length;
    else if (axis == 2) di = g.area;

    // Inner region
    for (int x = l_bounds[0]; x < u_bounds[0]; ++x) {
        for (int y = l_bounds[1]; y < u_bounds[1]; ++y) {
            for (int z = l_bounds[2]; z < u_bounds[2]; ++z) {
                int idx = x + y * g.length + z * g.area;
                float d;
                if (opt.order == 2) {
                    d = deriv2_flat(vals, idx, di, denom);
                } else {
                    d = deriv4_flat(vals, idx, di, denom);
                }

                switch (opt.op) {
                    case DerivOp::None: out[idx] = d; break;
                    case DerivOp::Add: out[idx] += d; break;
                    case DerivOp::Subtract: out[idx] -= d; break;
                }
            }
        }
    }

    // Edge/wrap handling
    bool do_wrap = opt.periodic && (g.cells == g.bounds.width[axis]);
    
    for (int pos = -opt.order/2; pos <= opt.order/2; ++pos) {
        if (pos == 0) continue;
        
        int a_idx;
        if (pos > 0) a_idx = pos - 1;
        else a_idx = g.bounds.width[axis] + pos;

        auto process_edge = [&](int x, int y, int z) {
            int idx = x + y * g.length + z * g.area;
            float d;
            
            if (do_wrap) {
                if (opt.order == 2) {
                    d = deriv2_wrap(vals, idx, di, pos, g.bounds.width[axis], denom);
                } else {
                    d = deriv4_wrap(vals, idx, di, pos, g.bounds.width[axis], denom);
                }
            } else {
                if (opt.order == 2) {
                    d = deriv2_edge(vals, idx, di, pos, denom);
                } else {
                    d = deriv4_edge(vals, idx, di, pos, denom);
                }
            }

            switch (opt.op) {
                case DerivOp::None: out[idx] = d; break;
                case DerivOp::Add: out[idx] += d; break;
                case DerivOp::Subtract: out[idx] -= d; break;
            }
        };

        switch (axis) {
            case 0: {
                int x = a_idx;
                for (int z = l_bounds[2]; z < u_bounds[2]; ++z) {
                    for (int y = l_bounds[1]; y < u_bounds[1]; ++y) {
                        process_edge(x, y, z);
                    }
                }
                break;
            }
            case 1: {
                int y = a_idx;
                for (int z = l_bounds[2]; z < u_bounds[2]; ++z) {
                    for (int x = l_bounds[0]; x < u_bounds[0]; ++x) {
                        process_edge(x, y, z);
                    }
                }
                break;
            }
            case 2: {
                int z = a_idx;
                for (int y = l_bounds[1]; y < u_bounds[1]; ++y) {
                    for (int x = l_bounds[0]; x < u_bounds[0]; ++x) {
                        process_edge(x, y, z);
                    }
                }
                break;
            }
        }
    }
}

void gradient(const GridLocation& g, const std::vector<float>& vals,
              std::array<std::vector<float>, 3>& out, const DerivOptions& opt) {
    deriv(g, vals, out[0], 0, opt);
    deriv(g, vals, out[1], 1, opt);
    deriv(g, vals, out[2], 2, opt);
}

void divergence(const GridLocation& g, const std::array<std::vector<float>, 3>& vecs,
                std::vector<float>& out, const DerivOptions& opt) {
    DerivOptions opt_copy = opt;
    deriv(g, vecs[0], out, 0, opt_copy);
    opt_copy.op = DerivOp::Add;
    deriv(g, vecs[1], out, 1, opt_copy);
    deriv(g, vecs[2], out, 2, opt_copy);
}

void curl(const GridLocation& g, const std::array<std::vector<float>, 3>& vecs,
          std::array<std::vector<float>, 3>& out, const DerivOptions& opt) {
    DerivOptions opt_copy = opt;
    
    // out[0] = dv_z/dy - dv_y/dz
    deriv(g, vecs[2], out[0], 1, opt_copy);
    opt_copy.op = DerivOp::Subtract;
    deriv(g, vecs[1], out[0], 2, opt_copy);
    
    // out[1] = dv_x/dz - dv_z/dx
    opt_copy.op = opt.op;
    deriv(g, vecs[0], out[1], 2, opt_copy);
    opt_copy.op = DerivOp::Subtract;
    deriv(g, vecs[2], out[1], 0, opt_copy);
    
    // out[2] = dv_y/dx - dv_x/dy
    opt_copy.op = opt.op;
    deriv(g, vecs[1], out[2], 0, opt_copy);
    opt_copy.op = DerivOp::Subtract;
    deriv(g, vecs[0], out[2], 1, opt_copy);
}

} // namespace geom
} // namespace tessera
