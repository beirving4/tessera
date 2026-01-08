#include "density/buffer.h"
#include "geom/deriv.h"
#include <stdexcept>

namespace asymptotic_tetra {
namespace density {

// ScalarBuffer implementation
void ScalarBuffer::slice(int low, int high) {
    slice_start = low;
    slice_end = high;
}

int ScalarBuffer::length() const {
    return slice_end - slice_start;
}

void ScalarBuffer::clear() {
    for (int i = slice_start; i < slice_end; ++i) {
        vals[i] = 0.0;
    }
}

std::vector<float> ScalarBuffer::finalized_scalar_buffer() {
    std::vector<float> result(length());
    for (int i = 0; i < length(); ++i) {
        result[i] = static_cast<float>(vals[slice_start + i]);
    }
    return result;
}

// VectorBuffer implementation
void VectorBuffer::slice(int low, int high) {
    slice_start = low;
    slice_end = high;
}

int VectorBuffer::length() const {
    return slice_end - slice_start;
}

void VectorBuffer::clear() {
    for (int i = slice_start; i < slice_end; ++i) {
        vecs[i] = {0.0, 0.0, 0.0};
    }
}

bool VectorBuffer::set_vectors(const std::vector<geom::Vec3f>& vs) {
    for (size_t i = 0; i < vs.size(); ++i) {
        vecs[i][0] = static_cast<double>(vs[i][0]);
        vecs[i][1] = static_cast<double>(vs[i][1]);
        vecs[i][2] = static_cast<double>(vs[i][2]);
    }
    return true;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
VectorBuffer::finalized_vector_buffer() {
    int len = length();
    std::vector<float> xs(len), ys(len), zs(len);
    for (int i = 0; i < len; ++i) {
        xs[i] = static_cast<float>(vecs[slice_start + i][0]);
        ys[i] = static_cast<float>(vecs[slice_start + i][1]);
        zs[i] = static_cast<float>(vecs[slice_start + i][2]);
    }
    return {xs, ys, zs};
}

// GradientBuffer implementation
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
GradientBuffer::finalized_vector_buffer() {
    if (!grid) {
        throw std::runtime_error("GradientBuffer requires grid location");
    }
    
    std::vector<float> vals32(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals32[i] = static_cast<float>(vals[i]);
    }
    
    std::array<std::vector<float>, 3> out = {
        std::vector<float>(vals.size()),
        std::vector<float>(vals.size()),
        std::vector<float>(vals.size())
    };
    
    geom::DerivOptions opt(true, geom::DerivOp::None, 4);
    geom::gradient(*grid, vals32, out, opt);
    
    return {out[0], out[1], out[2]};
}

// VelocityBuffer implementation
VelocityBuffer::VelocityBuffer(size_t len, size_t wlen)
    : VectorBuffer(len), weights(std::make_unique<VectorBuffer>(wlen)), num(len) {}

void VelocityBuffer::slice(int low, int high) {
    VectorBuffer::slice(low, high);
    // num is sliced along with vecs
}

void VelocityBuffer::clear() {
    VectorBuffer::clear();
    for (size_t i = 0; i < num.size(); ++i) {
        num[i] = 0;
    }
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
VelocityBuffer::finalized_vector_buffer() {
    int len = length();
    std::vector<float> xs(len), ys(len), zs(len);
    for (int i = 0; i < len; ++i) {
        int idx = slice_start + i;
        if (num[idx] == 0) continue;
        float n = static_cast<float>(num[idx]);
        xs[i] = static_cast<float>(vecs[idx][0]) / n;
        ys[i] = static_cast<float>(vecs[idx][1]) / n;
        zs[i] = static_cast<float>(vecs[idx][2]) / n;
    }
    return {xs, ys, zs};
}

// DivergenceBuffer implementation
DivergenceBuffer::DivergenceBuffer(size_t len, size_t wlen, geom::GridLocation* g)
    : VectorBuffer(len), weights(std::make_unique<VectorBuffer>(wlen)), num(len), grid(g) {}

void DivergenceBuffer::slice(int low, int high) {
    VectorBuffer::slice(low, high);
}

void DivergenceBuffer::clear() {
    VectorBuffer::clear();
    for (size_t i = 0; i < num.size(); ++i) {
        num[i] = 0;
    }
}

std::vector<float> DivergenceBuffer::finalized_scalar_buffer() {
    if (!grid) {
        throw std::runtime_error("DivergenceBuffer requires grid location");
    }
    
    std::array<std::vector<float>, 3> vec_comps = {
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size())
    };
    
    for (size_t i = 0; i < vecs.size(); ++i) {
        float n = (num[i] == 0) ? 1.0f : static_cast<float>(num[i]);
        vec_comps[0][i] = static_cast<float>(vecs[i][0]) / n;
        vec_comps[1][i] = static_cast<float>(vecs[i][1]) / n;
        vec_comps[2][i] = static_cast<float>(vecs[i][2]) / n;
    }
    
    std::vector<float> out(vecs.size());
    geom::DerivOptions opt(true, geom::DerivOp::None, 4);
    geom::divergence(*grid, vec_comps, out, opt);
    
    return out;
}

// CurlBuffer implementation
CurlBuffer::CurlBuffer(size_t len, size_t wlen, geom::GridLocation* g)
    : VectorBuffer(len), weights(std::make_unique<VectorBuffer>(wlen)), num(len), grid(g) {}

void CurlBuffer::slice(int low, int high) {
    VectorBuffer::slice(low, high);
}

void CurlBuffer::clear() {
    VectorBuffer::clear();
    for (size_t i = 0; i < num.size(); ++i) {
        num[i] = 0;
    }
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
CurlBuffer::finalized_vector_buffer() {
    if (!grid) {
        throw std::runtime_error("CurlBuffer requires grid location");
    }
    
    std::array<std::vector<float>, 3> vec_comps = {
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size())
    };
    
    for (size_t i = 0; i < vecs.size(); ++i) {
        float n = (num[i] == 0) ? 1.0f : static_cast<float>(num[i]);
        vec_comps[0][i] = static_cast<float>(vecs[i][0]) / n;
        vec_comps[1][i] = static_cast<float>(vecs[i][1]) / n;
        vec_comps[2][i] = static_cast<float>(vecs[i][2]) / n;
    }
    
    std::array<std::vector<float>, 3> out = {
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size()),
        std::vector<float>(vecs.size())
    };
    
    geom::DerivOptions opt(true, geom::DerivOp::None, 4);
    geom::curl(*grid, vec_comps, out, opt);
    
    return {out[0], out[1], out[2]};
}

// Factory functions
std::unique_ptr<Buffer> create_buffer(Quantity q, size_t len, size_t wlen, 
                                      geom::GridLocation* g) {
    switch (q) {
        case Quantity::Density:
            return std::make_unique<DensityBuffer>(len);
        case Quantity::DensityGradient: {
            auto buf = std::make_unique<GradientBuffer>(len);
            buf->grid = g;
            return buf;
        }
        case Quantity::Velocity:
            return std::make_unique<VelocityBuffer>(len, wlen);
        case Quantity::VelocityDivergence:
            return std::make_unique<DivergenceBuffer>(len, wlen, g);
        case Quantity::VelocityCurl:
            return std::make_unique<CurlBuffer>(len, wlen, g);
        default:
            throw std::runtime_error("Unrecognized Quantity");
    }
}

std::unique_ptr<Buffer> create_buffer(Quantity q, size_t len, size_t wlen, 
                                      const geom::GridLocation& g) {
    // Create a static copy to get a stable pointer
    // Note: In a real implementation, the buffer should own the GridLocation
    static std::vector<std::unique_ptr<geom::GridLocation>> grid_storage;
    grid_storage.push_back(std::make_unique<geom::GridLocation>(g));
    return create_buffer(q, len, wlen, grid_storage.back().get());
}

std::unique_ptr<Buffer> wrapper_density_buffer(std::vector<double>& rhos) {
    auto buf = std::make_unique<DensityBuffer>();
    buf->vals = std::move(rhos);
    buf->slice_end = static_cast<int>(buf->vals.size());
    return buf;
}

} // namespace density
} // namespace asymptotic_tetra
