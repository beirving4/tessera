#pragma once

#include <vector>
#include <array>
#include <memory>
#include "quantity.h"
#include "geom/vec.h"
#include "geom/grid.h"

namespace asymptotic_tetra {
namespace density {

/**
 * Buffer is the abstract interface for density field storage.
 */
class Buffer {
public:
    virtual ~Buffer() = default;

    // Array management
    virtual void slice(int low, int high) = 0;
    virtual int length() const = 0;
    virtual void clear() = 0;

    // Getters and setters
    virtual Quantity quantity() const = 0;
    virtual void set_grid_location(geom::GridLocation* g) = 0;
    virtual void set_grid_location(const geom::GridLocation& g) {
        // Default implementation creates a copy for derived classes to use
        // Derived classes can override for proper handling
        (void)g;
    }
    virtual bool set_vectors(const std::vector<geom::Vec3f>& vecs) = 0;

    // Buffer retrieval
    virtual std::vector<int>* count_buffer() = 0;
    virtual std::vector<double>* scalar_buffer() = 0;
    virtual std::vector<std::array<double, 3>>* vector_buffer() = 0;
    virtual std::vector<float> finalized_scalar_buffer() = 0;
    virtual std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() = 0;
    
    virtual bool has_count_buffer() const = 0;
    virtual bool has_scalar_buffer() const = 0;
    virtual bool has_vector_buffer() const = 0;
};

// Null buffer singleton for use as placeholder
class NilBuffer : public Buffer {
public:
    void slice(int, int) override {}
    int length() const override { return 0; }
    void clear() override {}
    Quantity quantity() const override { return Quantity::Density; }
    void set_grid_location(geom::GridLocation*) override {}
    bool set_vectors(const std::vector<geom::Vec3f>&) override { return false; }
    std::vector<int>* count_buffer() override { return nullptr; }
    std::vector<double>* scalar_buffer() override { return nullptr; }
    std::vector<std::array<double, 3>>* vector_buffer() override { return nullptr; }
    std::vector<float> finalized_scalar_buffer() override { return {}; }
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override { return {{}, {}, {}}; }
    bool has_count_buffer() const override { return false; }
    bool has_scalar_buffer() const override { return false; }
    bool has_vector_buffer() const override { return false; }
    
    static NilBuffer& instance() {
        static NilBuffer inst;
        return inst;
    }
};

/**
 * ScalarBuffer stores scalar field values.
 */
class ScalarBuffer : public Buffer {
public:
    std::vector<double> vals;
    int slice_start = 0;
    int slice_end = 0;

    ScalarBuffer() = default;
    explicit ScalarBuffer(size_t len) : vals(len), slice_end(static_cast<int>(len)) {}

    void slice(int low, int high) override;
    int length() const override;
    void clear() override;
    Quantity quantity() const override { return Quantity::Density; }
    void set_grid_location(geom::GridLocation*) override {}
    bool set_vectors(const std::vector<geom::Vec3f>&) override { return false; }
    std::vector<int>* count_buffer() override { return nullptr; }
    std::vector<double>* scalar_buffer() override { return &vals; }
    std::vector<std::array<double, 3>>* vector_buffer() override { return nullptr; }
    std::vector<float> finalized_scalar_buffer() override;
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override { return {{}, {}, {}}; }
    bool has_count_buffer() const override { return false; }
    bool has_scalar_buffer() const override { return true; }
    bool has_vector_buffer() const override { return false; }
};

/**
 * VectorBuffer stores vector field values.
 */
class VectorBuffer : public Buffer {
public:
    std::vector<std::array<double, 3>> vecs;
    int slice_start = 0;
    int slice_end = 0;

    VectorBuffer() = default;
    explicit VectorBuffer(size_t len) : vecs(len), slice_end(static_cast<int>(len)) {}

    void slice(int low, int high) override;
    int length() const override;
    void clear() override;
    Quantity quantity() const override { return Quantity::Velocity; }
    void set_grid_location(geom::GridLocation*) override {}
    bool set_vectors(const std::vector<geom::Vec3f>& vs) override;
    std::vector<int>* count_buffer() override { return nullptr; }
    std::vector<double>* scalar_buffer() override { return nullptr; }
    std::vector<std::array<double, 3>>* vector_buffer() override { return &vecs; }
    std::vector<float> finalized_scalar_buffer() override { return {}; }
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override;
    bool has_count_buffer() const override { return false; }
    bool has_scalar_buffer() const override { return false; }
    bool has_vector_buffer() const override { return true; }
};

/**
 * DensityBuffer is a specialized scalar buffer for density fields.
 */
class DensityBuffer : public ScalarBuffer {
public:
    using ScalarBuffer::ScalarBuffer;
    Quantity quantity() const override { return Quantity::Density; }
};

/**
 * GradientBuffer is for density gradient fields.
 */
class GradientBuffer : public ScalarBuffer {
public:
    geom::GridLocation* grid = nullptr;

    using ScalarBuffer::ScalarBuffer;
    Quantity quantity() const override { return Quantity::DensityGradient; }
    void set_grid_location(geom::GridLocation* g) override { grid = g; }
    std::vector<float> finalized_scalar_buffer() override { return {}; }
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override;
};

/**
 * VelocityBuffer stores velocity fields with counts.
 */
class VelocityBuffer : public VectorBuffer {
public:
    std::unique_ptr<VectorBuffer> weights;
    std::vector<int> num;

    VelocityBuffer() = default;
    VelocityBuffer(size_t len, size_t wlen);

    void slice(int low, int high) override;
    void clear() override;
    Quantity quantity() const override { return Quantity::Velocity; }
    std::vector<int>* count_buffer() override { return &num; }
    bool has_count_buffer() const override { return true; }
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override;
};

/**
 * DivergenceBuffer stores velocity divergence fields.
 */
class DivergenceBuffer : public VectorBuffer {
public:
    std::unique_ptr<VectorBuffer> weights;
    std::vector<int> num;
    geom::GridLocation* grid = nullptr;

    DivergenceBuffer() = default;
    DivergenceBuffer(size_t len, size_t wlen, geom::GridLocation* g);

    void slice(int low, int high) override;
    void clear() override;
    Quantity quantity() const override { return Quantity::VelocityDivergence; }
    void set_grid_location(geom::GridLocation* g) override { grid = g; }
    std::vector<int>* count_buffer() override { return &num; }
    bool has_count_buffer() const override { return true; }
    std::vector<float> finalized_scalar_buffer() override;
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override { return {{}, {}, {}}; }
};

/**
 * CurlBuffer stores velocity curl fields.
 */
class CurlBuffer : public VectorBuffer {
public:
    std::unique_ptr<VectorBuffer> weights;
    std::vector<int> num;
    geom::GridLocation* grid = nullptr;

    CurlBuffer() = default;
    CurlBuffer(size_t len, size_t wlen, geom::GridLocation* g);

    void slice(int low, int high) override;
    void clear() override;
    Quantity quantity() const override { return Quantity::VelocityCurl; }
    void set_grid_location(geom::GridLocation* g) override { grid = g; }
    std::vector<int>* count_buffer() override { return &num; }
    bool has_count_buffer() const override { return true; }
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> 
        finalized_vector_buffer() override;
};

/**
 * Create a buffer for the given quantity type.
 */
std::unique_ptr<Buffer> create_buffer(Quantity q, size_t len, size_t wlen, 
                                      geom::GridLocation* g);

/**
 * Create a buffer for the given quantity type (GridLocation by value).
 */
std::unique_ptr<Buffer> create_buffer(Quantity q, size_t len, size_t wlen, 
                                      const geom::GridLocation& g);

/**
 * Create a wrapper density buffer around existing data.
 */
std::unique_ptr<Buffer> wrapper_density_buffer(std::vector<double>& rhos);

} // namespace density
} // namespace asymptotic_tetra
