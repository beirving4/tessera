/**
 * @file sph_renderer.cpp
 * @brief SPH density renderer implementation
 *
 * SIMD-optimized SPH rendering with OpenMP parallelization.
 */

#include "density/sph_renderer.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD support detection
#if defined(__AVX2__)
#include <immintrin.h>
#define TESSERA_HAS_AVX2 1
#else
#define TESSERA_HAS_AVX2 0
#endif

namespace tessera {
namespace density {

// =============================================================================
// Constants
// =============================================================================

static constexpr double PI = 3.14159265358979323846;
static constexpr double INV_PI = 1.0 / PI;

// =============================================================================
// Kernel String Conversion
// =============================================================================

const char* sph_kernel_to_string(SPHKernel kernel) {
    switch (kernel) {
        case SPHKernel::CUBIC_SPLINE: return "cubic_spline";
        case SPHKernel::WENDLAND_C2: return "wendland_c2";
        case SPHKernel::WENDLAND_C4: return "wendland_c4";
        case SPHKernel::QUINTIC_SPLINE: return "quintic_spline";
        default: return "unknown";
    }
}

SPHKernel sph_kernel_from_string(const char* str) {
    if (strcmp(str, "cubic_spline") == 0 || strcmp(str, "cubic") == 0) {
        return SPHKernel::CUBIC_SPLINE;
    }
    if (strcmp(str, "wendland_c2") == 0 || strcmp(str, "wendland2") == 0) {
        return SPHKernel::WENDLAND_C2;
    }
    if (strcmp(str, "wendland_c4") == 0 || strcmp(str, "wendland4") == 0) {
        return SPHKernel::WENDLAND_C4;
    }
    if (strcmp(str, "quintic_spline") == 0 || strcmp(str, "quintic") == 0) {
        return SPHKernel::QUINTIC_SPLINE;
    }
    throw std::invalid_argument("Unknown SPH kernel: " + std::string(str));
}

// =============================================================================
// SPHConfig
// =============================================================================

double SPHConfig::compact_support_radius(double h) const {
    switch (kernel) {
        case SPHKernel::CUBIC_SPLINE:
        case SPHKernel::WENDLAND_C2:
        case SPHKernel::WENDLAND_C4:
            return 2.0 * h;
        case SPHKernel::QUINTIC_SPLINE:
            return 3.0 * h;
        default:
            return 2.0 * h;
    }
}

SPHConfig SPHConfig::for_halo(
    const std::array<double, 3>& halo_center,
    double box_width,
    double sim_box,
    int resolution
) {
    SPHConfig cfg;
    cfg.output_cells = resolution;
    cfg.box_size = box_width;
    cfg.center = halo_center;
    cfg.render_width = box_width;
    cfg.sim_box_size = sim_box;
    cfg.periodic = true;
    cfg.kernel = SPHKernel::CUBIC_SPLINE;
    cfg.smoothing_length = 0.0;  // Adaptive
    cfg.neighbors_target = 32;
    return cfg;
}

// =============================================================================
// SPH Kernel Evaluator
// =============================================================================

SPHKernelEvaluator::SPHKernelEvaluator(SPHKernel kernel)
    : kernel_(kernel)
{
    switch (kernel) {
        case SPHKernel::CUBIC_SPLINE:
        case SPHKernel::WENDLAND_C2:
        case SPHKernel::WENDLAND_C4:
            support_radius_ = 2.0;
            break;
        case SPHKernel::QUINTIC_SPLINE:
            support_radius_ = 3.0;
            break;
    }
    compute_normalization();
}

void SPHKernelEvaluator::compute_normalization() {
    // Normalization constants for different kernels
    // These ensure integral over all space = 1
    switch (kernel_) {
        case SPHKernel::CUBIC_SPLINE:
            // M4 cubic spline: compact support at 2h
            norm_1d_ = 2.0 / 3.0;
            norm_2d_ = 10.0 / (7.0 * PI);
            norm_3d_ = 1.0 / PI;
            break;
        case SPHKernel::WENDLAND_C2:
            // Wendland C2: (1-q)^4 * (1+4q), q=r/2h
            norm_1d_ = 5.0 / 4.0;
            norm_2d_ = 7.0 / PI;
            norm_3d_ = 21.0 / (2.0 * PI);
            break;
        case SPHKernel::WENDLAND_C4:
            // Wendland C4: (1-q)^6 * (1+6q+35/3*q^2)
            norm_1d_ = 3.0 / 2.0;
            norm_2d_ = 9.0 / PI;
            norm_3d_ = 495.0 / (32.0 * PI);
            break;
        case SPHKernel::QUINTIC_SPLINE:
            // M6 quintic spline: compact support at 3h
            norm_1d_ = 1.0 / 120.0;
            norm_2d_ = 7.0 / (478.0 * PI);
            norm_3d_ = 1.0 / (120.0 * PI);
            break;
    }
}

double SPHKernelEvaluator::evaluate(double r, double h) const {
    double q = r / h;

    if (q >= support_radius_) {
        return 0.0;
    }

    double h3 = h * h * h;
    double val = 0.0;

    switch (kernel_) {
        case SPHKernel::CUBIC_SPLINE: {
            // M4 cubic spline
            if (q < 1.0) {
                double q2 = q * q;
                double q3 = q2 * q;
                val = 1.0 - 1.5 * q2 + 0.75 * q3;
            } else {
                double t = 2.0 - q;
                val = 0.25 * t * t * t;
            }
            return norm_3d_ * val / h3;
        }

        case SPHKernel::WENDLAND_C2: {
            // Wendland C2: (1-q/2)^4 * (1+2q)
            double u = q / 2.0;  // Map to [0,1]
            if (u >= 1.0) return 0.0;
            double t = 1.0 - u;
            double t2 = t * t;
            double t4 = t2 * t2;
            val = t4 * (1.0 + 4.0 * u);
            return norm_3d_ * val / h3;
        }

        case SPHKernel::WENDLAND_C4: {
            // Wendland C4
            double u = q / 2.0;
            if (u >= 1.0) return 0.0;
            double t = 1.0 - u;
            double t2 = t * t;
            double t6 = t2 * t2 * t2;
            val = t6 * (1.0 + 6.0 * u + 35.0/3.0 * u * u);
            return norm_3d_ * val / h3;
        }

        case SPHKernel::QUINTIC_SPLINE: {
            // M6 quintic spline
            auto quintic = [](double x) {
                return x > 0 ? x * x * x * x * x : 0.0;
            };
            val = quintic(3.0 - q) - 6.0 * quintic(2.0 - q) + 15.0 * quintic(1.0 - q);
            return norm_3d_ * val / h3;
        }
    }

    return 0.0;
}

double SPHKernelEvaluator::evaluate_2d(double r_perp, double h) const {
    // 2D projected kernel (integrated along line of sight)
    // This is the surface density kernel
    double q = r_perp / h;

    if (q >= support_radius_) {
        return 0.0;
    }

    double h2 = h * h;
    double val = 0.0;

    switch (kernel_) {
        case SPHKernel::CUBIC_SPLINE: {
            // Analytic LOS integral of cubic spline
            if (q < 1.0) {
                double q2 = q * q;
                // Integrated form
                val = (1.0 - q2) * (1.0 - q2) * (2.0 + q2) / 3.0;
            } else if (q < 2.0) {
                double t = 2.0 - q;
                val = t * t * t * t / 6.0;
            }
            return norm_2d_ * val / h2;
        }

        case SPHKernel::WENDLAND_C2:
        case SPHKernel::WENDLAND_C4: {
            // Use numerical integration for Wendland kernels
            // Simple midpoint rule with 20 intervals
            double sum = 0.0;
            double z_max = h * support_radius_;
            double r2 = r_perp * r_perp;
            int n_steps = 20;
            double dz = 2.0 * z_max / n_steps;

            for (int i = 0; i < n_steps; ++i) {
                double z = -z_max + (i + 0.5) * dz;
                double r3d = std::sqrt(r2 + z * z);
                sum += evaluate(r3d, h);
            }
            return sum * dz;
        }

        case SPHKernel::QUINTIC_SPLINE: {
            // Numerical integration
            double sum = 0.0;
            double z_max = h * support_radius_;
            double r2 = r_perp * r_perp;
            int n_steps = 30;
            double dz = 2.0 * z_max / n_steps;

            for (int i = 0; i < n_steps; ++i) {
                double z = -z_max + (i + 0.5) * dz;
                double r3d = std::sqrt(r2 + z * z);
                sum += evaluate(r3d, h);
            }
            return sum * dz;
        }
    }

    return 0.0;
}

void SPHKernelEvaluator::evaluate_batch(const float* r, float h, float* out) const {
#if TESSERA_HAS_AVX2
    // SIMD evaluation for 8 distances at once
    __m256 vq = _mm256_loadu_ps(r);
    __m256 vh = _mm256_set1_ps(h);
    __m256 vq_scaled = _mm256_div_ps(vq, vh);

    __m256 vsupport = _mm256_set1_ps(static_cast<float>(support_radius_));
    __m256 vmask = _mm256_cmp_ps(vq_scaled, vsupport, _CMP_LT_OQ);

    // Compute kernel based on type
    __m256 vresult;

    switch (kernel_) {
        case SPHKernel::CUBIC_SPLINE: {
            // Simplified cubic spline for batch (inner region only for speed)
            __m256 vone = _mm256_set1_ps(1.0f);
            __m256 vq2 = _mm256_mul_ps(vq_scaled, vq_scaled);
            __m256 vq3 = _mm256_mul_ps(vq2, vq_scaled);

            // W = 1 - 1.5*q^2 + 0.75*q^3 for q < 1
            __m256 v1p5 = _mm256_set1_ps(1.5f);
            __m256 v0p75 = _mm256_set1_ps(0.75f);

            vresult = _mm256_sub_ps(vone, _mm256_mul_ps(v1p5, vq2));
            vresult = _mm256_add_ps(vresult, _mm256_mul_ps(v0p75, vq3));

            // For q >= 1, use outer formula: 0.25 * (2-q)^3
            __m256 vtwo = _mm256_set1_ps(2.0f);
            __m256 vt = _mm256_sub_ps(vtwo, vq_scaled);
            __m256 vt3 = _mm256_mul_ps(_mm256_mul_ps(vt, vt), vt);
            __m256 vouter = _mm256_mul_ps(_mm256_set1_ps(0.25f), vt3);

            // Select inner or outer based on q < 1
            __m256 vinner_mask = _mm256_cmp_ps(vq_scaled, vone, _CMP_LT_OQ);
            vresult = _mm256_blendv_ps(vouter, vresult, vinner_mask);
            break;
        }

        case SPHKernel::WENDLAND_C2: {
            // Wendland C2: (1-u)^4 * (1+4u) where u = q/2
            __m256 vhalf = _mm256_set1_ps(0.5f);
            __m256 vu = _mm256_mul_ps(vq_scaled, vhalf);
            __m256 vone = _mm256_set1_ps(1.0f);
            __m256 vfour = _mm256_set1_ps(4.0f);

            __m256 vt = _mm256_sub_ps(vone, vu);
            __m256 vt2 = _mm256_mul_ps(vt, vt);
            __m256 vt4 = _mm256_mul_ps(vt2, vt2);
            __m256 vfactor = _mm256_add_ps(vone, _mm256_mul_ps(vfour, vu));

            vresult = _mm256_mul_ps(vt4, vfactor);
            break;
        }

        default:
            // Fallback: compute one by one
            for (int i = 0; i < 8; ++i) {
                out[i] = static_cast<float>(evaluate(r[i], h));
            }
            return;
    }

    // Apply support mask (zero outside compact support)
    vresult = _mm256_and_ps(vresult, vmask);

    // Normalize
    float h3 = h * h * h;
    __m256 vnorm = _mm256_set1_ps(static_cast<float>(norm_3d_ / h3));
    vresult = _mm256_mul_ps(vresult, vnorm);

    _mm256_storeu_ps(out, vresult);

#else
    // Scalar fallback
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<float>(evaluate(r[i], h));
    }
#endif
}

// =============================================================================
// Spatial Hash Grid
// =============================================================================

SpatialHashGrid::SpatialHashGrid(
    const SPHParticles& particles,
    double cell_size,
    double box_size,
    bool periodic
)
    : cell_size_(cell_size)
    , box_size_(box_size)
    , periodic_(periodic)
    , n_particles_(static_cast<int64_t>(particles.size()))
    , particles_(&particles)
    , offset_(box_size / 2.0)  // Offset to handle centered coordinates
{
    if (cell_size <= 0) {
        throw std::invalid_argument("cell_size must be positive");
    }

    grid_dim_ = static_cast<int>(std::ceil(box_size / cell_size));
    if (grid_dim_ < 1) grid_dim_ = 1;

    // Allocate cells
    int64_t n_cells = static_cast<int64_t>(grid_dim_) * grid_dim_ * grid_dim_;
    cells_.resize(n_cells);

    // Insert particles into cells
    // Particles are in centered coordinates [-box_size/2, +box_size/2]
    // Shift to [0, box_size] for grid indexing
    for (size_t i = 0; i < particles.size(); ++i) {
        float px = particles.x[i] + static_cast<float>(offset_);
        float py = particles.y[i] + static_cast<float>(offset_);
        float pz = particles.z[i] + static_cast<float>(offset_);

        // Handle periodicity
        if (periodic_) {
            while (px < 0) px += static_cast<float>(box_size);
            while (px >= box_size) px -= static_cast<float>(box_size);
            while (py < 0) py += static_cast<float>(box_size);
            while (py >= box_size) py -= static_cast<float>(box_size);
            while (pz < 0) pz += static_cast<float>(box_size);
            while (pz >= box_size) pz -= static_cast<float>(box_size);
        }

        int ix = static_cast<int>(px / cell_size_);
        int iy = static_cast<int>(py / cell_size_);
        int iz = static_cast<int>(pz / cell_size_);

        ix = std::max(0, std::min(grid_dim_ - 1, ix));
        iy = std::max(0, std::min(grid_dim_ - 1, iy));
        iz = std::max(0, std::min(grid_dim_ - 1, iz));

        cells_[cell_index(ix, iy, iz)].push_back(static_cast<int32_t>(i));
    }
}

int64_t SpatialHashGrid::cell_index(int ix, int iy, int iz) const {
    return static_cast<int64_t>(iz) * grid_dim_ * grid_dim_ +
           static_cast<int64_t>(iy) * grid_dim_ +
           static_cast<int64_t>(ix);
}

void SpatialHashGrid::wrap_cell(int& ix, int& iy, int& iz) const {
    if (periodic_) {
        while (ix < 0) ix += grid_dim_;
        while (ix >= grid_dim_) ix -= grid_dim_;
        while (iy < 0) iy += grid_dim_;
        while (iy >= grid_dim_) iy -= grid_dim_;
        while (iz < 0) iz += grid_dim_;
        while (iz >= grid_dim_) iz -= grid_dim_;
    } else {
        ix = std::max(0, std::min(grid_dim_ - 1, ix));
        iy = std::max(0, std::min(grid_dim_ - 1, iy));
        iz = std::max(0, std::min(grid_dim_ - 1, iz));
    }
}

void SpatialHashGrid::find_neighbors(
    double x, double y, double z,
    double radius,
    std::vector<int32_t>& indices
) const {
    indices.clear();

    int n_cells_radius = static_cast<int>(std::ceil(radius / cell_size_)) + 1;

    // Apply offset to convert centered coordinates to grid coordinates
    int cx = static_cast<int>((x + offset_) / cell_size_);
    int cy = static_cast<int>((y + offset_) / cell_size_);
    int cz = static_cast<int>((z + offset_) / cell_size_);

    double radius_sq = radius * radius;

    for (int dz = -n_cells_radius; dz <= n_cells_radius; ++dz) {
        for (int dy = -n_cells_radius; dy <= n_cells_radius; ++dy) {
            for (int dx = -n_cells_radius; dx <= n_cells_radius; ++dx) {
                int ix = cx + dx;
                int iy = cy + dy;
                int iz = cz + dz;

                if (!periodic_) {
                    if (ix < 0 || ix >= grid_dim_ ||
                        iy < 0 || iy >= grid_dim_ ||
                        iz < 0 || iz >= grid_dim_) {
                        continue;
                    }
                } else {
                    wrap_cell(ix, iy, iz);
                }

                const auto& cell = cells_[cell_index(ix, iy, iz)];
                for (int32_t pidx : cell) {
                    double px = particles_->x[pidx];
                    double py = particles_->y[pidx];
                    double pz = particles_->z[pidx];

                    // Periodic distance
                    double ddx = px - x;
                    double ddy = py - y;
                    double ddz = pz - z;

                    if (periodic_) {
                        if (ddx > box_size_ / 2) ddx -= box_size_;
                        if (ddx < -box_size_ / 2) ddx += box_size_;
                        if (ddy > box_size_ / 2) ddy -= box_size_;
                        if (ddy < -box_size_ / 2) ddy += box_size_;
                        if (ddz > box_size_ / 2) ddz -= box_size_;
                        if (ddz < -box_size_ / 2) ddz += box_size_;
                    }

                    double dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    if (dist_sq < radius_sq) {
                        indices.push_back(pidx);
                    }
                }
            }
        }
    }
}

void SpatialHashGrid::find_neighbors_2d(
    double x, double y,
    double radius,
    int axis,
    std::vector<int32_t>& indices
) const {
    indices.clear();

    int n_cells_radius = static_cast<int>(std::ceil(radius / cell_size_)) + 1;

    // Apply offset to convert centered coordinates to grid coordinates
    double x_shifted = x + offset_;
    double y_shifted = y + offset_;

    // Map 2D query to 3D grid
    int cx, cy, cz;
    if (axis == 0) {  // Project along x
        cx = 0;
        cy = static_cast<int>(x_shifted / cell_size_);
        cz = static_cast<int>(y_shifted / cell_size_);
    } else if (axis == 1) {  // Project along y
        cx = static_cast<int>(x_shifted / cell_size_);
        cy = 0;
        cz = static_cast<int>(y_shifted / cell_size_);
    } else {  // Project along z
        cx = static_cast<int>(x_shifted / cell_size_);
        cy = static_cast<int>(y_shifted / cell_size_);
        cz = 0;
    }

    double radius_sq = radius * radius;

    // Search all cells along projection axis
    for (int da = 0; da < grid_dim_; ++da) {
        for (int d1 = -n_cells_radius; d1 <= n_cells_radius; ++d1) {
            for (int d2 = -n_cells_radius; d2 <= n_cells_radius; ++d2) {
                int ix, iy, iz;
                if (axis == 0) {
                    ix = da;
                    iy = cy + d1;
                    iz = cz + d2;
                } else if (axis == 1) {
                    ix = cx + d1;
                    iy = da;
                    iz = cz + d2;
                } else {
                    ix = cx + d1;
                    iy = cy + d2;
                    iz = da;
                }

                if (!periodic_) {
                    if (ix < 0 || ix >= grid_dim_ ||
                        iy < 0 || iy >= grid_dim_ ||
                        iz < 0 || iz >= grid_dim_) {
                        continue;
                    }
                } else {
                    wrap_cell(ix, iy, iz);
                }

                const auto& cell = cells_[cell_index(ix, iy, iz)];
                for (int32_t pidx : cell) {
                    double px, py;
                    if (axis == 0) {
                        px = particles_->y[pidx];
                        py = particles_->z[pidx];
                    } else if (axis == 1) {
                        px = particles_->x[pidx];
                        py = particles_->z[pidx];
                    } else {
                        px = particles_->x[pidx];
                        py = particles_->y[pidx];
                    }

                    double ddx = px - x;
                    double ddy = py - y;

                    if (periodic_) {
                        if (ddx > box_size_ / 2) ddx -= box_size_;
                        if (ddx < -box_size_ / 2) ddx += box_size_;
                        if (ddy > box_size_ / 2) ddy -= box_size_;
                        if (ddy < -box_size_ / 2) ddy += box_size_;
                    }

                    double dist_sq = ddx * ddx + ddy * ddy;
                    if (dist_sq < radius_sq) {
                        indices.push_back(pidx);
                    }
                }
            }
        }
    }
}

// =============================================================================
// SPH Renderer
// =============================================================================

SPHRenderer::SPHRenderer(const SPHConfig& config)
    : config_(config)
    , kernel_(config.kernel)
{
    if (config_.render_width <= 0) {
        config_.render_width = config_.box_size;
    }
}

void SPHRenderer::transform_to_render_coords(
    const SPHParticles& input,
    SPHParticles& output,
    double& render_width
) const {
    output.clear();
    output.reserve(input.size());

    double half_width = config_.render_width / 2.0;

    // For adaptive smoothing (smoothing_length=0), use a generous estimate
    // The smoothing length will be computed later based on particle density,
    // but we need to include enough particles in the search region first.
    // Use render_width/4 as a conservative estimate for h, giving a search
    // buffer of ~0.5*render_width which should capture particles whose kernels
    // extend into the render region.
    double h_estimate = config_.smoothing_length > 0
        ? config_.smoothing_length
        : config_.render_width / 4.0;
    double search_radius = half_width + kernel_.support_radius() * h_estimate;

    for (size_t i = 0; i < input.size(); ++i) {
        double dx = input.x[i] - config_.center[0];
        double dy = input.y[i] - config_.center[1];
        double dz = input.z[i] - config_.center[2];

        // Periodic wrap
        if (config_.periodic && config_.sim_box_size > 0) {
            double half_box = config_.sim_box_size / 2.0;
            if (dx > half_box) dx -= config_.sim_box_size;
            if (dx < -half_box) dx += config_.sim_box_size;
            if (dy > half_box) dy -= config_.sim_box_size;
            if (dy < -half_box) dy += config_.sim_box_size;
            if (dz > half_box) dz -= config_.sim_box_size;
            if (dz < -half_box) dz += config_.sim_box_size;
        }

        // Check if within search radius
        if (std::abs(dx) <= search_radius &&
            std::abs(dy) <= search_radius &&
            std::abs(dz) <= search_radius) {
            output.x.push_back(static_cast<float>(dx));
            output.y.push_back(static_cast<float>(dy));
            output.z.push_back(static_cast<float>(dz));

            if (!input.mass.empty()) {
                output.mass.push_back(input.mass[i]);
            }
            if (!input.h.empty()) {
                output.h.push_back(input.h[i]);
            }
        }
    }

    render_width = config_.render_width;
}

void SPHRenderer::compute_smoothing_lengths(SPHParticles& particles) {
    if (particles.empty()) return;

    // Estimate smoothing length from local density
    double n_dens = static_cast<double>(particles.size()) /
                   (config_.render_width * config_.render_width * config_.render_width);
    double mean_spacing = std::pow(1.0 / n_dens, 1.0/3.0);
    double h_estimate = mean_spacing * std::pow(config_.neighbors_target / (4.0 * PI / 3.0), 1.0/3.0);

    particles.h.resize(particles.size(), static_cast<float>(h_estimate));

    // Apply limits
    if (config_.h_min > 0) {
        for (auto& h : particles.h) {
            h = std::max(h, static_cast<float>(config_.h_min));
        }
    }
    if (config_.h_max > 0) {
        for (auto& h : particles.h) {
            h = std::min(h, static_cast<float>(config_.h_max));
        }
    }
}

SPHResult2D SPHRenderer::render_2d(const SPHParticles& particles) {
    auto start_time = std::chrono::high_resolution_clock::now();

    SPHResult2D result;
    result.cells = config_.output_cells;
    result.projection_axis = config_.projection_axis;
    result.n_particles = static_cast<int64_t>(particles.size());

    if (particles.empty()) {
        result.density.resize(result.cells * result.cells, 0.0);
        result.counts.resize(result.cells * result.cells, 0);
        return result;
    }

    // Transform to render coordinates
    SPHParticles local_particles;
    double render_width;
    transform_to_render_coords(particles, local_particles, render_width);

    if (local_particles.empty()) {
        result.density.resize(result.cells * result.cells, 0.0);
        result.counts.resize(result.cells * result.cells, 0);
        return result;
    }

    // Compute smoothing lengths if not provided
    if (local_particles.h.empty()) {
        if (config_.smoothing_length > 0) {
            local_particles.h.resize(local_particles.size(),
                static_cast<float>(config_.smoothing_length));
        } else {
            compute_smoothing_lengths(local_particles);
        }
    }

    result.cell_width = render_width / result.cells;

    // Set extent
    double half_width = render_width / 2.0;
    result.extent_x = {-half_width, half_width};
    result.extent_y = {-half_width, half_width};

    // Build spatial grid
    double max_h = *std::max_element(local_particles.h.begin(), local_particles.h.end());
    double grid_cell_size = max_h * kernel_.support_radius();
    SpatialHashGrid grid(local_particles, grid_cell_size, render_width, false);

    // Allocate output
    int64_t n_pixels = static_cast<int64_t>(result.cells) * result.cells;
    result.density.resize(n_pixels, 0.0);
    result.counts.resize(n_pixels, 0);

    int n_threads = config_.n_threads;
    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif
    }

    std::vector<int64_t> thread_kernel_evals(n_threads, 0);

    // Render with OpenMP
    #pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::vector<int32_t> neighbors;
        neighbors.reserve(256);

        int64_t local_evals = 0;

        #pragma omp for schedule(dynamic, 16)
        for (int iy = 0; iy < result.cells; ++iy) {
            double py = -half_width + (iy + 0.5) * result.cell_width;

            for (int ix = 0; ix < result.cells; ++ix) {
                double px = -half_width + (ix + 0.5) * result.cell_width;

                // Find particles that could contribute
                grid.find_neighbors_2d(px, py, grid_cell_size, config_.projection_axis, neighbors);

                double cell_density = 0.0;
                int64_t cell_count = 0;

                for (int32_t pidx : neighbors) {
                    float part_x, part_y;
                    if (config_.projection_axis == 0) {
                        part_x = local_particles.y[pidx];
                        part_y = local_particles.z[pidx];
                    } else if (config_.projection_axis == 1) {
                        part_x = local_particles.x[pidx];
                        part_y = local_particles.z[pidx];
                    } else {
                        part_x = local_particles.x[pidx];
                        part_y = local_particles.y[pidx];
                    }

                    double ddx = part_x - px;
                    double ddy = part_y - py;
                    double r_perp = std::sqrt(ddx * ddx + ddy * ddy);

                    float h = local_particles.h[pidx];
                    if (r_perp < h * kernel_.support_radius()) {
                        double w = kernel_.evaluate_2d(r_perp, h);
                        double mass = local_particles.mass.empty() ?
                            config_.particle_mass : local_particles.mass[pidx];
                        cell_density += mass * w;
                        ++cell_count;
                        ++local_evals;
                    }
                }

                int64_t idx = static_cast<int64_t>(iy) * result.cells + ix;
                result.density[idx] = cell_density;
                result.counts[idx] = cell_count;
            }
        }

        thread_kernel_evals[tid] = local_evals;
    }

    // Sum kernel evaluations
    for (int64_t evals : thread_kernel_evals) {
        result.kernel_evals += evals;
    }

    // Compute statistics
    result.total_mass = 0.0;
    for (double d : result.density) {
        result.total_mass += d;
    }
    result.total_mass *= result.cell_width * result.cell_width;
    result.mean_density = result.total_mass / (render_width * render_width);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    return result;
}

SPHResult2D SPHRenderer::render_2d(
    const float* x,
    const float* y,
    const float* z,
    int64_t n_particles,
    const float* masses
) {
    SPHParticles particles;
    particles.x.assign(x, x + n_particles);
    particles.y.assign(y, y + n_particles);
    particles.z.assign(z, z + n_particles);
    if (masses) {
        particles.mass.assign(masses, masses + n_particles);
    }
    return render_2d(particles);
}

SPHResult3D SPHRenderer::render_3d(const SPHParticles& particles) {
    auto start_time = std::chrono::high_resolution_clock::now();

    SPHResult3D result;
    result.cells = config_.output_cells;
    result.n_particles = static_cast<int64_t>(particles.size());

    if (particles.empty()) {
        int64_t n = static_cast<int64_t>(result.cells) * result.cells * result.cells;
        result.density.resize(n, 0.0);
        result.counts.resize(n, 0);
        return result;
    }

    // Transform to render coordinates
    SPHParticles local_particles;
    double render_width;
    transform_to_render_coords(particles, local_particles, render_width);

    if (local_particles.empty()) {
        int64_t n = static_cast<int64_t>(result.cells) * result.cells * result.cells;
        result.density.resize(n, 0.0);
        result.counts.resize(n, 0);
        return result;
    }

    // Compute smoothing lengths if not provided
    if (local_particles.h.empty()) {
        if (config_.smoothing_length > 0) {
            local_particles.h.resize(local_particles.size(),
                static_cast<float>(config_.smoothing_length));
        } else {
            compute_smoothing_lengths(local_particles);
        }
    }

    result.cell_width = render_width / result.cells;

    // Build spatial grid
    double max_h = *std::max_element(local_particles.h.begin(), local_particles.h.end());
    double grid_cell_size = max_h * kernel_.support_radius();
    SpatialHashGrid grid(local_particles, grid_cell_size, render_width, false);

    // Allocate output
    int64_t n_voxels = static_cast<int64_t>(result.cells) * result.cells * result.cells;
    result.density.resize(n_voxels, 0.0);
    result.counts.resize(n_voxels, 0);

    double half_width = render_width / 2.0;

    int n_threads = config_.n_threads;
    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif
    }

    std::vector<int64_t> thread_kernel_evals(n_threads, 0);

    #pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::vector<int32_t> neighbors;
        neighbors.reserve(256);

        int64_t local_evals = 0;

        #pragma omp for schedule(dynamic, 4)
        for (int iz = 0; iz < result.cells; ++iz) {
            double pz = -half_width + (iz + 0.5) * result.cell_width;

            for (int iy = 0; iy < result.cells; ++iy) {
                double py = -half_width + (iy + 0.5) * result.cell_width;

                for (int ix = 0; ix < result.cells; ++ix) {
                    double px = -half_width + (ix + 0.5) * result.cell_width;

                    grid.find_neighbors(px, py, pz, grid_cell_size, neighbors);

                    double cell_density = 0.0;
                    int64_t cell_count = 0;

                    for (int32_t pidx : neighbors) {
                        double ddx = local_particles.x[pidx] - px;
                        double ddy = local_particles.y[pidx] - py;
                        double ddz = local_particles.z[pidx] - pz;
                        double r = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

                        float h = local_particles.h[pidx];
                        if (r < h * kernel_.support_radius()) {
                            double w = kernel_.evaluate(r, h);
                            double mass = local_particles.mass.empty() ?
                                config_.particle_mass : local_particles.mass[pidx];
                            cell_density += mass * w;
                            ++cell_count;
                            ++local_evals;
                        }
                    }

                    int64_t idx = static_cast<int64_t>(iz) * result.cells * result.cells +
                                  static_cast<int64_t>(iy) * result.cells + ix;
                    result.density[idx] = cell_density;
                    result.counts[idx] = cell_count;
                }
            }
        }

        thread_kernel_evals[tid] = local_evals;
    }

    for (int64_t evals : thread_kernel_evals) {
        result.kernel_evals += evals;
    }

    // Compute statistics
    result.total_mass = 0.0;
    for (double d : result.density) {
        result.total_mass += d;
    }
    double cell_vol = result.cell_width * result.cell_width * result.cell_width;
    result.total_mass *= cell_vol;
    result.mean_density = result.total_mass / (render_width * render_width * render_width);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    return result;
}

// =============================================================================
// Convenience Functions
// =============================================================================

SPHResult2D render_sph_density_2d(
    const float* positions,
    int64_t n_particles,
    const std::array<double, 3>& center,
    double box_width,
    int output_cells,
    double sim_box_size,
    int projection_axis,
    SPHKernel kernel,
    double smoothing_length,
    int n_threads
) {
    SPHConfig config;
    config.output_cells = output_cells;
    config.box_size = box_width;
    config.center = center;
    config.render_width = box_width;
    config.sim_box_size = sim_box_size;
    config.kernel = kernel;
    config.smoothing_length = smoothing_length;
    config.projection_axis = projection_axis;
    config.n_threads = n_threads;
    config.periodic = true;

    SPHParticles particles;
    particles.x.resize(n_particles);
    particles.y.resize(n_particles);
    particles.z.resize(n_particles);

    for (int64_t i = 0; i < n_particles; ++i) {
        particles.x[i] = positions[i * 3 + 0];
        particles.y[i] = positions[i * 3 + 1];
        particles.z[i] = positions[i * 3 + 2];
    }

    SPHRenderer renderer(config);
    return renderer.render_2d(particles);
}

SPHResult2D render_sph_density_2d(
    const double* positions,
    int64_t n_particles,
    const std::array<double, 3>& center,
    double box_width,
    int output_cells,
    double sim_box_size,
    int projection_axis,
    SPHKernel kernel,
    double smoothing_length,
    int n_threads
) {
    SPHConfig config;
    config.output_cells = output_cells;
    config.box_size = box_width;
    config.center = center;
    config.render_width = box_width;
    config.sim_box_size = sim_box_size;
    config.kernel = kernel;
    config.smoothing_length = smoothing_length;
    config.projection_axis = projection_axis;
    config.n_threads = n_threads;
    config.periodic = true;

    SPHParticles particles;
    particles.x.resize(n_particles);
    particles.y.resize(n_particles);
    particles.z.resize(n_particles);

    for (int64_t i = 0; i < n_particles; ++i) {
        particles.x[i] = static_cast<float>(positions[i * 3 + 0]);
        particles.y[i] = static_cast<float>(positions[i * 3 + 1]);
        particles.z[i] = static_cast<float>(positions[i * 3 + 2]);
    }

    SPHRenderer renderer(config);
    return renderer.render_2d(particles);
}

double estimate_smoothing_length(
    int64_t n_particles,
    double box_size,
    int neighbors_target
) {
    double n_dens = static_cast<double>(n_particles) / (box_size * box_size * box_size);
    double mean_spacing = std::pow(1.0 / n_dens, 1.0/3.0);
    return mean_spacing * std::pow(neighbors_target / (4.0 * PI / 3.0), 1.0/3.0);
}

} // namespace density
} // namespace tessera
