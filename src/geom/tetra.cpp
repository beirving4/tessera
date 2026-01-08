#include "geom/tetra.h"
#include "math/rand.h"
#include <stdexcept>
#include <algorithm>

namespace asymptotic_tetra {
namespace geom {

// Direction lookup tables
const int64_t DIRS[TETRA_DIR_COUNT][2][3] = {
    {{1, 0, 0}, {1, 1, 0}},
    {{1, 0, 0}, {1, 0, 1}},
    {{0, 1, 0}, {1, 1, 0}},
    {{0, 0, 1}, {1, 0, 1}},
    {{0, 1, 0}, {0, 1, 1}},
    {{0, 0, 1}, {0, 1, 1}},
};

const int64_t CENTERS[TETRA_CENTERED_COUNT][3][3] = {
    {{0, 0, +1}, {0, +1, 0}, {+1, 0, 0}},
    {{0, 0, +1}, {0, +1, 0}, {-1, 0, 0}},
    {{0, 0, +1}, {0, -1, 0}, {+1, 0, 0}},
    {{0, 0, +1}, {0, -1, 0}, {-1, 0, 0}},
    {{0, 0, -1}, {0, +1, 0}, {+1, 0, 0}},
    {{0, 0, -1}, {0, +1, 0}, {-1, 0, 0}},
    {{0, 0, -1}, {0, -1, 0}, {+1, 0, 0}},
    {{0, 0, -1}, {0, -1, 0}, {-1, 0, 0}},
};

// TetraIdxs implementation
int64_t TetraIdxs::compress_coords(int64_t x, int64_t y, int64_t z,
                                    int64_t dx, int64_t dy, int64_t dz,
                                    int64_t count_width) {
    int64_t new_x = x + dx;
    int64_t new_y = y + dy;
    int64_t new_z = z + dz;
    return new_x + new_y * count_width + new_z * count_width * count_width;
}

std::pair<int64_t, bool> TetraIdxs::compress_coords_check(
    int64_t x, int64_t y, int64_t z,
    int64_t dx, int64_t dy, int64_t dz,
    int64_t count_width) {
    
    int64_t new_x = x + dx;
    int64_t new_y = y + dy;
    int64_t new_z = z + dz;

    if (new_x >= count_width || new_x < 0) return {-1, false};
    if (new_y >= count_width || new_y < 0) return {-1, false};
    if (new_z >= count_width || new_z < 0) return {-1, false};

    return {new_x + new_y * count_width + new_z * count_width * count_width, true};
}

TetraIdxs& TetraIdxs::init(int64_t idx, int64_t count_width, int64_t skip, int dir) {
    if (dir < 0 || dir >= 6) {
        throw std::runtime_error("Unknown direction for TetraIdxs.init()");
    }

    int64_t count_area = count_width * count_width;
    
    int64_t x = idx % count_width;
    int64_t y = (idx % count_area) / count_width;
    int64_t z = idx / count_area;

    indices[0] = compress_coords(
        x, y, z,
        skip * DIRS[dir][0][0], skip * DIRS[dir][0][1], skip * DIRS[dir][0][2],
        count_width
    );
    indices[1] = compress_coords(
        x, y, z,
        skip * DIRS[dir][1][0], skip * DIRS[dir][1][1], skip * DIRS[dir][1][2],
        count_width
    );
    indices[2] = compress_coords(x, y, z, skip, skip, skip, count_width);
    indices[3] = idx;

    return *this;
}

TetraIdxs& TetraIdxs::init_cartesian(int64_t x, int64_t y, int64_t z, 
                                      int64_t count_width, int dir) {
    int64_t count_area = count_width * count_width;
    indices[0] = compress_coords(
        x, y, z, DIRS[dir][0][0], DIRS[dir][0][1], DIRS[dir][0][2], count_width
    );
    indices[1] = compress_coords(
        x, y, z, DIRS[dir][1][0], DIRS[dir][1][1], DIRS[dir][1][2], count_width
    );
    indices[2] = compress_coords(x, y, z, 1, 1, 1, count_width);
    indices[3] = x + y * count_width + z * count_area;

    return *this;
}

std::pair<TetraIdxs*, bool> TetraIdxs::init_centered(int64_t idx, int64_t count_width, 
                                                      int64_t skip, int dir) {
    if (dir < 0 || dir >= 8) {
        throw std::runtime_error("Unknown direction for TetraIdxs.init_centered()");
    }

    int64_t count_area = count_width * count_width;

    int64_t x = idx % count_width;
    int64_t y = (idx % count_area) / count_width;
    int64_t z = idx / count_area;

    auto [idx0, ok0] = compress_coords_check(
        x, y, z,
        skip * CENTERS[dir][0][0],
        skip * CENTERS[dir][0][1],
        skip * CENTERS[dir][0][2],
        count_width
    );
    if (!ok0) return {this, false};
    indices[0] = idx0;

    auto [idx1, ok1] = compress_coords_check(
        x, y, z,
        skip * CENTERS[dir][1][0],
        skip * CENTERS[dir][1][1],
        skip * CENTERS[dir][1][2],
        count_width
    );
    if (!ok1) return {this, false};
    indices[1] = idx1;

    auto [idx2, ok2] = compress_coords_check(
        x, y, z,
        skip * CENTERS[dir][2][0],
        skip * CENTERS[dir][2][1],
        skip * CENTERS[dir][2][2],
        count_width
    );
    if (!ok2) return {this, false};
    indices[2] = idx2;

    indices[3] = idx;
    return {this, true};
}

// Tetra implementation
Tetra& Tetra::init(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2, const Vec3f& c3) {
    volume_valid = false;
    bary_valid = false;
    corners[0] = c0;
    corners[1] = c1;
    corners[2] = c2;
    corners[3] = c3;
    return *this;
}

double Tetra::signed_volume(const Vec3f& c0, const Vec3f& c1, 
                            const Vec3f& c2, const Vec3f& c3) {
    Vec3f& leg1 = vb_buf1;
    Vec3f& leg2 = vb_buf2;
    Vec3f& leg3 = vb_buf3;

    for (int i = 0; i < 3; ++i) {
        leg1[i] = c1[i] - c0[i];
        leg2[i] = c2[i] - c0[i];
        leg3[i] = c3[i] - c0[i];
    }

    leg2.cross_self(leg3);
    return leg1.dot(leg2) / 6.0;
}

double Tetra::volume() {
    if (volume_valid) return volume_val;
    
    volume_val = std::abs(signed_volume(corners[0], corners[1], corners[2], corners[3]));
    volume_valid = true;
    return volume_val;
}

bool Tetra::eps_eq(double x, double y, double eps) {
    return (x == 0 && y == 0) ||
           (x != 0 && y != 0 && std::abs((x - y) / x) <= eps);
}

bool Tetra::contains(const Vec3f& v) {
    double vol = volume();

    double vi = signed_volume(v, corners[0], corners[1], corners[2]);
    double vol_sum = std::abs(vi);
    bool sign = std::signbit(vi);
    if (vol_sum > vol * (1 + TETRA_EPS)) return false;

    vi = signed_volume(v, corners[1], corners[3], corners[2]);
    if (std::signbit(vi) != sign) return false;
    vol_sum += std::abs(vi);
    if (vol_sum > vol * (1 + TETRA_EPS)) return false;

    vi = signed_volume(v, corners[0], corners[3], corners[1]);
    if (std::signbit(vi) != sign) return false;
    vol_sum += std::abs(vi);
    if (vol_sum > vol * (1 + TETRA_EPS)) return false;

    vi = signed_volume(v, corners[0], corners[2], corners[3]);
    if (std::signbit(vi) != sign) return false;

    return eps_eq(std::abs(vi) + vol_sum, vol, TETRA_EPS);
}

const Vec3f& Tetra::barycenter() {
    if (bary_valid) return bary;

    for (int i = 0; i < 3; ++i) {
        bary[i] = (corners[0][i] + corners[1][i] + corners[2][i] + corners[3][i]) / 4.0f;
    }

    bary_valid = true;
    return bary;
}

CellBounds Tetra::cell_bounds(double cell_width) {
    CellBounds cb;
    cell_bounds_at(cell_width, cb);
    return cb;
}

void Tetra::cell_bounds_at(double cell_width, CellBounds& cb) {
    Vec3f mins = corners[0];
    Vec3f maxes = corners[0];

    for (int j = 1; j < 4; ++j) {
        for (int i = 0; i < 3; ++i) {
            if (corners[j][i] < mins[i]) {
                mins[i] = corners[j][i];
            } else if (corners[j][i] > maxes[i]) {
                maxes[i] = corners[j][i];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        maxes[i] -= mins[i];  // Now contains width
    }

    for (int i = 0; i < 3; ++i) {
        cb.origin[i] = static_cast<int>(std::floor(static_cast<double>(mins[i]) / cell_width));
        cb.width[i] = 1 + static_cast<int>(std::floor(
            static_cast<double>(maxes[i] + mins[i]) / cell_width
        ));
        cb.width[i] -= cb.origin[i];
    }
}

void Tetra::random_sample(math::Generator& gen, std::vector<double>& rand_buf, 
                          std::vector<Vec3f>& vec_buf) {
    size_t n = vec_buf.size();
    if (rand_buf.size() != n * 3) {
        throw std::runtime_error("rand_buf length must be 3x vec_buf length");
    }

    gen.uniform_at(0.0, 1.0, rand_buf);

    std::vector<double> xs(rand_buf.begin(), rand_buf.begin() + n);
    std::vector<double> ys(rand_buf.begin() + n, rand_buf.begin() + 2*n);
    std::vector<double> zs(rand_buf.begin() + 2*n, rand_buf.end());

    distribute(xs, ys, zs, vec_buf);
}

void Tetra::distribute(const std::vector<double>& xs, const std::vector<double>& ys,
                       const std::vector<double>& zs, std::vector<Vec3f>& vec_buf) {
    const Vec3f& bary_center = barycenter();
    
    // Compute displacement vectors to corners
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            sb_c[i][j] = corners[i][j] - bary_center[j];
        }
    }

    for (size_t i = 0; i < vec_buf.size(); ++i) {
        // Find barycentric coordinates using Rocchini-Cignoni algorithm
        float s = static_cast<float>(xs[i]);
        float t = static_cast<float>(ys[i]);
        float u = static_cast<float>(zs[i]);

        if (s + t > 1) {
            s = 1 - s;
            t = 1 - t;
        }

        if (t + u > 1) {
            float old_t = t;
            t = 1 - u;
            u = 1 - s - old_t;
        } else if (s + t + u > 1) {
            float old_s = s;
            s = 1 - t - u;
            u = old_s + t + u - 1;
        }
        float v = 1 - s - t - u;

        for (int j = 0; j < 3; ++j) {
            float d0 = sb_c[0][j] * s;
            float d1 = sb_c[1][j] * t;
            float d2 = sb_c[2][j] * u;
            float d3 = sb_c[3][j] * v;
            vec_buf[i][j] = bary_center[j] + d0 + d1 + d2 + d3;
        }
    }
}

void Tetra::distribute_tetra(const std::vector<Vec3f>& pts, std::vector<Vec3f>& out) {
    const Vec3f& bary_center = barycenter();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            sb_c[i][j] = corners[i][j] - bary_center[j];
        }
    }

    for (size_t i = 0; i < pts.size(); ++i) {
        float s = pts[i][0];
        float t = pts[i][1];
        float u = pts[i][2];
        float v = 1 - s - t - u;

        for (int j = 0; j < 3; ++j) {
            float d0 = sb_c[0][j] * s;
            float d1 = sb_c[1][j] * t;
            float d2 = sb_c[2][j] * u;
            float d3 = sb_c[3][j] * v;
            out[i][j] = bary_center[j] + d0 + d1 + d2 + d3;
        }
    }
}

void Tetra::distribute_tetra64(const std::vector<Vec3f>& pts, std::vector<std::array<double, 3>>& out) {
    const Vec3f& bary_center = barycenter();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            sb_c[i][j] = corners[i][j] - bary_center[j];
        }
    }

    for (size_t i = 0; i < pts.size(); ++i) {
        float s = pts[i][0];
        float t = pts[i][1];
        float u = pts[i][2];
        float v = 1 - s - t - u;

        for (int j = 0; j < 3; ++j) {
            float d0 = sb_c[0][j] * s;
            float d1 = sb_c[1][j] * t;
            float d2 = sb_c[2][j] * u;
            float d3 = sb_c[3][j] * v;
            out[i][j] = static_cast<double>(bary_center[j] + d0 + d1 + d2 + d3);
        }
    }
}

void distribute_unit(std::vector<Vec3f>& vec_buf) {
    for (auto& v : vec_buf) {
        float s = v[0];
        float t = v[1];
        float u = v[2];

        if (s + t > 1) {
            s = 1 - s;
            t = 1 - t;
        }

        if (t + u > 1) {
            float old_t = t;
            t = 1 - u;
            u = 1 - s - old_t;
        } else if (s + t + u > 1) {
            float old_s = s;
            s = 1 - t - u;
            u = old_s + t + u - 1;
        }

        v[0] = s;
        v[1] = t;
        v[2] = u;
    }
}

} // namespace geom
} // namespace asymptotic_tetra
