#include "io/position_tracker.h"
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <regex>
#include <limits>
#include <set>

// OpenMP support
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
namespace io {

// =============================================================================
// Helper functions
// =============================================================================

namespace {

/**
 * Safely read an attribute, returning default value if not found.
 */
template<typename T>
T read_attr(const HighFive::Group& group, const std::string& name, T default_val) {
    if (group.hasAttribute(name)) {
        T val;
        group.getAttribute(name).read(val);
        return val;
    }
    return default_val;
}

/**
 * Check if a dataset exists in a group.
 */
bool has_dataset(const HighFive::Group& group, const std::string& name) {
    return group.exist(name) && group.getObjectType(name) == HighFive::ObjectType::Dataset;
}

/**
 * Check if a group exists in the file.
 */
bool has_group(const HighFive::File& file, const std::string& name) {
    return file.exist(name) && file.getObjectType(name) == HighFive::ObjectType::Group;
}

/**
 * Read a 2D dataset and split into separate vectors.
 */
template<typename T>
void read_2d_dataset_split(
    const HighFive::DataSet& ds,
    std::vector<T>& x, std::vector<T>& y, std::vector<T>& z
) {
    auto dims = ds.getDimensions();
    if (dims.size() != 2 || dims[1] != 3) {
        throw std::runtime_error("Dataset has unexpected dimensions for 3D data");
    }

    std::vector<std::vector<T>> raw;
    ds.read(raw);

    size_t n = dims[0];
    x.resize(n);
    y.resize(n);
    z.resize(n);

    for (size_t i = 0; i < n; i++) {
        x[i] = raw[i][0];
        y[i] = raw[i][1];
        z[i] = raw[i][2];
    }
}

/**
 * Append vector src to dst.
 */
template<typename T>
void append_vector(std::vector<T>& dst, const std::vector<T>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

} // anonymous namespace

// =============================================================================
// HaloCatalogReader implementation
// =============================================================================

HaloCatalogReader::HaloCatalogReader(const std::string& catalog_dir)
    : catalog_dir_(catalog_dir) {
    // Validate directory exists
    namespace fs = std::filesystem;
    if (!fs::exists(catalog_dir) || !fs::is_directory(catalog_dir)) {
        throw std::runtime_error("Catalog directory does not exist: " + catalog_dir);
    }
}

HaloCatalogReader::~HaloCatalogReader() = default;

HaloCatalogReader::HaloCatalogReader(HaloCatalogReader&&) noexcept = default;
HaloCatalogReader& HaloCatalogReader::operator=(HaloCatalogReader&&) noexcept = default;

void HaloCatalogReader::scan_available_snapshots() const {
    if (snapshots_scanned_) return;

    namespace fs = std::filesystem;
    std::set<int32_t> snaps;

    // Pattern for FOF/SUBFIND catalogs: fof_subhalo_tab_NNN.*.hdf5 or fof_subhalo_tab_NNN.hdf5
    std::regex fof_pattern(R"(fof_subhalo_tab_(\d+)(?:\.\d+)?\.hdf5$)");
    // Alternative: groups_NNN.*.hdf5
    std::regex groups_pattern(R"(groups_(\d+)(?:\.\d+)?\.hdf5$)");
    // Subdirectory pattern: groups_NNN/
    std::regex subdir_pattern(R"(groups_(\d+)$)");

    for (const auto& entry : fs::directory_iterator(catalog_dir_)) {
        std::string fname = entry.path().filename().string();
        std::smatch match;

        if (entry.is_regular_file()) {
            // Check for files directly in catalog_dir
            if (std::regex_search(fname, match, fof_pattern)) {
                snaps.insert(std::stoi(match[1].str()));
            } else if (std::regex_search(fname, match, groups_pattern)) {
                snaps.insert(std::stoi(match[1].str()));
            }
        } else if (entry.is_directory()) {
            // Check for groups_NNN subdirectories
            if (std::regex_search(fname, match, subdir_pattern)) {
                snaps.insert(std::stoi(match[1].str()));
            }
        }
    }

    available_snapshots_.assign(snaps.begin(), snaps.end());
    snapshots_scanned_ = true;
}

std::vector<int32_t> HaloCatalogReader::get_available_snapshots() const {
    scan_available_snapshots();
    return available_snapshots_;
}

std::vector<std::string> HaloCatalogReader::find_catalog_files(int32_t snap_num) const {
    namespace fs = std::filesystem;
    std::vector<std::pair<int, std::string>> indexed_files;

    // Use 3-digit padding for pattern matching
    char padded[16];
    std::snprintf(padded, sizeof(padded), "%03d", snap_num);
    std::string snap_padded(padded);

    // Patterns for sharded files
    std::regex fof_sharded_padded(R"(fof_subhalo_tab_)" + snap_padded + R"(\.(\d+)\.hdf5$)");
    std::regex groups_sharded_padded(R"(groups_)" + snap_padded + R"(\.(\d+)\.hdf5$)");

    // Single file patterns
    std::string fof_single = "fof_subhalo_tab_" + snap_padded + ".hdf5";
    std::string groups_single = "groups_" + snap_padded + ".hdf5";

    // Check for subdirectory structure: catalog_dir/groups_NNN/fof_subhalo_tab_NNN.*.hdf5
    fs::path subdir = fs::path(catalog_dir_) / ("groups_" + snap_padded);
    fs::path search_dir = fs::exists(subdir) ? subdir : fs::path(catalog_dir_);

    for (const auto& entry : fs::directory_iterator(search_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string fname = entry.path().filename().string();
        std::smatch match;

        // Check for single file
        if (fname == fof_single || fname == groups_single) {
            return {entry.path().string()};
        }

        // Check for sharded files
        if (std::regex_search(fname, match, fof_sharded_padded) ||
            std::regex_search(fname, match, groups_sharded_padded)) {
            int idx = std::stoi(match[1].str());
            indexed_files.emplace_back(idx, entry.path().string());
        }
    }

    // Sort by shard index
    std::sort(indexed_files.begin(), indexed_files.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> result;
    result.reserve(indexed_files.size());
    for (const auto& [idx, path] : indexed_files) {
        result.push_back(path);
    }

    return result;
}

HaloCatalogData HaloCatalogReader::load_snapshot(int32_t snap_num) const {
    HaloCatalogData data;
    data.snap_num = snap_num;

    auto files = find_catalog_files(snap_num);
    if (files.empty()) {
        return data;  // Return empty data
    }

    // Collect data from all shards
    std::vector<float> all_pos_x, all_pos_y, all_pos_z;
    std::vector<float> all_m200c, all_r200c;

    for (const auto& fpath : files) {
        try {
            HighFive::File file(fpath, HighFive::File::ReadOnly);

            // Check for Group data
            if (!has_group(file, "Group")) {
                continue;
            }

            auto group = file.getGroup("Group");

            // Read positions
            if (has_dataset(group, "GroupPos")) {
                std::vector<float> px, py, pz;
                read_2d_dataset_split(group.getDataSet("GroupPos"), px, py, pz);
                append_vector(all_pos_x, px);
                append_vector(all_pos_y, py);
                append_vector(all_pos_z, pz);
            }

            // Read M200c
            if (has_dataset(group, "Group_M_Crit200")) {
                std::vector<float> m200c;
                group.getDataSet("Group_M_Crit200").read(m200c);
                append_vector(all_m200c, m200c);
            }

            // Read R200c (optional)
            if (has_dataset(group, "Group_R_Crit200")) {
                std::vector<float> r200c;
                group.getDataSet("Group_R_Crit200").read(r200c);
                append_vector(all_r200c, r200c);
            }

            // Get scale factor from header (first file only)
            if (data.scale_factor == 0.0f && file.exist("Header")) {
                auto header = file.getGroup("Header");
                data.scale_factor = read_attr<float>(header, "Time", 0.0f);
            }

        } catch (const std::exception& e) {
            std::cerr << "Warning: Error reading " << fpath << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Move data to result
    data.pos_x = std::move(all_pos_x);
    data.pos_y = std::move(all_pos_y);
    data.pos_z = std::move(all_pos_z);
    data.m200c = std::move(all_m200c);
    data.r200c = std::move(all_r200c);

    // If no R200c, fill with zeros
    if (data.r200c.empty() && !data.m200c.empty()) {
        data.r200c.resize(data.m200c.size(), 0.0f);
    }

    return data;
}

std::map<int32_t, HaloCatalogData> HaloCatalogReader::load_snapshots(
    const std::vector<int32_t>& snap_nums
) const {
    std::map<int32_t, HaloCatalogData> result;

    for (int32_t snap : snap_nums) {
        auto data = load_snapshot(snap);
        if (data.is_valid()) {
            result[snap] = std::move(data);
        }
    }

    return result;
}

std::map<int32_t, HaloCatalogData> HaloCatalogReader::load_all_snapshots() const {
    return load_snapshots(get_available_snapshots());
}

// =============================================================================
// PositionMatcher implementation
// =============================================================================

PositionMatcher::PositionMatcher(float box_size, PositionMatchConfig config)
    : box_size_(box_size), config_(config) {}

float PositionMatcher::periodic_distance(
    const std::array<float, 3>& pos1,
    const std::array<float, 3>& pos2
) const {
    float dist_sq = 0.0f;
    for (int i = 0; i < 3; i++) {
        float delta = std::abs(pos1[i] - pos2[i]);
        if (delta > box_size_ / 2.0f) {
            delta = box_size_ - delta;
        }
        dist_sq += delta * delta;
    }
    return std::sqrt(dist_sq);
}

void PositionMatcher::compute_distances_simd(
    const std::array<float, 3>& target,
    const HaloCatalogData& catalog,
    std::vector<float>& distances
) const {
    const size_t n = catalog.n_groups();
    distances.resize(n);

#if TESSERA_HAS_AVX2
    // Process 8 groups at a time using AVX2
    const float half_box = box_size_ / 2.0f;
    const __m256 target_x = _mm256_set1_ps(target[0]);
    const __m256 target_y = _mm256_set1_ps(target[1]);
    const __m256 target_z = _mm256_set1_ps(target[2]);
    const __m256 half_box_vec = _mm256_set1_ps(half_box);
    const __m256 box_size_vec = _mm256_set1_ps(box_size_);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load positions
        __m256 px = _mm256_loadu_ps(&catalog.pos_x[i]);
        __m256 py = _mm256_loadu_ps(&catalog.pos_y[i]);
        __m256 pz = _mm256_loadu_ps(&catalog.pos_z[i]);

        // Compute |delta|
        __m256 dx = _mm256_and_ps(_mm256_sub_ps(px, target_x), sign_mask);
        __m256 dy = _mm256_and_ps(_mm256_sub_ps(py, target_y), sign_mask);
        __m256 dz = _mm256_and_ps(_mm256_sub_ps(pz, target_z), sign_mask);

        // Apply periodic wrapping: if delta > half_box, delta = box_size - delta
        __m256 wrap_x = _mm256_cmp_ps(dx, half_box_vec, _CMP_GT_OQ);
        __m256 wrap_y = _mm256_cmp_ps(dy, half_box_vec, _CMP_GT_OQ);
        __m256 wrap_z = _mm256_cmp_ps(dz, half_box_vec, _CMP_GT_OQ);

        dx = _mm256_blendv_ps(dx, _mm256_sub_ps(box_size_vec, dx), wrap_x);
        dy = _mm256_blendv_ps(dy, _mm256_sub_ps(box_size_vec, dy), wrap_y);
        dz = _mm256_blendv_ps(dz, _mm256_sub_ps(box_size_vec, dz), wrap_z);

        // Compute distance squared
        __m256 dist_sq = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
            _mm256_mul_ps(dz, dz)
        );

        // Compute sqrt
        __m256 dist = _mm256_sqrt_ps(dist_sq);

        // Store
        _mm256_storeu_ps(&distances[i], dist);
    }

    // Handle remainder
    for (; i < n; i++) {
        std::array<float, 3> pos = {catalog.pos_x[i], catalog.pos_y[i], catalog.pos_z[i]};
        distances[i] = periodic_distance(target, pos);
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < n; i++) {
        std::array<float, 3> pos = {catalog.pos_x[i], catalog.pos_y[i], catalog.pos_z[i]};
        distances[i] = periodic_distance(target, pos);
    }
#endif
}

PositionMatchResult PositionMatcher::find_best_match(
    const std::array<float, 3>& target_pos,
    float target_mass,
    const HaloCatalogData& catalog
) const {
    PositionMatchResult result;

    if (catalog.empty()) {
        return result;
    }

    // Compute distances to all groups
    std::vector<float> distances;
    compute_distances_simd(target_pos, catalog, distances);

    const float min_mass = target_mass * config_.min_mass_ratio;

    // Find best match
    for (size_t i = 0; i < catalog.n_groups(); i++) {
        float dist = distances[i];
        float mass = catalog.m200c[i];

        // Skip if outside search radius or below minimum mass
        if (dist >= config_.search_radius || mass <= min_mass) {
            continue;
        }

        // Compute score
        float mass_ratio = 0.0f;
        if (target_mass > 0.0f && mass > 0.0f) {
            mass_ratio = std::abs(std::log10(mass / target_mass));
        }
        float score = dist + config_.mass_weight * mass_ratio;

        if (score < result.score) {
            result.group_nr = static_cast<int32_t>(i);
            result.score = score;
            result.distance = dist;
        }
    }

    return result;
}

std::vector<PositionMatchResult> PositionMatcher::find_best_matches_batch(
    const std::vector<std::array<float, 3>>& target_positions,
    const std::vector<float>& target_masses,
    const HaloCatalogData& catalog
) const {
    const size_t n = target_positions.size();
    std::vector<PositionMatchResult> results(n);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (size_t i = 0; i < n; i++) {
        results[i] = find_best_match(target_positions[i], target_masses[i], catalog);
    }

    return results;
}

// =============================================================================
// PositionBranchExtractor implementation
// =============================================================================

PositionBranchExtractor::PositionBranchExtractor(
    HaloCatalogReader& reader,
    float box_size,
    PositionMatchConfig config
) : reader_(reader), matcher_(box_size, config) {}

std::vector<TrackedHaloInfo> PositionBranchExtractor::track_branch(
    int32_t start_snap,
    int32_t start_group_nr,
    bool backward,
    bool forward,
    std::optional<int32_t> snap_min,
    std::optional<int32_t> snap_max
) {
    // Load starting catalog
    auto start_catalog = reader_.load_snapshot(start_snap);
    if (!start_catalog.is_valid() || start_group_nr >= static_cast<int32_t>(start_catalog.n_groups())) {
        return {};
    }

    // Create starting halo info
    TrackedHaloInfo start_halo;
    start_halo.snap_num = start_snap;
    start_halo.group_nr = start_group_nr;
    start_halo.scale_factor = start_catalog.scale_factor;
    start_halo.position = {
        start_catalog.pos_x[start_group_nr],
        start_catalog.pos_y[start_group_nr],
        start_catalog.pos_z[start_group_nr]
    };
    start_halo.m200c = start_catalog.m200c[start_group_nr];
    start_halo.r200c = start_catalog.r200c.empty() ? 0.0f : start_catalog.r200c[start_group_nr];
    start_halo.match_score = 0.0f;
    start_halo.match_distance = 0.0f;

    std::vector<TrackedHaloInfo> branch = {start_halo};

    // Get available snapshots
    auto available = reader_.get_available_snapshots();
    if (available.empty()) {
        return branch;
    }

    // Apply limits
    std::vector<int32_t> valid_snaps;
    for (int32_t s : available) {
        if (snap_min && s < *snap_min) continue;
        if (snap_max && s > *snap_max) continue;
        valid_snaps.push_back(s);
    }

    // Track backward
    if (backward) {
        std::vector<int32_t> backward_snaps;
        for (int32_t s : valid_snaps) {
            if (s < start_snap) backward_snaps.push_back(s);
        }
        std::sort(backward_snaps.rbegin(), backward_snaps.rend());  // Descending

        std::array<float, 3> current_pos = start_halo.position;
        float current_mass = start_halo.m200c;

        for (int32_t snap : backward_snaps) {
            auto catalog = reader_.load_snapshot(snap);
            if (!catalog.is_valid()) continue;

            auto match = matcher_.find_best_match(current_pos, current_mass, catalog);
            if (!match.found()) continue;

            TrackedHaloInfo halo;
            halo.snap_num = snap;
            halo.group_nr = match.group_nr;
            halo.scale_factor = catalog.scale_factor;
            halo.position = {
                catalog.pos_x[match.group_nr],
                catalog.pos_y[match.group_nr],
                catalog.pos_z[match.group_nr]
            };
            halo.m200c = catalog.m200c[match.group_nr];
            halo.r200c = catalog.r200c.empty() ? 0.0f : catalog.r200c[match.group_nr];
            halo.match_score = match.score;
            halo.match_distance = match.distance;

            branch.push_back(halo);

            current_pos = halo.position;
            current_mass = halo.m200c;
        }
    }

    // Track forward
    if (forward) {
        std::vector<int32_t> forward_snaps;
        for (int32_t s : valid_snaps) {
            if (s > start_snap) forward_snaps.push_back(s);
        }
        std::sort(forward_snaps.begin(), forward_snaps.end());  // Ascending

        std::array<float, 3> current_pos = start_halo.position;
        float current_mass = start_halo.m200c;

        for (int32_t snap : forward_snaps) {
            auto catalog = reader_.load_snapshot(snap);
            if (!catalog.is_valid()) continue;

            auto match = matcher_.find_best_match(current_pos, current_mass, catalog);
            if (!match.found()) continue;

            TrackedHaloInfo halo;
            halo.snap_num = snap;
            halo.group_nr = match.group_nr;
            halo.scale_factor = catalog.scale_factor;
            halo.position = {
                catalog.pos_x[match.group_nr],
                catalog.pos_y[match.group_nr],
                catalog.pos_z[match.group_nr]
            };
            halo.m200c = catalog.m200c[match.group_nr];
            halo.r200c = catalog.r200c.empty() ? 0.0f : catalog.r200c[match.group_nr];
            halo.match_score = match.score;
            halo.match_distance = match.distance;

            branch.push_back(halo);

            current_pos = halo.position;
            current_mass = halo.m200c;
        }
    }

    // Sort by scale factor (early to late)
    std::sort(branch.begin(), branch.end(),
              [](const TrackedHaloInfo& a, const TrackedHaloInfo& b) {
                  return a.scale_factor < b.scale_factor;
              });

    return branch;
}

std::vector<TrackedHaloInfo> PositionBranchExtractor::track_branch_preloaded(
    const std::map<int32_t, HaloCatalogData>& catalogs,
    int32_t start_snap,
    int32_t start_group_nr,
    bool backward,
    bool forward
) {
    // Find starting catalog
    auto start_it = catalogs.find(start_snap);
    if (start_it == catalogs.end() || !start_it->second.is_valid()) {
        return {};
    }

    const auto& start_catalog = start_it->second;
    if (start_group_nr >= static_cast<int32_t>(start_catalog.n_groups())) {
        return {};
    }

    // Create starting halo info
    TrackedHaloInfo start_halo;
    start_halo.snap_num = start_snap;
    start_halo.group_nr = start_group_nr;
    start_halo.scale_factor = start_catalog.scale_factor;
    start_halo.position = {
        start_catalog.pos_x[start_group_nr],
        start_catalog.pos_y[start_group_nr],
        start_catalog.pos_z[start_group_nr]
    };
    start_halo.m200c = start_catalog.m200c[start_group_nr];
    start_halo.r200c = start_catalog.r200c.empty() ? 0.0f : start_catalog.r200c[start_group_nr];
    start_halo.match_score = 0.0f;
    start_halo.match_distance = 0.0f;

    std::vector<TrackedHaloInfo> branch = {start_halo};

    // Collect available snapshots
    std::vector<int32_t> backward_snaps, forward_snaps;
    for (const auto& [snap, _] : catalogs) {
        if (snap < start_snap) backward_snaps.push_back(snap);
        else if (snap > start_snap) forward_snaps.push_back(snap);
    }
    std::sort(backward_snaps.rbegin(), backward_snaps.rend());  // Descending
    std::sort(forward_snaps.begin(), forward_snaps.end());       // Ascending

    // Track backward
    if (backward) {
        std::array<float, 3> current_pos = start_halo.position;
        float current_mass = start_halo.m200c;

        for (int32_t snap : backward_snaps) {
            const auto& catalog = catalogs.at(snap);
            if (!catalog.is_valid()) continue;

            auto match = matcher_.find_best_match(current_pos, current_mass, catalog);
            if (!match.found()) continue;

            TrackedHaloInfo halo;
            halo.snap_num = snap;
            halo.group_nr = match.group_nr;
            halo.scale_factor = catalog.scale_factor;
            halo.position = {
                catalog.pos_x[match.group_nr],
                catalog.pos_y[match.group_nr],
                catalog.pos_z[match.group_nr]
            };
            halo.m200c = catalog.m200c[match.group_nr];
            halo.r200c = catalog.r200c.empty() ? 0.0f : catalog.r200c[match.group_nr];
            halo.match_score = match.score;
            halo.match_distance = match.distance;

            branch.push_back(halo);

            current_pos = halo.position;
            current_mass = halo.m200c;
        }
    }

    // Track forward
    if (forward) {
        std::array<float, 3> current_pos = start_halo.position;
        float current_mass = start_halo.m200c;

        for (int32_t snap : forward_snaps) {
            const auto& catalog = catalogs.at(snap);
            if (!catalog.is_valid()) continue;

            auto match = matcher_.find_best_match(current_pos, current_mass, catalog);
            if (!match.found()) continue;

            TrackedHaloInfo halo;
            halo.snap_num = snap;
            halo.group_nr = match.group_nr;
            halo.scale_factor = catalog.scale_factor;
            halo.position = {
                catalog.pos_x[match.group_nr],
                catalog.pos_y[match.group_nr],
                catalog.pos_z[match.group_nr]
            };
            halo.m200c = catalog.m200c[match.group_nr];
            halo.r200c = catalog.r200c.empty() ? 0.0f : catalog.r200c[match.group_nr];
            halo.match_score = match.score;
            halo.match_distance = match.distance;

            branch.push_back(halo);

            current_pos = halo.position;
            current_mass = halo.m200c;
        }
    }

    // Sort by scale factor (early to late)
    std::sort(branch.begin(), branch.end(),
              [](const TrackedHaloInfo& a, const TrackedHaloInfo& b) {
                  return a.scale_factor < b.scale_factor;
              });

    return branch;
}

void PositionBranchExtractor::unwrap_coordinates(std::vector<TrackedHaloInfo>& branch) {
    if (branch.size() < 2) return;

    const float half_box = matcher_.box_size() / 2.0f;
    const float box_size = matcher_.box_size();

    for (size_t i = 1; i < branch.size(); i++) {
        for (int dim = 0; dim < 3; dim++) {
            float diff = branch[i].position[dim] - branch[i-1].position[dim];
            if (diff > half_box) {
                branch[i].position[dim] -= box_size;
            } else if (diff < -half_box) {
                branch[i].position[dim] += box_size;
            }
        }
    }
}

void PositionBranchExtractor::preload_all_catalogs() {
    preloaded_catalogs_ = reader_.load_all_snapshots();
}

void PositionBranchExtractor::clear_preloaded_catalogs() {
    preloaded_catalogs_.clear();
}

// =============================================================================
// Parallel extraction
// =============================================================================

PositionBranchCatalogData extract_position_branches_parallel(
    HaloCatalogReader& reader,
    float box_size,
    const std::vector<BranchStartPoint>& start_points,
    PositionMatchConfig config,
    PositionBranchExtractor::Strategy strategy,
    int n_threads,
    std::function<void(size_t, size_t)> progress_callback
) {
    const size_t n_branches = start_points.size();
    PositionBranchCatalogData result;

    if (n_branches == 0) {
        return result;
    }

#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#endif

    // Storage for per-branch results
    std::vector<std::vector<TrackedHaloInfo>> all_branches(n_branches);
    std::atomic<size_t> completed{0};

    if (strategy == PositionBranchExtractor::Strategy::PRELOAD_CATALOGS) {
        // Option B: Preload all catalogs, then parallelize matching
        std::cout << "Loading all halo catalogs..." << std::endl;
        auto catalogs = reader.load_all_snapshots();
        std::cout << "Loaded " << catalogs.size() << " snapshots" << std::endl;

        PositionMatcher matcher(box_size, config);

#ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 16)
            for (size_t i = 0; i < n_branches; i++) {
                const auto& start = start_points[i];

                // Find starting catalog
                auto start_it = catalogs.find(start.snap_num);
                if (start_it == catalogs.end() || !start_it->second.is_valid()) {
                    completed++;
                    continue;
                }

                const auto& start_catalog = start_it->second;
                if (start.group_nr >= static_cast<int32_t>(start_catalog.n_groups())) {
                    completed++;
                    continue;
                }

                // Track branch using preloaded catalogs
                std::vector<TrackedHaloInfo> branch;

                // Create starting halo
                TrackedHaloInfo start_halo;
                start_halo.snap_num = start.snap_num;
                start_halo.group_nr = start.group_nr;
                start_halo.scale_factor = start_catalog.scale_factor;
                start_halo.position = {
                    start_catalog.pos_x[start.group_nr],
                    start_catalog.pos_y[start.group_nr],
                    start_catalog.pos_z[start.group_nr]
                };
                start_halo.m200c = start_catalog.m200c[start.group_nr];
                start_halo.r200c = start_catalog.r200c.empty() ? 0.0f
                                   : start_catalog.r200c[start.group_nr];
                branch.push_back(start_halo);

                // Get sorted snapshots
                std::vector<int32_t> backward_snaps;
                for (const auto& [snap, _] : catalogs) {
                    if (snap < start.snap_num) backward_snaps.push_back(snap);
                }
                std::sort(backward_snaps.rbegin(), backward_snaps.rend());

                // Track backward
                std::array<float, 3> current_pos = start_halo.position;
                float current_mass = start_halo.m200c;

                for (int32_t snap : backward_snaps) {
                    const auto& catalog = catalogs.at(snap);
                    if (!catalog.is_valid()) continue;

                    auto match = matcher.find_best_match(current_pos, current_mass, catalog);
                    if (!match.found()) continue;

                    TrackedHaloInfo halo;
                    halo.snap_num = snap;
                    halo.group_nr = match.group_nr;
                    halo.scale_factor = catalog.scale_factor;
                    halo.position = {
                        catalog.pos_x[match.group_nr],
                        catalog.pos_y[match.group_nr],
                        catalog.pos_z[match.group_nr]
                    };
                    halo.m200c = catalog.m200c[match.group_nr];
                    halo.r200c = catalog.r200c.empty() ? 0.0f : catalog.r200c[match.group_nr];
                    halo.match_score = match.score;
                    halo.match_distance = match.distance;

                    branch.push_back(halo);
                    current_pos = halo.position;
                    current_mass = halo.m200c;
                }

                // Sort by scale factor
                std::sort(branch.begin(), branch.end(),
                          [](const TrackedHaloInfo& a, const TrackedHaloInfo& b) {
                              return a.scale_factor < b.scale_factor;
                          });

                all_branches[i] = std::move(branch);

                size_t done = ++completed;
                if (progress_callback && done % 100 == 0) {
                    progress_callback(done, n_branches);
                }
            }
        }
#else
        // Single-threaded fallback
        PositionBranchExtractor extractor(reader, box_size, config);
        for (size_t i = 0; i < n_branches; i++) {
            const auto& start = start_points[i];
            all_branches[i] = extractor.track_branch_preloaded(
                catalogs, start.snap_num, start.group_nr, true, false
            );
            if (progress_callback && (i + 1) % 100 == 0) {
                progress_callback(i + 1, n_branches);
            }
        }
#endif

    } else {
        // Option A: Parallelize across branches (each thread loads catalogs)
#ifdef _OPENMP
        #pragma omp parallel
        {
            // Each thread gets its own extractor (and thus its own catalog cache)
            PositionBranchExtractor extractor(reader, box_size, config);

            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < n_branches; i++) {
                const auto& start = start_points[i];
                all_branches[i] = extractor.track_branch(
                    start.snap_num, start.group_nr, true, false
                );

                size_t done = ++completed;
                if (progress_callback && done % 10 == 0) {
                    progress_callback(done, n_branches);
                }
            }
        }
#else
        PositionBranchExtractor extractor(reader, box_size, config);
        for (size_t i = 0; i < n_branches; i++) {
            const auto& start = start_points[i];
            all_branches[i] = extractor.track_branch(
                start.snap_num, start.group_nr, true, false
            );
            if (progress_callback && (i + 1) % 10 == 0) {
                progress_callback(i + 1, n_branches);
            }
        }
#endif
    }

    // Merge results into flattened format
    size_t total_halos = 0;
    for (const auto& branch : all_branches) {
        total_halos += branch.size();
    }

    // Reserve space
    result.branch_id.reserve(n_branches);
    result.start_offset.reserve(n_branches);
    result.length.reserve(n_branches);
    result.root_snap.reserve(n_branches);
    result.root_group.reserve(n_branches);
    result.root_m200c.reserve(n_branches);
    result.a_min.reserve(n_branches);
    result.a_max.reserve(n_branches);

    result.snap_num.reserve(total_halos);
    result.group_nr.reserve(total_halos);
    result.scale_factor.reserve(total_halos);
    result.pos_x.reserve(total_halos);
    result.pos_y.reserve(total_halos);
    result.pos_z.reserve(total_halos);
    result.m200c.reserve(total_halos);
    result.r200c.reserve(total_halos);
    result.match_score.reserve(total_halos);
    result.match_distance.reserve(total_halos);

    int64_t offset = 0;
    for (size_t i = 0; i < n_branches; i++) {
        const auto& branch = all_branches[i];
        if (branch.empty()) continue;

        // Branch index
        int32_t branch_id = (start_points[i].branch_id >= 0) ?
                            start_points[i].branch_id : static_cast<int32_t>(i);
        result.branch_id.push_back(branch_id);
        result.start_offset.push_back(offset);
        result.length.push_back(static_cast<int32_t>(branch.size()));

        // Root info (last halo = highest scale factor)
        const auto& root = branch.back();
        result.root_snap.push_back(root.snap_num);
        result.root_group.push_back(root.group_nr);
        result.root_m200c.push_back(root.m200c);

        // Scale factor range
        result.a_min.push_back(branch.front().scale_factor);
        result.a_max.push_back(branch.back().scale_factor);

        // Branch data
        for (const auto& halo : branch) {
            result.snap_num.push_back(halo.snap_num);
            result.group_nr.push_back(halo.group_nr);
            result.scale_factor.push_back(halo.scale_factor);
            result.pos_x.push_back(halo.position[0]);
            result.pos_y.push_back(halo.position[1]);
            result.pos_z.push_back(halo.position[2]);
            result.m200c.push_back(halo.m200c);
            result.r200c.push_back(halo.r200c);
            result.match_score.push_back(halo.match_score);
            result.match_distance.push_back(halo.match_distance);
        }

        offset += branch.size();
    }

    if (progress_callback) {
        progress_callback(n_branches, n_branches);
    }

    return result;
}

// =============================================================================
// Validation function
// =============================================================================

std::map<int32_t, std::unordered_map<std::string, float>> validate_branch_with_position_matching(
    const std::vector<int32_t>& tree_snap_nums,
    const std::vector<int32_t>& tree_group_nrs,
    const std::vector<float>& tree_m200c,
    HaloCatalogReader& reader,
    float box_size,
    PositionMatchConfig config
) {
    std::map<int32_t, std::unordered_map<std::string, float>> result;

    if (tree_snap_nums.empty()) {
        return result;
    }

    // Find the final snapshot (highest scale factor proxy)
    int32_t ref_snap = *std::max_element(tree_snap_nums.begin(), tree_snap_nums.end());
    int32_t ref_group = -1;

    for (size_t i = 0; i < tree_snap_nums.size(); i++) {
        if (tree_snap_nums[i] == ref_snap) {
            ref_group = tree_group_nrs[i];
            break;
        }
    }

    if (ref_group < 0) {
        return result;
    }

    // Do position-matched tracking
    PositionBranchExtractor extractor(reader, box_size, config);
    auto pos_branch = extractor.track_branch(ref_snap, ref_group, true, true);

    // Build lookup by snapshot
    std::map<int32_t, TrackedHaloInfo> pos_by_snap;
    for (const auto& h : pos_branch) {
        pos_by_snap[h.snap_num] = h;
    }

    // Compare
    for (size_t i = 0; i < tree_snap_nums.size(); i++) {
        int32_t snap = tree_snap_nums[i];
        int32_t tree_group = tree_group_nrs[i];
        float tree_mass = tree_m200c[i];

        std::unordered_map<std::string, float> snap_result;
        snap_result["tree_group"] = static_cast<float>(tree_group);

        auto pos_it = pos_by_snap.find(snap);
        if (pos_it != pos_by_snap.end()) {
            const auto& pos_h = pos_it->second;
            snap_result["pos_group"] = static_cast<float>(pos_h.group_nr);
            snap_result["match"] = (tree_group == pos_h.group_nr) ? 1.0f : 0.0f;

            if (tree_mass > 0.0f && pos_h.m200c > 0.0f) {
                snap_result["mass_ratio"] = pos_h.m200c / tree_mass;
            } else {
                snap_result["mass_ratio"] = -1.0f;  // Invalid
            }

            snap_result["tree_m200c"] = tree_mass;
            snap_result["pos_m200c"] = pos_h.m200c;
        } else {
            snap_result["pos_group"] = -1.0f;
            snap_result["match"] = 0.0f;
            snap_result["mass_ratio"] = -1.0f;
            snap_result["tree_m200c"] = tree_mass;
            snap_result["pos_m200c"] = -1.0f;
        }

        result[snap] = snap_result;
    }

    return result;
}

} // namespace io
} // namespace tessera
