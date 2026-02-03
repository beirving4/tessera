#include "io/merger_tree.h"
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <queue>
#include <numeric>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include <regex>

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
// Helper functions (following patterns from gadget4.cpp)
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
 * Read a 1D dataset.
 */
template<typename T>
std::vector<T> read_1d_dataset(const HighFive::DataSet& ds) {
    std::vector<T> result;
    ds.read(result);
    return result;
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
 * Discover all merger tree files in a directory.
 * Handles split files (trees.0.hdf5, trees.1.hdf5, ...) and single files (trees.hdf5).
 *
 * @param path Path to directory or first file (trees.0.hdf5)
 * @return Vector of file paths sorted by file index
 */
std::vector<std::string> discover_tree_files(const std::string& path) {
    namespace fs = std::filesystem;

    fs::path input_path(path);
    fs::path directory;

    // Determine if input is a file or directory
    if (fs::is_directory(input_path)) {
        directory = input_path;
    } else if (fs::is_regular_file(input_path)) {
        directory = input_path.parent_path();
    } else {
        throw std::runtime_error("Path does not exist: " + path);
    }

    // Look for tree files in the directory
    std::vector<std::pair<int, std::string>> indexed_files;

    // Pattern for split files: trees.N.hdf5
    std::regex split_pattern(R"(trees\.(\d+)\.hdf5$)");

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;

        std::string fname = entry.path().filename().string();
        std::smatch match;

        if (std::regex_search(fname, match, split_pattern)) {
            int idx = std::stoi(match[1].str());
            indexed_files.emplace_back(idx, entry.path().string());
        } else if (fname == "trees.hdf5") {
            // Single file case
            return {entry.path().string()};
        }
    }

    if (indexed_files.empty()) {
        throw std::runtime_error("No tree files found in directory: " + directory.string());
    }

    // Sort by file index
    std::sort(indexed_files.begin(), indexed_files.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Extract paths
    std::vector<std::string> result;
    result.reserve(indexed_files.size());
    for (const auto& [idx, filepath] : indexed_files) {
        result.push_back(filepath);
    }

    return result;
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
// TreeHalosSOA methods
// =============================================================================

void TreeHalosSOA::reserve(size_t n) {
    snap_num.reserve(n);
    subhalo_nr.reserve(n);
    group_nr.reserve(n);
    tree_id.reserve(n);
    tree_index.reserve(n);

    pos_x.reserve(n);
    pos_y.reserve(n);
    pos_z.reserve(n);
    vel_x.reserve(n);
    vel_y.reserve(n);
    vel_z.reserve(n);

    mass.reserve(n);
    m200c.reserve(n);
    r200c.reserve(n);

    main_progenitor.reserve(n);
    descendant.reserve(n);
    first_progenitor.reserve(n);
    next_progenitor.reserve(n);
}

void TreeHalosSOA::clear() {
    snap_num.clear();
    subhalo_nr.clear();
    group_nr.clear();
    tree_id.clear();
    tree_index.clear();

    pos_x.clear();
    pos_y.clear();
    pos_z.clear();
    vel_x.clear();
    vel_y.clear();
    vel_z.clear();

    mass.clear();
    m200c.clear();
    r200c.clear();

    main_progenitor.clear();
    descendant.clear();
    first_progenitor.clear();
    next_progenitor.clear();
}

void TreeHalosSOA::clear_full_data() {
    // Clear Stage 2 data only, keep search data (snap_num, subhalo_nr)
    group_nr.clear();
    group_nr.shrink_to_fit();
    tree_id.clear();
    tree_id.shrink_to_fit();
    tree_index.clear();
    tree_index.shrink_to_fit();

    pos_x.clear();
    pos_x.shrink_to_fit();
    pos_y.clear();
    pos_y.shrink_to_fit();
    pos_z.clear();
    pos_z.shrink_to_fit();
    vel_x.clear();
    vel_x.shrink_to_fit();
    vel_y.clear();
    vel_y.shrink_to_fit();
    vel_z.clear();
    vel_z.shrink_to_fit();

    mass.clear();
    mass.shrink_to_fit();
    m200c.clear();
    m200c.shrink_to_fit();
    r200c.clear();
    r200c.shrink_to_fit();

    main_progenitor.clear();
    main_progenitor.shrink_to_fit();
    descendant.clear();
    descendant.shrink_to_fit();
    first_progenitor.clear();
    first_progenitor.shrink_to_fit();
    next_progenitor.clear();
    next_progenitor.shrink_to_fit();
}

// =============================================================================
// MergerTree implementation
// =============================================================================

MergerTree::MergerTree(const std::string& filename)
    : files_(discover_tree_files(filename)) {
    if (files_.empty()) {
        throw std::runtime_error("No tree files found for: " + filename);
    }
    load_header();
}

MergerTree::~MergerTree() = default;

MergerTree::MergerTree(MergerTree&&) noexcept = default;
MergerTree& MergerTree::operator=(MergerTree&&) noexcept = default;

void MergerTree::load_header() {
    // Read header from first file (contains totals across all files)
    HighFive::File file(files_[0], HighFive::File::ReadOnly);

    if (!file.exist("Header")) {
        throw std::runtime_error("File does not contain Header group: " + files_[0]);
    }

    auto header_group = file.getGroup("Header");

    // Basic counts
    header_.n_trees = read_attr<int64_t>(header_group, "Ntrees_Total", 0);
    header_.n_halos = read_attr<int64_t>(header_group, "Nhalos_Total", 0);
    header_.n_files = read_attr<int32_t>(header_group, "NumFiles", 1);
    header_.last_snap = read_attr<int32_t>(header_group, "LastSnapShotNr", 0);

    // Read time arrays
    if (file.exist("TreeTimes")) {
        auto times_group = file.getGroup("TreeTimes");

        if (has_dataset(times_group, "Time")) {
            auto ds = times_group.getDataSet("Time");
            std::vector<double> times;
            ds.read(times);
            header_.snap_times.resize(times.size());
            for (size_t i = 0; i < times.size(); i++) {
                header_.snap_times[i] = static_cast<float>(times[i]);
            }
        }

        if (has_dataset(times_group, "Redshift")) {
            auto ds = times_group.getDataSet("Redshift");
            std::vector<double> redshifts;
            ds.read(redshifts);
            header_.snap_redshifts.resize(redshifts.size());
            for (size_t i = 0; i < redshifts.size(); i++) {
                header_.snap_redshifts[i] = static_cast<float>(redshifts[i]);
            }
        }
    }

    // Read cosmological parameters
    if (file.exist("Parameters")) {
        auto params_group = file.getGroup("Parameters");
        header_.box_size = read_attr<double>(params_group, "BoxSize", 0.0);
        header_.hubble = read_attr<double>(params_group, "HubbleParam", 0.0);
        header_.omega_matter = read_attr<double>(params_group, "Omega0", 0.0);
        header_.omega_lambda = read_attr<double>(params_group, "OmegaLambda", 0.0);
    }
}

void MergerTree::load_tree_table() {
    if (table_loaded_) return;

    // TreeTable is split across files - only files with tree roots have TreeTable
    // StartOffset values are local to each file, need to convert to global offsets
    std::vector<int64_t> all_lengths, all_offsets, all_tree_ids;
    all_lengths.reserve(header_.n_trees);
    all_offsets.reserve(header_.n_trees);
    all_tree_ids.reserve(header_.n_trees);

    int64_t halo_offset = 0;  // Cumulative halo count from previous files

    for (const auto& filepath : files_) {
        HighFive::File file(filepath, HighFive::File::ReadOnly);

        // Count halos in this file first (needed for offset calculation)
        int64_t halos_in_file = 0;
        if (file.exist("TreeHalos") && has_dataset(file.getGroup("TreeHalos"), "SnapNum")) {
            auto dims = file.getGroup("TreeHalos").getDataSet("SnapNum").getDimensions();
            halos_in_file = dims[0];
        }

        // Only some files have TreeTable (those containing tree roots)
        if (file.exist("TreeTable")) {
            auto table_group = file.getGroup("TreeTable");

            std::vector<int64_t> lengths, offsets, tree_ids;

            if (has_dataset(table_group, "Length")) {
                table_group.getDataSet("Length").read(lengths);
            }
            if (has_dataset(table_group, "StartOffset")) {
                table_group.getDataSet("StartOffset").read(offsets);
            }
            if (has_dataset(table_group, "TreeID")) {
                table_group.getDataSet("TreeID").read(tree_ids);
            }

            // Adjust offsets to global and append
            for (size_t i = 0; i < lengths.size(); i++) {
                all_lengths.push_back(lengths[i]);
                all_offsets.push_back(offsets[i] + halo_offset);
                all_tree_ids.push_back(i < tree_ids.size() ? tree_ids[i] :
                                       static_cast<int64_t>(all_lengths.size() - 1));
            }
        }

        halo_offset += halos_in_file;
    }

    // Build tree table entries
    size_t n_trees = all_lengths.size();
    tree_table_.resize(n_trees);

    for (size_t i = 0; i < n_trees; i++) {
        tree_table_[i].length = all_lengths[i];
        tree_table_[i].start_offset = all_offsets[i];
        tree_table_[i].tree_id = all_tree_ids[i];
    }

    table_loaded_ = true;
}

void MergerTree::load_search_data() {
    // Stage 1: Load only search-critical data (snap_num, subhalo_nr)
    // This is ~8 bytes/halo vs ~90 bytes/halo for full data
    if (search_loaded_) return;

    // Reserve space for all halos
    tree_halos_.snap_num.reserve(header_.n_halos);
    tree_halos_.subhalo_nr.reserve(header_.n_halos);

    // Read and concatenate from all files
    for (const auto& filepath : files_) {
        HighFive::File file(filepath, HighFive::File::ReadOnly);

        if (!file.exist("TreeHalos")) {
            throw std::runtime_error("File does not contain TreeHalos group: " + filepath);
        }

        auto halos_group = file.getGroup("TreeHalos");

        // Read and append snap_num
        if (has_dataset(halos_group, "SnapNum")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("SnapNum"));
            append_vector(tree_halos_.snap_num, data);
        }
        // Read and append subhalo_nr
        if (has_dataset(halos_group, "SubhaloNr")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("SubhaloNr"));
            append_vector(tree_halos_.subhalo_nr, data);
        }
    }

    search_loaded_ = true;
}

void MergerTree::load_full_halo_data() {
    // Stage 2: Load remaining halo data (positions, velocities, masses, tree links)
    // Called by make_halo_info() and tree traversal functions
    if (halos_loaded_) return;

    // Ensure search data is loaded first
    load_search_data();

    // Reserve space
    tree_halos_.group_nr.reserve(header_.n_halos);
    tree_halos_.tree_id.reserve(header_.n_halos);
    tree_halos_.tree_index.reserve(header_.n_halos);
    tree_halos_.pos_x.reserve(header_.n_halos);
    tree_halos_.pos_y.reserve(header_.n_halos);
    tree_halos_.pos_z.reserve(header_.n_halos);
    tree_halos_.vel_x.reserve(header_.n_halos);
    tree_halos_.vel_y.reserve(header_.n_halos);
    tree_halos_.vel_z.reserve(header_.n_halos);
    tree_halos_.mass.reserve(header_.n_halos);
    tree_halos_.m200c.reserve(header_.n_halos);
    tree_halos_.r200c.reserve(header_.n_halos);
    tree_halos_.main_progenitor.reserve(header_.n_halos);
    tree_halos_.descendant.reserve(header_.n_halos);
    tree_halos_.first_progenitor.reserve(header_.n_halos);
    tree_halos_.next_progenitor.reserve(header_.n_halos);

    // Read and concatenate from all files
    for (const auto& filepath : files_) {
        HighFive::File file(filepath, HighFive::File::ReadOnly);
        auto halos_group = file.getGroup("TreeHalos");

        // Read remaining identification arrays
        if (has_dataset(halos_group, "GroupNr")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("GroupNr"));
            append_vector(tree_halos_.group_nr, data);
        }
        if (has_dataset(halos_group, "TreeID")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeID"));
            append_vector(tree_halos_.tree_id, data);
        }
        if (has_dataset(halos_group, "TreeIndex")) {
            auto data = read_1d_dataset<int64_t>(halos_group.getDataSet("TreeIndex"));
            append_vector(tree_halos_.tree_index, data);
        }

        // Read positions (split 2D array into SOA)
        if (has_dataset(halos_group, "SubhaloPos")) {
            std::vector<float> px, py, pz;
            read_2d_dataset_split<float>(halos_group.getDataSet("SubhaloPos"), px, py, pz);
            append_vector(tree_halos_.pos_x, px);
            append_vector(tree_halos_.pos_y, py);
            append_vector(tree_halos_.pos_z, pz);
        }

        // Read velocities
        if (has_dataset(halos_group, "SubhaloVel")) {
            std::vector<float> vx, vy, vz;
            read_2d_dataset_split<float>(halos_group.getDataSet("SubhaloVel"), vx, vy, vz);
            append_vector(tree_halos_.vel_x, vx);
            append_vector(tree_halos_.vel_y, vy);
            append_vector(tree_halos_.vel_z, vz);
        }

        // Read masses
        if (has_dataset(halos_group, "SubhaloMass")) {
            auto data = read_1d_dataset<float>(halos_group.getDataSet("SubhaloMass"));
            append_vector(tree_halos_.mass, data);
        }
        if (has_dataset(halos_group, "Group_M_Crit200")) {
            auto data = read_1d_dataset<float>(halos_group.getDataSet("Group_M_Crit200"));
            append_vector(tree_halos_.m200c, data);
        }
        if (has_dataset(halos_group, "Group_R_Crit200")) {
            auto data = read_1d_dataset<float>(halos_group.getDataSet("Group_R_Crit200"));
            append_vector(tree_halos_.r200c, data);
        }

        // Read tree links (relative indices within tree)
        if (has_dataset(halos_group, "TreeMainProgenitor")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeMainProgenitor"));
            append_vector(tree_halos_.main_progenitor, data);
        }
        if (has_dataset(halos_group, "TreeDescendant")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeDescendant"));
            append_vector(tree_halos_.descendant, data);
        }
        if (has_dataset(halos_group, "TreeFirstProgenitor")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeFirstProgenitor"));
            append_vector(tree_halos_.first_progenitor, data);
        }
        if (has_dataset(halos_group, "TreeNextProgenitor")) {
            auto data = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeNextProgenitor"));
            append_vector(tree_halos_.next_progenitor, data);
        }
    }

    halos_loaded_ = true;
}

void MergerTree::load_tree_halos() {
    // Legacy function: Load all data at once (calls both stages)
    if (halos_loaded_) return;
    load_search_data();
    load_full_halo_data();
}

int32_t MergerTree::find_nearest_snapshot(float scale_factor) const {
    if (header_.snap_times.empty()) {
        return -1;
    }

    float min_diff = std::abs(header_.snap_times[0] - scale_factor);
    int32_t best_snap = 0;

    for (size_t i = 1; i < header_.snap_times.size(); i++) {
        float diff = std::abs(header_.snap_times[i] - scale_factor);
        if (diff < min_diff) {
            min_diff = diff;
            best_snap = static_cast<int32_t>(i);
        }
    }

    return best_snap;
}

const TreeTableEntry& MergerTree::get_tree_entry(int64_t tree_id) {
    load_tree_table();

    if (tree_id < 0 || static_cast<size_t>(tree_id) >= tree_table_.size()) {
        throw std::out_of_range("tree_id out of range");
    }

    return tree_table_[tree_id];
}

std::pair<int64_t, int64_t> MergerTree::get_tree_indices(int64_t tree_id) {
    const auto& entry = get_tree_entry(tree_id);
    return {entry.start_offset, entry.start_offset + entry.length};
}

int64_t MergerTree::get_root_halo_index(int64_t tree_id) {
    const auto& entry = get_tree_entry(tree_id);
    return entry.start_offset;
}

const TreeHalosSOA& MergerTree::tree_halos() {
    // Callers expect full data access, so load everything
    load_full_halo_data();
    return tree_halos_;
}

std::optional<int64_t> MergerTree::find_halo_in_tree(
    int32_t snap_num,
    int32_t subhalo_nr,
    std::optional<int64_t> tree_id
) {
    // Only load search-critical data (Stage 1) for fast first call
    load_search_data();

    int64_t start_idx, end_idx;

    if (tree_id.has_value()) {
        // Search only within specified tree
        load_tree_table();
        auto [start, end] = get_tree_indices(*tree_id);
        start_idx = start;
        end_idx = end;
    } else {
        // Search all halos
        start_idx = 0;
        end_idx = static_cast<int64_t>(tree_halos_.size());
    }

#if TESSERA_HAS_AVX2
    int64_t result = find_halo_simd(snap_num, subhalo_nr, start_idx, end_idx);
#else
    int64_t result = find_halo_scalar(snap_num, subhalo_nr, start_idx, end_idx);
#endif

    if (result < 0) {
        return std::nullopt;
    }
    return result;
}

// Scalar fallback implementation
int64_t MergerTree::find_halo_scalar(
    int32_t snap_num,
    int32_t subhalo_nr,
    int64_t start_idx,
    int64_t end_idx
) {
    const int32_t* snap_data = tree_halos_.snap_num.data();
    const int32_t* sub_data = tree_halos_.subhalo_nr.data();

    for (int64_t i = start_idx; i < end_idx; i++) {
        if (snap_data[i] == snap_num && sub_data[i] == subhalo_nr) {
            return i;
        }
    }

    return -1;
}

#if TESSERA_HAS_AVX2
// AVX2 SIMD implementation - compares 8 int32s per cycle
int64_t MergerTree::find_halo_simd(
    int32_t snap_num,
    int32_t subhalo_nr,
    int64_t start_idx,
    int64_t end_idx
) {
    const int32_t* snap_data = tree_halos_.snap_num.data();
    const int32_t* sub_data = tree_halos_.subhalo_nr.data();

    // Create broadcast vectors for comparison targets
    __m256i snap_target = _mm256_set1_epi32(snap_num);
    __m256i sub_target = _mm256_set1_epi32(subhalo_nr);

    // Process 8 elements at a time
    int64_t i = start_idx;
    int64_t simd_end = start_idx + ((end_idx - start_idx) & ~7);  // Round down to multiple of 8

    for (; i < simd_end; i += 8) {
        // Load 8 snapshot numbers and 8 subhalo numbers
        __m256i snap_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(snap_data + i));
        __m256i sub_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub_data + i));

        // Compare: snap_vec == snap_target AND sub_vec == sub_target
        __m256i snap_match = _mm256_cmpeq_epi32(snap_vec, snap_target);
        __m256i sub_match = _mm256_cmpeq_epi32(sub_vec, sub_target);
        __m256i both_match = _mm256_and_si256(snap_match, sub_match);

        // Check if any matches
        int mask = _mm256_movemask_epi8(both_match);
        if (mask != 0) {
            // Find the first match
            // Each int32 produces 4 bytes in the mask, so divide by 4
            int byte_offset = __builtin_ctz(mask);  // Count trailing zeros
            int element_offset = byte_offset / 4;   // 4 bytes per int32
            return i + element_offset;
        }
    }

    // Handle remaining elements with scalar code
    return find_halo_scalar(snap_num, subhalo_nr, i, end_idx);
}
#else
// Non-AVX2 systems just use scalar
int64_t MergerTree::find_halo_simd(
    int32_t snap_num,
    int32_t subhalo_nr,
    int64_t start_idx,
    int64_t end_idx
) {
    return find_halo_scalar(snap_num, subhalo_nr, start_idx, end_idx);
}
#endif

HaloInfo MergerTree::make_halo_info(int64_t tree_index) {
    // Need full data for positions, velocities, masses, etc.
    load_full_halo_data();

    if (tree_index < 0 || static_cast<size_t>(tree_index) >= tree_halos_.size()) {
        return HaloInfo{};  // Return invalid HaloInfo
    }

    HaloInfo info;
    info.tree_index = tree_index;

    const auto& h = tree_halos_;
    size_t idx = static_cast<size_t>(tree_index);

    info.snap_num = h.snap_num[idx];
    info.subhalo_nr = h.subhalo_nr[idx];
    info.group_nr = h.group_nr[idx];

    info.scale_factor = get_scale_factor(info.snap_num);
    info.redshift = get_redshift(info.snap_num);

    info.position = {h.pos_x[idx], h.pos_y[idx], h.pos_z[idx]};
    info.velocity = {h.vel_x[idx], h.vel_y[idx], h.vel_z[idx]};

    info.mass = h.mass.empty() ? 0.0f : h.mass[idx];
    info.m200c = h.m200c.empty() ? 0.0f : h.m200c[idx];
    info.r200c = h.r200c.empty() ? 0.0f : h.r200c[idx];

    return info;
}

void MergerTree::clear_cache() {
    tree_table_.clear();
    tree_table_.shrink_to_fit();
    tree_halos_.clear();
    table_loaded_ = false;
    search_loaded_ = false;
    halos_loaded_ = false;
    range_loaded_ = false;
    loaded_range_start_ = 0;
    loaded_range_end_ = 0;
}

void MergerTree::clear_halos_cache() {
    tree_halos_.clear();
    search_loaded_ = false;
    halos_loaded_ = false;
    range_loaded_ = false;
    loaded_range_start_ = 0;
    loaded_range_end_ = 0;
}

void MergerTree::load_tree_halos_range(int64_t start_idx, int64_t end_idx) {
    // Range loading with split files is not yet supported
    if (files_.size() > 1) {
        throw std::runtime_error(
            "load_tree_halos_range() does not support split merger tree files. "
            "Use load_search_data() and load_full_halo_data() instead."
        );
    }

    // Clear any existing data
    clear_halos_cache();

    if (start_idx >= end_idx || start_idx < 0) {
        return;
    }

    HighFive::File file(files_[0], HighFive::File::ReadOnly);

    if (!file.exist("TreeHalos")) {
        throw std::runtime_error("File does not contain TreeHalos group: " + files_[0]);
    }

    auto halos_group = file.getGroup("TreeHalos");
    size_t count = static_cast<size_t>(end_idx - start_idx);

    // Helper lambda to read a slice of a 1D dataset
    auto read_slice_1d = [&](const std::string& name, auto& vec) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        if (has_dataset(halos_group, name)) {
            auto ds = halos_group.getDataSet(name);
            vec.resize(count);
            ds.select({static_cast<size_t>(start_idx)}, {count}).read(vec.data());
        }
    };

    // Helper lambda to read a slice of a 2D dataset into separate x,y,z vectors
    auto read_slice_2d = [&](const std::string& name,
                             std::vector<float>& x,
                             std::vector<float>& y,
                             std::vector<float>& z) {
        if (has_dataset(halos_group, name)) {
            auto ds = halos_group.getDataSet(name);
            std::vector<std::array<float, 3>> raw(count);
            ds.select({static_cast<size_t>(start_idx), 0}, {count, 3}).read(raw.data());

            x.resize(count);
            y.resize(count);
            z.resize(count);
            for (size_t i = 0; i < count; i++) {
                x[i] = raw[i][0];
                y[i] = raw[i][1];
                z[i] = raw[i][2];
            }
        }
    };

    // Read all fields
    read_slice_1d("SnapNum", tree_halos_.snap_num);
    read_slice_1d("SubhaloNr", tree_halos_.subhalo_nr);
    read_slice_1d("GroupNr", tree_halos_.group_nr);
    read_slice_1d("TreeID", tree_halos_.tree_id);
    read_slice_1d("SubhaloMass", tree_halos_.mass);
    read_slice_1d("Group_M_Crit200", tree_halos_.m200c);
    read_slice_1d("Group_R_Crit200", tree_halos_.r200c);
    read_slice_1d("TreeMainProgenitor", tree_halos_.main_progenitor);
    read_slice_1d("TreeDescendant", tree_halos_.descendant);
    read_slice_1d("TreeFirstProgenitor", tree_halos_.first_progenitor);
    read_slice_1d("TreeNextProgenitor", tree_halos_.next_progenitor);

    read_slice_2d("SubhaloPos", tree_halos_.pos_x, tree_halos_.pos_y, tree_halos_.pos_z);
    read_slice_2d("SubhaloVel", tree_halos_.vel_x, tree_halos_.vel_y, tree_halos_.vel_z);

    // Mark as range-loaded
    range_loaded_ = true;
    loaded_range_start_ = start_idx;
    loaded_range_end_ = end_idx;
    halos_loaded_ = false;  // Not fully loaded
    search_loaded_ = false;
}

size_t MergerTree::memory_usage() const {
    size_t total = 0;

    // Header (fixed size, small)
    total += sizeof(MergerTreeHeader);
    total += header_.snap_times.capacity() * sizeof(float);
    total += header_.snap_redshifts.capacity() * sizeof(float);

    // Tree table
    total += tree_table_.capacity() * sizeof(TreeTableEntry);

    // Tree halos (each vector)
    total += tree_halos_.snap_num.capacity() * sizeof(int32_t);
    total += tree_halos_.subhalo_nr.capacity() * sizeof(int32_t);
    total += tree_halos_.group_nr.capacity() * sizeof(int32_t);
    total += tree_halos_.tree_id.capacity() * sizeof(int32_t);
    total += tree_halos_.tree_index.capacity() * sizeof(int64_t);

    total += tree_halos_.pos_x.capacity() * sizeof(float);
    total += tree_halos_.pos_y.capacity() * sizeof(float);
    total += tree_halos_.pos_z.capacity() * sizeof(float);
    total += tree_halos_.vel_x.capacity() * sizeof(float);
    total += tree_halos_.vel_y.capacity() * sizeof(float);
    total += tree_halos_.vel_z.capacity() * sizeof(float);

    total += tree_halos_.mass.capacity() * sizeof(float);
    total += tree_halos_.m200c.capacity() * sizeof(float);
    total += tree_halos_.r200c.capacity() * sizeof(float);

    total += tree_halos_.main_progenitor.capacity() * sizeof(int32_t);
    total += tree_halos_.descendant.capacity() * sizeof(int32_t);
    total += tree_halos_.first_progenitor.capacity() * sizeof(int32_t);
    total += tree_halos_.next_progenitor.capacity() * sizeof(int32_t);

    return total;
}

// =============================================================================
// HaloTracker implementation
// =============================================================================

HaloTracker::HaloTracker(MergerTree& tree) : tree_(tree) {}

int64_t HaloTracker::get_tree_start(int64_t absolute_idx) {
    const auto& halos = tree_.tree_halos();
    int32_t tree_id = halos.tree_id[absolute_idx];
    return tree_.get_tree_entry(tree_id).start_offset;
}

std::vector<HaloInfo> HaloTracker::trace_main_branch(
    int32_t snap_num,
    int32_t subhalo_nr,
    bool backward,
    bool forward,
    std::optional<float> a_min,
    std::optional<float> a_max
) {
    // Find starting halo
    auto start_idx_opt = tree_.find_halo_in_tree(snap_num, subhalo_nr);
    if (!start_idx_opt) {
        return {};
    }

    int64_t start_idx = *start_idx_opt;
    const auto& halos = tree_.tree_halos();
    int64_t tree_start = get_tree_start(start_idx);

    std::vector<int64_t> branch_indices;
    branch_indices.push_back(start_idx);

    // Trace backward (progenitors) with prefetching
    if (backward) {
        int64_t current = start_idx;
        const int32_t* main_prog = halos.main_progenitor.data();
        const int32_t* snap_data = halos.snap_num.data();

        while (true) {
            int32_t prog_rel = main_prog[current];
            int64_t prog_idx = MergerTree::get_absolute_index(prog_rel, tree_start);

            if (prog_idx == -1) break;

            // Prefetch next progenitor for better cache behavior
            if (prog_idx >= 0 && static_cast<size_t>(prog_idx) < halos.size()) {
                __builtin_prefetch(&main_prog[prog_idx], 0, 3);
                __builtin_prefetch(&snap_data[prog_idx], 0, 3);
            }

            // Check scale factor limit
            if (a_min.has_value()) {
                float prog_a = tree_.get_scale_factor(snap_data[prog_idx]);
                if (prog_a < *a_min) break;
            }

            branch_indices.push_back(prog_idx);
            current = prog_idx;
        }
    }

    // Trace forward (descendants) with prefetching
    if (forward) {
        int64_t current = start_idx;
        const int32_t* desc = halos.descendant.data();
        const int32_t* snap_data = halos.snap_num.data();

        while (true) {
            int32_t desc_rel = desc[current];
            int64_t desc_idx = MergerTree::get_absolute_index(desc_rel, tree_start);

            if (desc_idx == -1) break;

            // Prefetch next descendant
            if (desc_idx >= 0 && static_cast<size_t>(desc_idx) < halos.size()) {
                __builtin_prefetch(&desc[desc_idx], 0, 3);
                __builtin_prefetch(&snap_data[desc_idx], 0, 3);
            }

            // Check scale factor limit
            if (a_max.has_value()) {
                float desc_a = tree_.get_scale_factor(snap_data[desc_idx]);
                if (desc_a > *a_max) break;
            }

            branch_indices.push_back(desc_idx);
            current = desc_idx;
        }
    }

    // Build HaloInfo objects
    std::vector<HaloInfo> branch;
    branch.reserve(branch_indices.size());

    for (int64_t idx : branch_indices) {
        branch.push_back(tree_.make_halo_info(idx));
    }

    // Sort by scale factor (late to early, root-first order)
    // This is natural for tree-walking algorithms that start from z=0
    std::sort(branch.begin(), branch.end(),
              [](const HaloInfo& a, const HaloInfo& b) {
                  return a.scale_factor > b.scale_factor;
              });

    return branch;
}

std::vector<HaloInfo> HaloTracker::trace_from_tree_id(
    int64_t tree_id,
    bool backward,
    bool forward,
    std::optional<float> a_min,
    std::optional<float> a_max
) {
    int64_t root_idx = tree_.get_root_halo_index(tree_id);
    const auto& halos = tree_.tree_halos();

    int32_t snap_num = halos.snap_num[root_idx];
    int32_t subhalo_nr = halos.subhalo_nr[root_idx];

    return trace_main_branch(snap_num, subhalo_nr, backward, forward, a_min, a_max);
}

std::unordered_map<int32_t, std::vector<HaloInfo>> HaloTracker::trace_all_progenitors(
    int32_t snap_num,
    int32_t subhalo_nr,
    std::optional<float> a_min
) {
    auto start_idx_opt = tree_.find_halo_in_tree(snap_num, subhalo_nr);
    if (!start_idx_opt) {
        return {};
    }

    int64_t start_idx = *start_idx_opt;
    const auto& halos = tree_.tree_halos();
    int64_t tree_start = get_tree_start(start_idx);

    std::unordered_map<int32_t, std::vector<HaloInfo>> progenitors;

    // BFS through all progenitors
    std::queue<int64_t> to_visit;
    std::vector<bool> visited(halos.size(), false);

    to_visit.push(start_idx);

    const int32_t* first_prog = halos.first_progenitor.data();
    const int32_t* next_prog = halos.next_progenitor.data();
    const int32_t* snap_data = halos.snap_num.data();

    while (!to_visit.empty()) {
        int64_t current = to_visit.front();
        to_visit.pop();

        if (current < 0 || static_cast<size_t>(current) >= halos.size()) continue;
        if (visited[current]) continue;
        visited[current] = true;

        // Check scale factor limit
        float current_a = tree_.get_scale_factor(snap_data[current]);
        if (a_min.has_value() && current_a < *a_min) continue;

        // Add to result
        HaloInfo info = tree_.make_halo_info(current);
        progenitors[info.snap_num].push_back(info);

        // Add all progenitors to visit list
        int32_t prog_rel = first_prog[current];
        int64_t prog = MergerTree::get_absolute_index(prog_rel, tree_start);

        while (prog != -1) {
            to_visit.push(prog);

            if (prog >= 0 && static_cast<size_t>(prog) < halos.size()) {
                prog_rel = next_prog[prog];
                prog = MergerTree::get_absolute_index(prog_rel, tree_start);
            } else {
                break;
            }
        }
    }

    return progenitors;
}

void HaloTracker::unwrap_coordinates(
    std::vector<HaloInfo>& branch,
    std::optional<double> box_size
) {
    if (branch.size() < 2) return;

    double box = box_size.value_or(tree_.box_size());
    float half_box = static_cast<float>(box / 2.0);
    float box_f = static_cast<float>(box);

    for (size_t i = 1; i < branch.size(); i++) {
        for (int dim = 0; dim < 3; dim++) {
            float diff = branch[i].position[dim] - branch[i - 1].position[dim];

            if (diff > half_box) {
                branch[i].position[dim] -= box_f;
            } else if (diff < -half_box) {
                branch[i].position[dim] += box_f;
            }
        }
    }
}

void HaloTracker::unwrap_coordinates_static(
    std::vector<std::array<float, 3>>& positions,
    float box_size
) {
    if (positions.size() < 2) return;

    float half_box = box_size / 2.0f;

    for (size_t i = 1; i < positions.size(); i++) {
        for (int dim = 0; dim < 3; dim++) {
            float diff = positions[i][dim] - positions[i - 1][dim];

            if (diff > half_box) {
                // Apply correction to all subsequent positions
                for (size_t j = i; j < positions.size(); j++) {
                    positions[j][dim] -= box_size;
                }
            } else if (diff < -half_box) {
                for (size_t j = i; j < positions.size(); j++) {
                    positions[j][dim] += box_size;
                }
            }
        }
    }
}

// =============================================================================
// Free functions
// =============================================================================

MergerTreeHeader read_merger_tree_header(const std::string& filename) {
    MergerTree tree(filename);
    return tree.header();
}

// =============================================================================
// Parallel Branch Extraction
// =============================================================================

namespace {

/**
 * Thread-local branch data for parallel extraction.
 */
struct ThreadLocalBranch {
    std::vector<int32_t> snap_num;
    std::vector<int32_t> subhalo_nr;
    std::vector<float> scale_factor;
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> m200c;
    std::vector<float> r200c;
    std::vector<float> mass;

    void reserve(size_t n) {
        snap_num.reserve(n);
        subhalo_nr.reserve(n);
        scale_factor.reserve(n);
        pos_x.reserve(n);
        pos_y.reserve(n);
        pos_z.reserve(n);
        vel_x.reserve(n);
        vel_y.reserve(n);
        vel_z.reserve(n);
        m200c.reserve(n);
        r200c.reserve(n);
        mass.reserve(n);
    }

    void clear() {
        snap_num.clear();
        subhalo_nr.clear();
        scale_factor.clear();
        pos_x.clear();
        pos_y.clear();
        pos_z.clear();
        vel_x.clear();
        vel_y.clear();
        vel_z.clear();
        m200c.clear();
        r200c.clear();
        mass.clear();
    }

    size_t size() const { return snap_num.size(); }
};

/**
 * Trace main branch backward from root halo (internal, no locking needed).
 * Returns branch data sorted by scale factor (early to late).
 */
void trace_branch_internal(
    const TreeHalosSOA& halos,
    const MergerTreeHeader& header,
    int64_t root_idx,
    int64_t tree_start,
    std::optional<float> a_min,
    ThreadLocalBranch& out
) {
    out.clear();

    // Collect indices backward from root
    std::vector<int64_t> branch_indices;
    branch_indices.reserve(128);  // Most branches < 128 halos

    int64_t current = root_idx;
    const int32_t* main_prog = halos.main_progenitor.data();
    const int32_t* snap_data = halos.snap_num.data();

    branch_indices.push_back(current);

    while (true) {
        int32_t prog_rel = main_prog[current];
        int64_t prog_idx = MergerTree::get_absolute_index(prog_rel, tree_start);

        if (prog_idx == -1) break;

        // Check scale factor limit
        if (a_min.has_value()) {
            float prog_a = header.get_scale_factor(snap_data[prog_idx]);
            if (prog_a < *a_min) break;
        }

        branch_indices.push_back(prog_idx);
        current = prog_idx;
    }

    // Sort by scale factor (late to early, root-first order)
    // This is natural for tree-walking algorithms that start from z=0
    std::sort(branch_indices.begin(), branch_indices.end(),
              [&snap_data, &header](int64_t a, int64_t b) {
                  return header.get_scale_factor(snap_data[a]) >
                         header.get_scale_factor(snap_data[b]);
              });

    // Copy data to output
    out.reserve(branch_indices.size());
    for (int64_t idx : branch_indices) {
        out.snap_num.push_back(halos.snap_num[idx]);
        out.subhalo_nr.push_back(halos.subhalo_nr[idx]);
        out.scale_factor.push_back(header.get_scale_factor(halos.snap_num[idx]));
        out.pos_x.push_back(halos.pos_x[idx]);
        out.pos_y.push_back(halos.pos_y[idx]);
        out.pos_z.push_back(halos.pos_z[idx]);
        out.vel_x.push_back(halos.vel_x[idx]);
        out.vel_y.push_back(halos.vel_y[idx]);
        out.vel_z.push_back(halos.vel_z[idx]);
        out.m200c.push_back(halos.m200c.empty() ? 0.0f : halos.m200c[idx]);
        out.r200c.push_back(halos.r200c.empty() ? 0.0f : halos.r200c[idx]);
        out.mass.push_back(halos.mass[idx]);
    }
}

} // anonymous namespace

// Internal struct for tree metadata
struct TreeInfoInternal {
    int32_t tree_id;
    int64_t root_idx;
    int64_t start_offset;  // Start in TreeHalos
    int64_t length;        // Number of halos in tree
    float root_m200c;
    float root_r200c;
    int32_t root_snap;
    int32_t root_subhalo;
    size_t output_idx;     // Index in output (for re-sorting after chunked processing)
};

/**
 * Internal function for chunked (low-memory) branch extraction.
 *
 * This function processes trees in chunks based on their HDF5 index ranges,
 * loading only the required halo data for each chunk. This uses significantly
 * less memory than loading all halos at once, at the cost of some performance
 * due to repeated I/O and less efficient parallelization.
 *
 * Algorithm:
 * 1. Sort trees by their end index (start_offset + length) to group overlapping ranges
 * 2. Build chunks where each chunk's halo range stays within chunk_size
 * 3. For each chunk: load range, extract branches in parallel, store results
 * 4. Merge all results
 */
BranchCatalogData extract_branches_chunked(
    MergerTree& tree,
    std::vector<TreeInfoInternal>& filtered_trees,
    std::optional<float> a_min,
    int n_threads,
    size_t chunk_size,
    std::function<void(size_t, size_t)> progress_callback
) {
    const auto& header = tree.header();
    size_t n_trees = filtered_trees.size();

    // Set number of threads
#ifdef _OPENMP
    int actual_threads = n_threads > 0 ? n_threads : omp_get_max_threads();
    omp_set_num_threads(actual_threads);
#else
    int actual_threads = 1;
    (void)n_threads;
#endif

    // Sort trees by end index to process in order (better for chunking)
    // Keep track of original output order
    for (size_t i = 0; i < n_trees; i++) {
        filtered_trees[i].output_idx = i;  // Preserve M200c-sorted order
    }

    // Create a working copy sorted by halo index range for efficient chunking
    std::vector<TreeInfoInternal> trees_by_range = filtered_trees;
    std::sort(trees_by_range.begin(), trees_by_range.end(),
              [](const TreeInfoInternal& a, const TreeInfoInternal& b) {
                  return a.start_offset < b.start_offset;
              });

    // Results storage - indexed by output_idx
    std::vector<ThreadLocalBranch> all_branches(n_trees);
    size_t trees_processed = 0;

    // Process trees in chunks based on halo index range
    size_t tree_idx = 0;
    while (tree_idx < n_trees) {

        // Find the range of trees for this chunk
        int64_t chunk_start = trees_by_range[tree_idx].start_offset;
        int64_t chunk_end = trees_by_range[tree_idx].start_offset +
                           trees_by_range[tree_idx].length;

        std::vector<size_t> chunk_trees;
        chunk_trees.push_back(tree_idx);

        // Add more trees to this chunk while they fit
        while (tree_idx + chunk_trees.size() < n_trees) {
            size_t next_idx = tree_idx + chunk_trees.size();
            const auto& next_tree = trees_by_range[next_idx];
            int64_t next_end = next_tree.start_offset + next_tree.length;

            // Check if adding this tree would exceed chunk_size
            int64_t new_range = std::max(chunk_end, next_end) - chunk_start;
            if (static_cast<size_t>(new_range) > chunk_size) {
                break;
            }

            chunk_trees.push_back(next_idx);
            chunk_end = std::max(chunk_end, next_end);
        }

        // Load the halo data for this chunk's range
        tree.load_tree_halos_range(chunk_start, chunk_end);
        // Use tree_halos_loaded() to avoid triggering full reload
        const auto& halos = tree.tree_halos_loaded();

        // Parallel extraction for trees in this chunk
        size_t chunk_n_trees = chunk_trees.size();

        #pragma omp parallel
        {
            ThreadLocalBranch local_branch;
            local_branch.reserve(128);

            #pragma omp for schedule(dynamic, 16)
            for (size_t i = 0; i < chunk_n_trees; i++) {
                const auto& info = trees_by_range[chunk_trees[i]];

                // For range-loaded data, indices need to be adjusted
                int64_t range_offset = tree.loaded_range().first;

                // Trace branch - need to adjust indices for the loaded range
                int64_t root_idx_in_range = info.root_idx - range_offset;

                local_branch.clear();

                // Manual branch tracing with range-adjusted indices
                const auto& main_prog = halos.main_progenitor;
                const auto& snap_data = halos.snap_num;

                std::vector<int64_t> branch_indices;
                branch_indices.reserve(128);

                int64_t current = root_idx_in_range;
                branch_indices.push_back(current);

                while (true) {
                    int32_t prog_rel = main_prog[current];
                    // Convert relative->absolute, then adjust for range
                    int64_t prog_abs = (prog_rel == -1) ? -1 :
                        (info.start_offset + prog_rel);
                    int64_t prog_idx = (prog_abs == -1) ? -1 :
                        (prog_abs - range_offset);

                    if (prog_idx < 0 ||
                        prog_abs < tree.loaded_range().first ||
                        prog_abs >= tree.loaded_range().second) {
                        break;
                    }

                    // Check scale factor limit
                    if (a_min.has_value()) {
                        float prog_a = header.get_scale_factor(snap_data[prog_idx]);
                        if (prog_a < *a_min) break;
                    }

                    branch_indices.push_back(prog_idx);
                    current = prog_idx;
                }

                // Sort by scale factor (late to early, root-first order)
                std::sort(branch_indices.begin(), branch_indices.end(),
                          [&snap_data, &header](int64_t a, int64_t b) {
                              return header.get_scale_factor(snap_data[a]) >
                                     header.get_scale_factor(snap_data[b]);
                          });

                // Copy data to local branch
                for (int64_t idx : branch_indices) {
                    local_branch.snap_num.push_back(halos.snap_num[idx]);
                    local_branch.subhalo_nr.push_back(halos.subhalo_nr[idx]);
                    local_branch.scale_factor.push_back(
                        header.get_scale_factor(halos.snap_num[idx]));
                    local_branch.pos_x.push_back(halos.pos_x[idx]);
                    local_branch.pos_y.push_back(halos.pos_y[idx]);
                    local_branch.pos_z.push_back(halos.pos_z[idx]);
                    local_branch.vel_x.push_back(halos.vel_x[idx]);
                    local_branch.vel_y.push_back(halos.vel_y[idx]);
                    local_branch.vel_z.push_back(halos.vel_z[idx]);
                    local_branch.m200c.push_back(
                        halos.m200c.empty() ? 0.0f : halos.m200c[idx]);
                    local_branch.r200c.push_back(
                        halos.r200c.empty() ? 0.0f : halos.r200c[idx]);
                    local_branch.mass.push_back(halos.mass[idx]);
                }

                // Store in output position (maintaining M200c sort order)
                #pragma omp critical
                {
                    all_branches[info.output_idx] = std::move(local_branch);
                }
                local_branch.clear();
            }
        }

        // Update progress
        trees_processed += chunk_trees.size();
        if (progress_callback) {
            progress_callback(trees_processed, n_trees);
        }

        tree_idx += chunk_trees.size();
    }

    // Clear the halos cache after all chunks are processed
    tree.clear_halos_cache();

    // Build final result
    BranchCatalogData result;
    std::vector<int32_t> branch_lengths(n_trees);
    size_t total_halos = 0;

    for (size_t i = 0; i < n_trees; i++) {
        branch_lengths[i] = static_cast<int32_t>(all_branches[i].size());
        total_halos += all_branches[i].size();
    }

    // Index arrays
    result.tree_id.resize(n_trees);
    result.start_offset.resize(n_trees);
    result.length.resize(n_trees);
    result.root_snap.resize(n_trees);
    result.root_subhalo.resize(n_trees);
    result.root_m200c.resize(n_trees);
    result.root_r200c.resize(n_trees);
    result.a_min.resize(n_trees);
    result.a_max.resize(n_trees);

    // Data arrays
    result.snap_num.resize(total_halos);
    result.subhalo_nr.resize(total_halos);
    result.scale_factor.resize(total_halos);
    result.pos_x.resize(total_halos);
    result.pos_y.resize(total_halos);
    result.pos_z.resize(total_halos);
    result.vel_x.resize(total_halos);
    result.vel_y.resize(total_halos);
    result.vel_z.resize(total_halos);
    result.m200c.resize(total_halos);
    result.r200c.resize(total_halos);
    result.mass.resize(total_halos);

    // Fill results using original M200c-sorted order
    int64_t current_offset = 0;
    for (size_t i = 0; i < n_trees; i++) {
        const auto& info = filtered_trees[i];  // Already in M200c order
        const auto& branch = all_branches[i];
        int32_t len = branch_lengths[i];

        result.tree_id[i] = info.tree_id;
        result.start_offset[i] = current_offset;
        result.length[i] = len;
        result.root_snap[i] = info.root_snap;
        result.root_subhalo[i] = info.root_subhalo;
        result.root_m200c[i] = info.root_m200c;
        result.root_r200c[i] = info.root_r200c;

        if (len == 0) {
            result.a_min[i] = 0.0f;
            result.a_max[i] = 0.0f;
        } else {
            // Root-first order: front is latest (a_max), back is earliest (a_min)
            result.a_max[i] = branch.scale_factor.front();
            result.a_min[i] = branch.scale_factor.back();

            // Copy data
            std::copy(branch.snap_num.begin(), branch.snap_num.end(),
                      result.snap_num.begin() + current_offset);
            std::copy(branch.subhalo_nr.begin(), branch.subhalo_nr.end(),
                      result.subhalo_nr.begin() + current_offset);
            std::copy(branch.scale_factor.begin(), branch.scale_factor.end(),
                      result.scale_factor.begin() + current_offset);
            std::copy(branch.pos_x.begin(), branch.pos_x.end(),
                      result.pos_x.begin() + current_offset);
            std::copy(branch.pos_y.begin(), branch.pos_y.end(),
                      result.pos_y.begin() + current_offset);
            std::copy(branch.pos_z.begin(), branch.pos_z.end(),
                      result.pos_z.begin() + current_offset);
            std::copy(branch.vel_x.begin(), branch.vel_x.end(),
                      result.vel_x.begin() + current_offset);
            std::copy(branch.vel_y.begin(), branch.vel_y.end(),
                      result.vel_y.begin() + current_offset);
            std::copy(branch.vel_z.begin(), branch.vel_z.end(),
                      result.vel_z.begin() + current_offset);
            std::copy(branch.m200c.begin(), branch.m200c.end(),
                      result.m200c.begin() + current_offset);
            std::copy(branch.r200c.begin(), branch.r200c.end(),
                      result.r200c.begin() + current_offset);
            std::copy(branch.mass.begin(), branch.mass.end(),
                      result.mass.begin() + current_offset);
        }

        current_offset += len;
    }

    return result;
}

BranchCatalogData extract_all_branches_parallel(
    MergerTree& tree,
    std::optional<float> mass_min,
    std::optional<float> a_min,
    int n_threads,
    size_t chunk_size,
    std::function<void(size_t, size_t)> progress_callback
) {
    // Debug to file

    const auto& header = tree.header();
    size_t total_trees = static_cast<size_t>(tree.n_trees());

    // Build list of tree metadata from TreeTable (small, always loaded)
    std::vector<TreeInfoInternal> tree_infos(total_trees);

    // For chunked mode, we only need TreeTable info initially
    // For non-chunked mode, we load all data upfront
    if (chunk_size == 0) {
        // Load all tree data (shared, read-only after this)
        const auto& halos = tree.tree_halos();  // Triggers full data load

        // Fill tree info from loaded data
        #pragma omp parallel for if(total_trees > 1000)
        for (size_t i = 0; i < total_trees; i++) {
            const auto& entry = tree.get_tree_entry(static_cast<int64_t>(i));
            int64_t root_idx = entry.start_offset;
            tree_infos[i].tree_id = static_cast<int32_t>(i);
            tree_infos[i].root_idx = root_idx;
            tree_infos[i].start_offset = entry.start_offset;
            tree_infos[i].length = entry.length;
            tree_infos[i].root_m200c = halos.m200c.empty() ? 0.0f : halos.m200c[root_idx];
            tree_infos[i].root_r200c = halos.r200c.empty() ? 0.0f : halos.r200c[root_idx];
            tree_infos[i].root_snap = halos.snap_num[root_idx];
            tree_infos[i].root_subhalo = halos.subhalo_nr[root_idx];
            tree_infos[i].output_idx = i;
        }
    } else {
        // Chunked mode: only load TreeTable, get root info from file later
        // First pass: get tree table info and read root halo data in chunks
        for (size_t i = 0; i < total_trees; i++) {
            const auto& entry = tree.get_tree_entry(static_cast<int64_t>(i));
            tree_infos[i].tree_id = static_cast<int32_t>(i);
            tree_infos[i].root_idx = entry.start_offset;
            tree_infos[i].start_offset = entry.start_offset;
            tree_infos[i].length = entry.length;
            tree_infos[i].output_idx = i;
            // M200c, snap, subhalo will be filled when we load root halo data
        }

        // Load root halo data to get M200c for sorting/filtering
        // Read in chunks of root indices to avoid loading everything
        const size_t root_chunk_size = 100000;
        for (size_t start = 0; start < total_trees; start += root_chunk_size) {
            size_t end = std::min(start + root_chunk_size, total_trees);

            // Find the range of root indices for this chunk
            int64_t min_idx = tree_infos[start].root_idx;
            int64_t max_idx = tree_infos[start].root_idx;
            for (size_t i = start; i < end; i++) {
                min_idx = std::min(min_idx, tree_infos[i].root_idx);
                max_idx = std::max(max_idx, tree_infos[i].root_idx);
            }

            // Load the range containing all root halos for this chunk
            tree.load_tree_halos_range(min_idx, max_idx + 1);
            const auto& halos = tree.tree_halos_loaded();

            // Fill root halo info
            for (size_t i = start; i < end; i++) {
                int64_t rel_idx = tree.to_range_index(tree_infos[i].root_idx);
                if (rel_idx >= 0) {
                    tree_infos[i].root_m200c = halos.m200c.empty() ? 0.0f : halos.m200c[rel_idx];
                    tree_infos[i].root_r200c = halos.r200c.empty() ? 0.0f : halos.r200c[rel_idx];
                    tree_infos[i].root_snap = halos.snap_num[rel_idx];
                    tree_infos[i].root_subhalo = halos.subhalo_nr[rel_idx];
                }
            }
        }
        tree.clear_halos_cache();
    }

    // Sort by M200c descending
    std::sort(tree_infos.begin(), tree_infos.end(),
              [](const TreeInfoInternal& a, const TreeInfoInternal& b) {
                  return a.root_m200c > b.root_m200c;
              });

    // Apply mass filter
    std::vector<TreeInfoInternal> filtered_trees;
    if (mass_min.has_value()) {
        filtered_trees.reserve(total_trees);
        for (const auto& info : tree_infos) {
            if (info.root_m200c >= *mass_min) {
                filtered_trees.push_back(info);
            }
        }
    } else {
        filtered_trees = std::move(tree_infos);
    }

    size_t n_trees = filtered_trees.size();
    if (n_trees == 0) {
        return BranchCatalogData{};
    }

    // For chunked mode, use streaming extraction
    if (chunk_size > 0) {
        return extract_branches_chunked(tree, filtered_trees, a_min, n_threads,
                                        chunk_size, progress_callback);
    }

    // Non-chunked mode: all data is loaded, extract in parallel
    const auto& halos = tree.tree_halos();  // Already loaded above

    // Set number of threads
#ifdef _OPENMP
    int actual_threads = n_threads > 0 ? n_threads : omp_get_max_threads();
    omp_set_num_threads(actual_threads);
#else
    int actual_threads = 1;
    (void)n_threads;  // Unused without OpenMP
#endif

    // Thread-local storage for branch extraction
    std::vector<std::vector<ThreadLocalBranch>> thread_branches(actual_threads);
    for (auto& tb : thread_branches) {
        tb.resize(n_trees);
    }

    // Pre-compute tree start offsets (shared read-only)
    std::vector<int64_t> tree_starts(n_trees);
    for (size_t i = 0; i < n_trees; i++) {
        int32_t tid = filtered_trees[i].tree_id;
        tree_starts[i] = tree.get_tree_entry(tid).start_offset;
    }

    // Progress tracking (callback is called outside parallel region)
    std::atomic<size_t> trees_done{0};

    // Parallel branch extraction
    #pragma omp parallel
    {
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
#else
        int thread_id = 0;
#endif
        ThreadLocalBranch local_branch;
        local_branch.reserve(128);

        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n_trees; i++) {
            const auto& info = filtered_trees[i];

            // Trace branch backward from root
            trace_branch_internal(
                halos, header,
                info.root_idx, tree_starts[i],
                a_min,
                local_branch
            );

            // Store in thread-local slot
            thread_branches[thread_id][i] = std::move(local_branch);
            local_branch.clear();

            // Increment counter (callback is handled outside parallel region)
            ++trees_done;
        }
    }

    // Call progress callback after parallel section completes (GIL-safe)
    if (progress_callback) {
        progress_callback(n_trees, n_trees);
    }

    // Merge results: compute total halos and offsets
    std::vector<int32_t> branch_lengths(n_trees);
    size_t total_halos = 0;

    for (size_t i = 0; i < n_trees; i++) {
        // Find which thread processed this tree
        size_t len = 0;
        for (int t = 0; t < actual_threads; t++) {
            if (!thread_branches[t][i].snap_num.empty()) {
                len = thread_branches[t][i].size();
                break;
            }
        }
        branch_lengths[i] = static_cast<int32_t>(len);
        total_halos += len;
    }

    // Build output
    BranchCatalogData result;

    // Index arrays
    result.tree_id.resize(n_trees);
    result.start_offset.resize(n_trees);
    result.length.resize(n_trees);
    result.root_snap.resize(n_trees);
    result.root_subhalo.resize(n_trees);
    result.root_m200c.resize(n_trees);
    result.root_r200c.resize(n_trees);
    result.a_min.resize(n_trees);
    result.a_max.resize(n_trees);

    // Data arrays
    result.snap_num.resize(total_halos);
    result.subhalo_nr.resize(total_halos);
    result.scale_factor.resize(total_halos);
    result.pos_x.resize(total_halos);
    result.pos_y.resize(total_halos);
    result.pos_z.resize(total_halos);
    result.vel_x.resize(total_halos);
    result.vel_y.resize(total_halos);
    result.vel_z.resize(total_halos);
    result.m200c.resize(total_halos);
    result.r200c.resize(total_halos);
    result.mass.resize(total_halos);

    // Fill index and copy data
    int64_t current_offset = 0;
    for (size_t i = 0; i < n_trees; i++) {
        const auto& info = filtered_trees[i];
        int32_t len = branch_lengths[i];

        result.tree_id[i] = info.tree_id;
        result.start_offset[i] = current_offset;
        result.length[i] = len;
        result.root_snap[i] = info.root_snap;
        result.root_subhalo[i] = info.root_subhalo;
        result.root_m200c[i] = info.root_m200c;
        result.root_r200c[i] = info.root_r200c;

        if (len == 0) {
            result.a_min[i] = 0.0f;
            result.a_max[i] = 0.0f;
        } else {
            // Find the thread that has this branch's data
            for (int t = 0; t < actual_threads; t++) {
                const auto& branch = thread_branches[t][i];
                if (!branch.snap_num.empty()) {
                    // Root-first order: front is latest (a_max), back is earliest (a_min)
                    result.a_max[i] = branch.scale_factor.front();
                    result.a_min[i] = branch.scale_factor.back();

                    // Copy data
                    std::copy(branch.snap_num.begin(), branch.snap_num.end(),
                              result.snap_num.begin() + current_offset);
                    std::copy(branch.subhalo_nr.begin(), branch.subhalo_nr.end(),
                              result.subhalo_nr.begin() + current_offset);
                    std::copy(branch.scale_factor.begin(), branch.scale_factor.end(),
                              result.scale_factor.begin() + current_offset);
                    std::copy(branch.pos_x.begin(), branch.pos_x.end(),
                              result.pos_x.begin() + current_offset);
                    std::copy(branch.pos_y.begin(), branch.pos_y.end(),
                              result.pos_y.begin() + current_offset);
                    std::copy(branch.pos_z.begin(), branch.pos_z.end(),
                              result.pos_z.begin() + current_offset);
                    std::copy(branch.vel_x.begin(), branch.vel_x.end(),
                              result.vel_x.begin() + current_offset);
                    std::copy(branch.vel_y.begin(), branch.vel_y.end(),
                              result.vel_y.begin() + current_offset);
                    std::copy(branch.vel_z.begin(), branch.vel_z.end(),
                              result.vel_z.begin() + current_offset);
                    std::copy(branch.m200c.begin(), branch.m200c.end(),
                              result.m200c.begin() + current_offset);
                    std::copy(branch.r200c.begin(), branch.r200c.end(),
                              result.r200c.begin() + current_offset);
                    std::copy(branch.mass.begin(), branch.mass.end(),
                              result.mass.begin() + current_offset);
                    break;
                }
            }
        }

        current_offset += len;
    }

    return result;
}

} // namespace io
} // namespace tessera
