#include "io/merger_tree.h"
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <queue>

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
    : filename_(filename) {
    load_header();
}

MergerTree::~MergerTree() = default;

MergerTree::MergerTree(MergerTree&&) noexcept = default;
MergerTree& MergerTree::operator=(MergerTree&&) noexcept = default;

void MergerTree::load_header() {
    HighFive::File file(filename_, HighFive::File::ReadOnly);

    if (!file.exist("Header")) {
        throw std::runtime_error("File does not contain Header group: " + filename_);
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

    HighFive::File file(filename_, HighFive::File::ReadOnly);

    if (!file.exist("TreeTable")) {
        throw std::runtime_error("File does not contain TreeTable group: " + filename_);
    }

    auto table_group = file.getGroup("TreeTable");

    // Read tree table arrays
    std::vector<int64_t> lengths, offsets, tree_ids;

    if (has_dataset(table_group, "Length")) {
        auto ds = table_group.getDataSet("Length");
        ds.read(lengths);
    }

    if (has_dataset(table_group, "StartOffset")) {
        auto ds = table_group.getDataSet("StartOffset");
        ds.read(offsets);
    }

    if (has_dataset(table_group, "TreeID")) {
        auto ds = table_group.getDataSet("TreeID");
        ds.read(tree_ids);
    }

    // Build tree table entries
    size_t n_trees = lengths.size();
    tree_table_.resize(n_trees);

    for (size_t i = 0; i < n_trees; i++) {
        tree_table_[i].length = i < lengths.size() ? lengths[i] : 0;
        tree_table_[i].start_offset = i < offsets.size() ? offsets[i] : 0;
        tree_table_[i].tree_id = i < tree_ids.size() ? tree_ids[i] : static_cast<int64_t>(i);
    }

    table_loaded_ = true;
}

void MergerTree::load_search_data() {
    // Stage 1: Load only search-critical data (snap_num, subhalo_nr)
    // This is ~8 bytes/halo vs ~90 bytes/halo for full data
    if (search_loaded_) return;

    HighFive::File file(filename_, HighFive::File::ReadOnly);

    if (!file.exist("TreeHalos")) {
        throw std::runtime_error("File does not contain TreeHalos group: " + filename_);
    }

    auto halos_group = file.getGroup("TreeHalos");

    // Read only the two arrays needed for SIMD search
    if (has_dataset(halos_group, "SnapNum")) {
        tree_halos_.snap_num = read_1d_dataset<int32_t>(halos_group.getDataSet("SnapNum"));
    }
    if (has_dataset(halos_group, "SubhaloNr")) {
        tree_halos_.subhalo_nr = read_1d_dataset<int32_t>(halos_group.getDataSet("SubhaloNr"));
    }

    search_loaded_ = true;
}

void MergerTree::load_full_halo_data() {
    // Stage 2: Load remaining halo data (positions, velocities, masses, tree links)
    // Called by make_halo_info() and tree traversal functions
    if (halos_loaded_) return;

    // Ensure search data is loaded first
    load_search_data();

    HighFive::File file(filename_, HighFive::File::ReadOnly);
    auto halos_group = file.getGroup("TreeHalos");

    // Read remaining identification arrays
    if (has_dataset(halos_group, "GroupNr")) {
        tree_halos_.group_nr = read_1d_dataset<int32_t>(halos_group.getDataSet("GroupNr"));
    }
    if (has_dataset(halos_group, "TreeID")) {
        tree_halos_.tree_id = read_1d_dataset<int32_t>(halos_group.getDataSet("TreeID"));
    }
    if (has_dataset(halos_group, "TreeIndex")) {
        tree_halos_.tree_index = read_1d_dataset<int64_t>(halos_group.getDataSet("TreeIndex"));
    }

    // Read positions (split 2D array into SOA)
    if (has_dataset(halos_group, "SubhaloPos")) {
        read_2d_dataset_split<float>(
            halos_group.getDataSet("SubhaloPos"),
            tree_halos_.pos_x, tree_halos_.pos_y, tree_halos_.pos_z
        );
    }

    // Read velocities
    if (has_dataset(halos_group, "SubhaloVel")) {
        read_2d_dataset_split<float>(
            halos_group.getDataSet("SubhaloVel"),
            tree_halos_.vel_x, tree_halos_.vel_y, tree_halos_.vel_z
        );
    }

    // Read masses
    if (has_dataset(halos_group, "SubhaloMass")) {
        tree_halos_.mass = read_1d_dataset<float>(halos_group.getDataSet("SubhaloMass"));
    }
    if (has_dataset(halos_group, "Group_M_Crit200")) {
        tree_halos_.m200c = read_1d_dataset<float>(halos_group.getDataSet("Group_M_Crit200"));
    }
    if (has_dataset(halos_group, "Group_R_Crit200")) {
        tree_halos_.r200c = read_1d_dataset<float>(halos_group.getDataSet("Group_R_Crit200"));
    }

    // Read tree links (relative indices within tree)
    if (has_dataset(halos_group, "TreeMainProgenitor")) {
        tree_halos_.main_progenitor = read_1d_dataset<int32_t>(
            halos_group.getDataSet("TreeMainProgenitor"));
    }
    if (has_dataset(halos_group, "TreeDescendant")) {
        tree_halos_.descendant = read_1d_dataset<int32_t>(
            halos_group.getDataSet("TreeDescendant"));
    }
    if (has_dataset(halos_group, "TreeFirstProgenitor")) {
        tree_halos_.first_progenitor = read_1d_dataset<int32_t>(
            halos_group.getDataSet("TreeFirstProgenitor"));
    }
    if (has_dataset(halos_group, "TreeNextProgenitor")) {
        tree_halos_.next_progenitor = read_1d_dataset<int32_t>(
            halos_group.getDataSet("TreeNextProgenitor"));
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

    info.mass = h.mass[idx];
    info.m200c = h.m200c[idx];
    info.r200c = h.r200c[idx];

    return info;
}

void MergerTree::clear_cache() {
    tree_table_.clear();
    tree_table_.shrink_to_fit();
    tree_halos_.clear();
    table_loaded_ = false;
    search_loaded_ = false;
    halos_loaded_ = false;
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

    // Sort by scale factor (early to late)
    std::sort(branch.begin(), branch.end(),
              [](const HaloInfo& a, const HaloInfo& b) {
                  return a.scale_factor < b.scale_factor;
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

} // namespace io
} // namespace tessera
