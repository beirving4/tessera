#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

namespace tessera {
namespace io {

/**
 * @brief GADGET-4 Merger Tree units and conventions.
 *
 * Merger trees use the same internal code units as GADGET-4:
 * - Mass: 10^10 M_sun/h  (UnitMass_in_g = 1.989e43)
 * - Length: Mpc/h (typically) for positions and radii
 * - Velocity: km/s
 *
 * Tree linking arrays (TreeMainProgenitor, TreeDescendant, etc.) store
 * relative indices within each tree. Use get_absolute_index() to convert
 * to absolute TreeHalos indices.
 */

// Forward declarations
class MergerTree;
class HaloTracker;

/**
 * Merger tree file header information.
 *
 * Loaded immediately when MergerTree is constructed (small data).
 */
struct MergerTreeHeader {
    int64_t n_trees = 0;        ///< Number of trees (z=0 halos)
    int64_t n_halos = 0;        ///< Total number of halos across all trees
    int32_t n_files = 1;        ///< Number of distributed files
    int32_t last_snap = 0;      ///< Last snapshot number

    // Time arrays (indexed by snapshot number)
    std::vector<float> snap_times;      ///< Scale factor at each snapshot
    std::vector<float> snap_redshifts;  ///< Redshift at each snapshot

    // Cosmological parameters
    double box_size = 0.0;      ///< Simulation box size [Mpc/h]
    double hubble = 0.0;        ///< Hubble parameter h = H0/100
    double omega_matter = 0.0;  ///< Matter density parameter Omega_m
    double omega_lambda = 0.0;  ///< Dark energy density parameter Omega_Lambda

    /**
     * Get number of snapshots.
     */
    int32_t n_snapshots() const { return static_cast<int32_t>(snap_times.size()); }

    /**
     * Get scale factor for a snapshot number.
     */
    float get_scale_factor(int snap_num) const {
        return snap_times[snap_num];
    }

    /**
     * Get redshift for a snapshot number.
     */
    float get_redshift(int snap_num) const {
        return snap_redshifts[snap_num];
    }
};

/**
 * Information about a single halo (AOS format for output).
 *
 * This is the natural representation for accessing individual halos
 * and is returned by HaloTracker methods.
 *
 * @note All values are in GADGET-4 internal code units:
 *       - Positions: Mpc/h
 *       - Velocities: km/s
 *       - Masses: 10^10 M_sun/h
 *       - Radii: Mpc/h
 */
struct HaloInfo {
    int64_t tree_index = -1;    ///< Index in TreeHalos arrays
    int32_t snap_num = -1;      ///< Snapshot number
    int32_t subhalo_nr = -1;    ///< Subhalo index in catalog
    int32_t group_nr = -1;      ///< FOF group index

    float scale_factor = 0.0f;  ///< Scale factor a = 1/(1+z)
    float redshift = 0.0f;      ///< Redshift z

    std::array<float, 3> position = {0, 0, 0};  ///< Comoving position [Mpc/h]
    std::array<float, 3> velocity = {0, 0, 0};  ///< Velocity [km/s]

    float mass = 0.0f;          ///< Subhalo mass [10^10 M_sun/h]
    float m200c = 0.0f;         ///< M200 critical [10^10 M_sun/h]
    float r200c = 0.0f;         ///< R200 critical [Mpc/h]

    /**
     * Check if this represents a valid halo.
     */
    bool is_valid() const { return tree_index >= 0; }
};

/**
 * Tree table entry for locating halos within a specific tree.
 */
struct TreeTableEntry {
    int64_t start_offset = 0;   ///< Start index in TreeHalos arrays
    int64_t length = 0;         ///< Number of halos in this tree
    int64_t tree_id = 0;        ///< Tree identifier
};

/**
 * Bulk halo data in Structure-of-Arrays format for SIMD operations.
 *
 * This layout enables efficient vectorized search operations (AVX2 can
 * compare 8 int32s simultaneously). Used internally by MergerTree for
 * find_halo_in_tree() and similar operations.
 *
 * Data is loaded in two stages for efficiency:
 * - Stage 1 (search): snap_num, subhalo_nr only (~8 bytes/halo)
 * - Stage 2 (full): All remaining fields (~80 bytes/halo)
 */
struct TreeHalosSOA {
    // =========================================================================
    // Stage 1: Search-critical data (loaded first for fast find_halo_in_tree)
    // =========================================================================
    std::vector<int32_t> snap_num;      ///< Snapshot numbers
    std::vector<int32_t> subhalo_nr;    ///< Subhalo indices

    // =========================================================================
    // Stage 2: Full data (loaded on demand for make_halo_info, tracing, etc.)
    // =========================================================================
    std::vector<int32_t> group_nr;      ///< FOF group indices
    std::vector<int32_t> tree_id;       ///< Tree identifier for each halo
    std::vector<int64_t> tree_index;    ///< Index within tree (for relative->absolute conversion)

    // Positions and velocities (float32 matching HDF5 storage)
    std::vector<float> pos_x, pos_y, pos_z;  ///< Comoving positions [Mpc/h]
    std::vector<float> vel_x, vel_y, vel_z;  ///< Velocities [km/s]

    // Masses and radii
    std::vector<float> mass;            ///< Subhalo mass [10^10 M_sun/h]
    std::vector<float> m200c;           ///< M200 critical [10^10 M_sun/h]
    std::vector<float> r200c;           ///< R200 critical [Mpc/h]

    // Tree links (relative indices within tree, -1 = no link)
    std::vector<int32_t> main_progenitor;   ///< TreeMainProgenitor
    std::vector<int32_t> descendant;        ///< TreeDescendant
    std::vector<int32_t> first_progenitor;  ///< TreeFirstProgenitor
    std::vector<int32_t> next_progenitor;   ///< TreeNextProgenitor

    /**
     * Number of halos.
     */
    size_t size() const { return snap_num.size(); }

    /**
     * Check if empty.
     */
    bool empty() const { return snap_num.empty(); }

    /**
     * Reserve memory for expected number of halos.
     */
    void reserve(size_t n);

    /**
     * Clear all data.
     */
    void clear();

    /**
     * Clear only stage 2 data (keeps search data).
     */
    void clear_full_data();
};

/**
 * GADGET-4 Merger Tree Reader.
 *
 * Efficient reader for GADGET-4 merger tree HDF5 files with lazy loading.
 * Header information is loaded immediately; TreeTable and TreeHalos are
 * loaded on first access to minimize memory usage.
 *
 * Example usage:
 * @code
 *   MergerTree tree("trees.hdf5");
 *   std::cout << "Found " << tree.n_trees() << " trees\n";
 *
 *   // Find a halo
 *   auto idx = tree.find_halo_in_tree(74, 0);  // snap 74, subhalo 0
 *   if (idx) {
 *       HaloInfo info = tree.make_halo_info(*idx);
 *       std::cout << "Mass: " << info.mass << "\n";
 *   }
 * @endcode
 *
 * @note Uses SIMD-optimized search when AVX2 is available.
 */
class MergerTree {
public:
    /**
     * Construct merger tree reader.
     * @param filename Path to trees.hdf5 file
     * @throws std::runtime_error if file cannot be opened
     */
    explicit MergerTree(const std::string& filename);

    /**
     * Destructor (cleans up cached data).
     */
    ~MergerTree();

    // Non-copyable (owns file resources)
    MergerTree(const MergerTree&) = delete;
    MergerTree& operator=(const MergerTree&) = delete;

    // Movable
    MergerTree(MergerTree&&) noexcept;
    MergerTree& operator=(MergerTree&&) noexcept;

    // =========================================================================
    // Header accessors (no lazy loading needed)
    // =========================================================================

    /**
     * Get header information.
     */
    const MergerTreeHeader& header() const { return header_; }

    /**
     * Number of trees (z=0 halos).
     */
    int64_t n_trees() const { return header_.n_trees; }

    /**
     * Total number of halos across all trees.
     */
    int64_t n_halos() const { return header_.n_halos; }

    /**
     * Number of snapshots.
     */
    int32_t n_snapshots() const { return header_.n_snapshots(); }

    /**
     * Simulation box size [Mpc/h].
     */
    double box_size() const { return header_.box_size; }

    /**
     * Get scale factor for a snapshot number.
     */
    float get_scale_factor(int snap_num) const {
        return header_.get_scale_factor(snap_num);
    }

    /**
     * Get redshift for a snapshot number.
     */
    float get_redshift(int snap_num) const {
        return header_.get_redshift(snap_num);
    }

    /**
     * Find snapshot number nearest to given scale factor.
     */
    int32_t find_nearest_snapshot(float scale_factor) const;

    // =========================================================================
    // Tree table access (lazy loaded)
    // =========================================================================

    /**
     * Get tree table entry for a specific tree.
     * @param tree_id Tree identifier (0 to n_trees-1)
     */
    const TreeTableEntry& get_tree_entry(int64_t tree_id);

    /**
     * Get indices into TreeHalos for a specific tree.
     * @param tree_id Tree identifier (0 to n_trees-1)
     * @return Pair of (start_index, end_index)
     */
    std::pair<int64_t, int64_t> get_tree_indices(int64_t tree_id);

    /**
     * Get TreeHalos index of the root (final snapshot) halo for a tree.
     * @param tree_id Tree identifier (0 to n_trees-1)
     */
    int64_t get_root_halo_index(int64_t tree_id);

    // =========================================================================
    // Halo search (lazy loaded, SIMD optimized)
    // =========================================================================

    /**
     * Find TreeHalos index for a given snapshot/subhalo combination.
     *
     * Uses SIMD-optimized search when AVX2 is available (8x parallel comparison).
     *
     * @param snap_num Snapshot number
     * @param subhalo_nr Subhalo index at that snapshot
     * @param tree_id If provided, only search within this tree
     * @return TreeHalos index, or nullopt if not found
     */
    std::optional<int64_t> find_halo_in_tree(
        int32_t snap_num,
        int32_t subhalo_nr,
        std::optional<int64_t> tree_id = std::nullopt
    );

    /**
     * Create HaloInfo from a TreeHalos index.
     * @param tree_index Absolute index in TreeHalos arrays
     */
    HaloInfo make_halo_info(int64_t tree_index);

    /**
     * Convert relative tree index to absolute TreeHalos index.
     * @param relative_idx Index relative to tree start (-1 = invalid)
     * @param tree_start Absolute start offset of the tree
     * @return Absolute index, or -1 if relative_idx was -1
     */
    static int64_t get_absolute_index(int32_t relative_idx, int64_t tree_start) {
        return relative_idx == -1 ? -1 : tree_start + relative_idx;
    }

    // =========================================================================
    // Raw data access (for advanced use)
    // =========================================================================

    /**
     * Get raw TreeHalos data (triggers lazy load).
     */
    const TreeHalosSOA& tree_halos();

    /**
     * Check if TreeHalos search data is loaded (snap_num, subhalo_nr).
     */
    bool search_data_loaded() const { return search_loaded_; }

    /**
     * Check if full TreeHalos data is loaded.
     */
    bool halos_loaded() const { return halos_loaded_; }

    /**
     * Check if TreeTable data is loaded.
     */
    bool table_loaded() const { return table_loaded_; }

    // =========================================================================
    // Memory management
    // =========================================================================

    /**
     * Clear cached TreeTable and TreeHalos data to free memory.
     */
    void clear_cache();

    /**
     * Clear only TreeHalos data (keeps TreeTable for chunked processing).
     */
    void clear_halos_cache();

    /**
     * Get approximate memory usage in bytes.
     */
    size_t memory_usage() const;

    // =========================================================================
    // Partial/Range loading (for memory-constrained chunked processing)
    // =========================================================================

    /**
     * Load a range of TreeHalos data for memory-efficient chunked processing.
     *
     * Only loads halos in [start_idx, end_idx) range. Clears any previously
     * loaded data first. Use clear_halos_cache() between chunks.
     *
     * @param start_idx Start index (inclusive) in TreeHalos
     * @param end_idx End index (exclusive) in TreeHalos
     */
    void load_tree_halos_range(int64_t start_idx, int64_t end_idx);

    /**
     * Check if partial/range data is loaded.
     */
    bool range_loaded() const { return range_loaded_; }

    /**
     * Get the currently loaded range [start, end).
     * Returns {0, 0} if no range is loaded.
     */
    std::pair<int64_t, int64_t> loaded_range() const {
        return {loaded_range_start_, loaded_range_end_};
    }

    /**
     * Get raw access to tree_halos_ without triggering lazy load.
     * Use this after calling load_tree_halos_range() to access range data.
     */
    const TreeHalosSOA& tree_halos_loaded() const {
        return tree_halos_;
    }

    /**
     * Convert absolute index to range-relative index.
     * Returns -1 if index is outside loaded range.
     */
    int64_t to_range_index(int64_t absolute_idx) const {
        if (!range_loaded_ || absolute_idx < loaded_range_start_ ||
            absolute_idx >= loaded_range_end_) {
            return -1;
        }
        return absolute_idx - loaded_range_start_;
    }

private:
    std::vector<std::string> files_;  ///< List of split files (or single file)
    MergerTreeHeader header_;

    // Lazy-loaded data
    std::vector<TreeTableEntry> tree_table_;
    TreeHalosSOA tree_halos_;
    bool table_loaded_ = false;
    bool search_loaded_ = false;  ///< Stage 1: snap_num, subhalo_nr loaded
    bool halos_loaded_ = false;   ///< Stage 2: All halo data loaded

    // Range loading state
    bool range_loaded_ = false;
    int64_t loaded_range_start_ = 0;
    int64_t loaded_range_end_ = 0;

    // Internal methods
    void load_header();
    void load_tree_table();
    void load_search_data();      ///< Stage 1: Load only search-critical data
    void load_full_halo_data();   ///< Stage 2: Load remaining halo data
    void load_tree_halos();       ///< Legacy: Load all at once (calls both stages)

    // SIMD search implementation (returns -1 if not found)
    int64_t find_halo_simd(int32_t snap_num, int32_t subhalo_nr,
                           int64_t start_idx, int64_t end_idx);
    int64_t find_halo_scalar(int32_t snap_num, int32_t subhalo_nr,
                             int64_t start_idx, int64_t end_idx);
};

/**
 * Halo Branch Tracer.
 *
 * Traces progenitor and descendant branches through merger trees.
 * Uses prefetching for efficient pointer-chasing through tree links.
 *
 * Example usage:
 * @code
 *   MergerTree tree("trees.hdf5");
 *   HaloTracker tracker(tree);
 *
 *   // Trace from z=0 most massive halo
 *   auto branch = tracker.trace_main_branch(74, 0, true, false, 0.1f);
 *
 *   for (const auto& halo : branch) {
 *       std::cout << "a=" << halo.scale_factor
 *                 << ": M200=" << halo.m200c << "\n";
 *   }
 * @endcode
 */
class HaloTracker {
public:
    /**
     * Construct halo tracker.
     * @param tree Reference to merger tree (must outlive HaloTracker)
     */
    explicit HaloTracker(MergerTree& tree);

    /**
     * Trace the main branch through a halo.
     *
     * Uses prefetching for efficient pointer chasing through tree links.
     *
     * @param snap_num Reference snapshot number
     * @param subhalo_nr Subhalo index at reference snapshot
     * @param backward Trace progenitors (earlier times)
     * @param forward Trace descendants (later times)
     * @param a_min Minimum scale factor (nullopt = no limit)
     * @param a_max Maximum scale factor (nullopt = no limit)
     * @return Vector of HaloInfo sorted by scale factor (early to late)
     */
    std::vector<HaloInfo> trace_main_branch(
        int32_t snap_num,
        int32_t subhalo_nr,
        bool backward = true,
        bool forward = true,
        std::optional<float> a_min = std::nullopt,
        std::optional<float> a_max = std::nullopt
    );

    /**
     * Trace main branch starting from a tree's root halo.
     * @param tree_id Tree identifier (0 = most massive z=0 halo)
     */
    std::vector<HaloInfo> trace_from_tree_id(
        int64_t tree_id,
        bool backward = true,
        bool forward = true,
        std::optional<float> a_min = std::nullopt,
        std::optional<float> a_max = std::nullopt
    );

    /**
     * Trace ALL progenitor branches (not just main branch).
     *
     * Returns map from snapshot number to list of progenitors at that snapshot.
     * Useful for visualizing mergers.
     *
     * @param snap_num Reference snapshot
     * @param subhalo_nr Subhalo index
     * @param a_min Minimum scale factor (nullopt = no limit)
     * @return Map of snap_num -> vector of HaloInfo
     */
    std::unordered_map<int32_t, std::vector<HaloInfo>> trace_all_progenitors(
        int32_t snap_num,
        int32_t subhalo_nr,
        std::optional<float> a_min = std::nullopt
    );

    /**
     * Unwrap periodic coordinates for smooth tracking.
     *
     * Ensures position changes are continuous (no jumps across box boundary).
     *
     * @param branch Branch to unwrap (modified in-place)
     * @param box_size Box size (default: from merger tree)
     */
    void unwrap_coordinates(
        std::vector<HaloInfo>& branch,
        std::optional<double> box_size = std::nullopt
    );

    /**
     * Static version of unwrap_coordinates for testing.
     * @param positions Shape (N, 3) positions at successive times
     * @param box_size Periodic box size
     */
    static void unwrap_coordinates_static(
        std::vector<std::array<float, 3>>& positions,
        float box_size
    );

private:
    MergerTree& tree_;

    // Get tree start offset for absolute index conversion
    int64_t get_tree_start(int64_t absolute_idx);
};

// =============================================================================
// Parallel Branch Extraction
// =============================================================================

/**
 * Branch catalog data in Structure-of-Arrays format.
 *
 * Contains pre-extracted main progenitor branches for all trees,
 * stored in flattened arrays for efficient HDF5 writing and random access.
 * Trees are sorted by root M200c (descending) for convenient mass-ordered access.
 */
struct BranchCatalogData {
    // =========================================================================
    // Branch Index (small, always loaded by BranchCatalog reader)
    // =========================================================================
    std::vector<int32_t> tree_id;       ///< Tree identifier for each branch
    std::vector<int64_t> start_offset;  ///< Start index in flattened data arrays
    std::vector<int32_t> length;        ///< Number of halos in each branch
    std::vector<int32_t> root_snap;     ///< Root halo snapshot number
    std::vector<int32_t> root_subhalo;  ///< Root halo subhalo index
    std::vector<float> root_m200c;      ///< Root halo M200c for filtering
    std::vector<float> root_r200c;      ///< Root halo R200c
    std::vector<float> a_min;           ///< Earliest scale factor in branch
    std::vector<float> a_max;           ///< Latest scale factor in branch

    // =========================================================================
    // Branch Data (flattened, compressed in HDF5)
    // =========================================================================
    std::vector<int32_t> snap_num;      ///< Snapshot numbers
    std::vector<int32_t> subhalo_nr;    ///< Subhalo indices
    std::vector<float> scale_factor;    ///< Scale factors
    std::vector<float> pos_x, pos_y, pos_z;  ///< Positions [Mpc/h]
    std::vector<float> vel_x, vel_y, vel_z;  ///< Velocities [km/s]
    std::vector<float> m200c;           ///< M200 critical [10^10 Msun/h]
    std::vector<float> r200c;           ///< R200 critical [Mpc/h]
    std::vector<float> mass;            ///< Subhalo mass [10^10 Msun/h]

    /**
     * Number of branches (trees).
     */
    size_t n_branches() const { return tree_id.size(); }

    /**
     * Total number of halos across all branches.
     */
    size_t n_halos() const { return snap_num.size(); }

    /**
     * Check if empty.
     */
    bool empty() const { return tree_id.empty(); }
};

/**
 * Extract main progenitor branches for all trees in parallel.
 *
 * Uses OpenMP to parallelize branch tracing across trees. Tree data is loaded
 * once (shared memory) and each thread traces a subset of trees independently.
 *
 * @param tree MergerTree instance (data will be loaded if not already)
 * @param mass_min Minimum root M200c filter (nullopt = no filter)
 * @param a_min Minimum scale factor to trace to (nullopt = trace all)
 * @param n_threads Number of threads (0 = use OMP_NUM_THREADS or all cores)
 * @param chunk_size Number of trees to process per chunk (0 = all at once).
 *                   Use non-zero for memory-constrained systems. Each chunk
 *                   loads only the necessary TreeHalos data, reducing peak
 *                   memory usage at the cost of repeated I/O.
 * @param progress_callback Optional callback for progress updates (trees_done, total_trees)
 * @return BranchCatalogData with all extracted branches
 *
 * @note Trees are sorted by root M200c (descending) in the output.
 * @note Thread-safe: uses thread-local storage for intermediate results.
 * @note For chunk_size > 0, memory usage is approximately:
 *       chunk_size * avg_branch_length * 100 bytes
 */
BranchCatalogData extract_all_branches_parallel(
    MergerTree& tree,
    std::optional<float> mass_min = std::nullopt,
    std::optional<float> a_min = std::nullopt,
    int n_threads = 0,
    size_t chunk_size = 0,
    std::function<void(size_t, size_t)> progress_callback = nullptr
);

// =============================================================================
// Free functions for convenient single-file access
// =============================================================================

/**
 * Read merger tree header only (small, fast).
 * @param filename Path to trees.hdf5 file
 * @return Header information
 */
MergerTreeHeader read_merger_tree_header(const std::string& filename);

} // namespace io
} // namespace tessera
