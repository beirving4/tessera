#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <map>

namespace tessera {
namespace io {

/**
 * @brief Position-Based Halo Tracking
 *
 * Alternative to merger tree linking that tracks halos by spatial position
 * and mass. Useful for:
 * - Validating merger tree results
 * - Handling tracking errors in merger trees
 * - Cases where merger trees are unavailable
 *
 * All positions are in comoving coordinates [Mpc/h or cMpc/h].
 * Masses are in GADGET-4 units [10^10 M_sun/h].
 */

// Forward declarations
class HaloCatalogReader;
class PositionMatcher;
class PositionBranchExtractor;
class SpatialGrid;

/**
 * Data for all FOF groups at a single snapshot.
 *
 * Stored in Structure-of-Arrays format for SIMD-friendly operations.
 */
struct HaloCatalogData {
    int32_t snap_num = -1;          ///< Snapshot number
    float scale_factor = 0.0f;      ///< Scale factor a = 1/(1+z)

    // Group properties (indexed by group_nr)
    std::vector<float> pos_x, pos_y, pos_z;  ///< Comoving positions [Mpc/h]
    std::vector<float> m200c;       ///< M200 critical [10^10 Msun/h]
    std::vector<float> r200c;       ///< R200 critical [Mpc/h]

    /**
     * Number of groups in this snapshot.
     */
    size_t n_groups() const { return m200c.size(); }

    /**
     * Check if empty.
     */
    bool empty() const { return m200c.empty(); }

    /**
     * Check if valid (has data).
     */
    bool is_valid() const { return snap_num >= 0 && !empty(); }

    /**
     * Reserve memory for expected number of groups.
     */
    void reserve(size_t n) {
        pos_x.reserve(n);
        pos_y.reserve(n);
        pos_z.reserve(n);
        m200c.reserve(n);
        r200c.reserve(n);
    }

    /**
     * Clear all data.
     */
    void clear() {
        snap_num = -1;
        scale_factor = 0.0f;
        pos_x.clear();
        pos_y.clear();
        pos_z.clear();
        m200c.clear();
        r200c.clear();
    }
};

/**
 * Spatial grid index for O(1) neighbor lookups.
 *
 * Divides the simulation box into cubic cells and indexes groups by cell.
 * Enables fast neighbor queries by only checking groups in nearby cells
 * instead of all groups in the catalog.
 *
 * Handles periodic boundary conditions automatically.
 */
class SpatialGrid {
public:
    /**
     * Construct spatial grid.
     * @param box_size Simulation box size [Mpc/h]
     * @param cell_size Size of each cell [Mpc/h], typically = search_radius
     */
    SpatialGrid(float box_size, float cell_size);

    /**
     * Build index from catalog data.
     * @param catalog Halo catalog to index
     */
    void build(const HaloCatalogData& catalog);

    /**
     * Get candidate group indices near a position.
     *
     * Returns all groups within 1 cell radius (27 cells checked).
     * Caller should still compute exact distances to filter.
     *
     * @param pos Query position [Mpc/h]
     * @return Vector of group indices to check
     */
    std::vector<int32_t> get_candidates(const std::array<float, 3>& pos) const;

    /**
     * Check if index has been built.
     */
    bool is_built() const { return built_; }

    /**
     * Get number of cells per dimension.
     */
    int32_t n_cells_per_dim() const { return n_cells_; }

    /**
     * Get total number of cells (dense extent = n_cells_per_dim^3), not just
     * the occupied ones actually stored.
     */
    size_t n_cells_total() const {
        return static_cast<size_t>(n_cells_) * n_cells_ * n_cells_;
    }

    /**
     * Get cell size.
     */
    float cell_size() const { return cell_size_; }

    /**
     * Clear the index.
     */
    void clear();

    /**
     * Get statistics about the index.
     * @return Tuple of (min groups per cell, max groups per cell, avg groups per cell)
     */
    std::tuple<size_t, size_t, float> get_stats() const;

private:
    float box_size_;
    float cell_size_;
    int32_t n_cells_;  // cells per dimension
    bool built_ = false;

    // Sparse 3D grid: cells_[flat_cell_index] = group indices in that cell.
    // Only OCCUPIED cells are stored. A dense (box/cell_size)^3 grid is
    // infeasible for large boxes (L2048: 410^3 ~= 69M cells) where occupancy
    // is tiny -- a hash map keeps build cost + memory O(n_groups), not O(box^3).
    std::unordered_map<size_t, std::vector<int32_t>> cells_;

    // Convert position to cell index
    int32_t pos_to_cell_1d(float x) const;
    size_t cell_3d_to_flat(int32_t ix, int32_t iy, int32_t iz) const;
};

/**
 * Indexed halo catalog with spatial grid for fast neighbor queries.
 *
 * Wraps HaloCatalogData with a pre-built spatial index.
 */
struct IndexedHaloCatalog {
    HaloCatalogData data;       ///< The underlying catalog data
    SpatialGrid grid;           ///< Spatial index

    /**
     * Construct from catalog data.
     * @param catalog Catalog data to wrap
     * @param cell_size Cell size for spatial grid [Mpc/h]
     */
    IndexedHaloCatalog(HaloCatalogData catalog, float box_size, float cell_size);

    /**
     * Check if valid.
     */
    bool is_valid() const { return data.is_valid() && grid.is_built(); }
};

/**
 * Information about a tracked halo at one snapshot.
 */
struct TrackedHaloInfo {
    int32_t snap_num = -1;          ///< Snapshot number
    int32_t group_nr = -1;          ///< FOF group index
    float scale_factor = 0.0f;      ///< Scale factor a = 1/(1+z)

    std::array<float, 3> position = {0, 0, 0};  ///< Comoving position [Mpc/h]
    float m200c = 0.0f;             ///< M200 critical [10^10 Msun/h]
    float r200c = 0.0f;             ///< R200 critical [Mpc/h]

    float match_score = 0.0f;       ///< Quality score (lower is better)
    float match_distance = 0.0f;    ///< Spatial distance from previous halo

    /**
     * Check if this represents a valid halo.
     */
    bool is_valid() const { return snap_num >= 0 && group_nr >= 0; }
};

/**
 * Result of a position match operation.
 */
struct PositionMatchResult {
    int32_t group_nr = -1;          ///< Matched group index (-1 if no match)
    float score = 1e30f;            ///< Match score (large value = no match)
    float distance = 1e30f;         ///< Spatial distance (large value = no match)

    /**
     * Check if a match was found.
     */
    bool found() const { return group_nr >= 0; }
};

/**
 * Configuration for position matching.
 */
struct PositionMatchConfig {
    float search_radius = 5.0f;     ///< Maximum search radius [Mpc/h]
    float mass_weight = 0.3f;       ///< Weight for mass similarity in scoring
    float min_mass_ratio = 0.05f;   ///< Minimum mass ratio to consider

    /**
     * Default configuration.
     */
    static PositionMatchConfig default_config() {
        return PositionMatchConfig{};
    }
};

/**
 * FOF/SUBFIND Halo Catalog Reader.
 *
 * Loads halo data from GADGET-4 FOF/SUBFIND output files.
 * Handles both single-file and sharded catalogs.
 */
class HaloCatalogReader {
public:
    /**
     * Construct catalog reader.
     * @param catalog_dir Directory containing halo catalogs
     */
    explicit HaloCatalogReader(const std::string& catalog_dir);

    /**
     * Destructor.
     */
    ~HaloCatalogReader();

    // Non-copyable
    HaloCatalogReader(const HaloCatalogReader&) = delete;
    HaloCatalogReader& operator=(const HaloCatalogReader&) = delete;

    // Movable
    HaloCatalogReader(HaloCatalogReader&&) noexcept;
    HaloCatalogReader& operator=(HaloCatalogReader&&) noexcept;

    /**
     * Get list of available snapshot numbers.
     */
    std::vector<int32_t> get_available_snapshots() const;

    /**
     * Load FOF group data for a snapshot.
     * @param snap_num Snapshot number
     * @return Catalog data, or empty data if not found
     */
    HaloCatalogData load_snapshot(int32_t snap_num) const;

    /**
     * Load multiple snapshots at once.
     * @param snap_nums Snapshot numbers to load
     * @return Map of snap_num -> catalog data
     */
    std::map<int32_t, HaloCatalogData> load_snapshots(
        const std::vector<int32_t>& snap_nums
    ) const;

    /**
     * Load all available snapshots.
     * @return Map of snap_num -> catalog data
     */
    std::map<int32_t, HaloCatalogData> load_all_snapshots() const;

    /**
     * Get catalog directory path.
     */
    const std::string& catalog_dir() const { return catalog_dir_; }

private:
    std::string catalog_dir_;
    mutable std::vector<int32_t> available_snapshots_;  // Cached
    mutable bool snapshots_scanned_ = false;

    std::vector<std::string> find_catalog_files(int32_t snap_num) const;
    void scan_available_snapshots() const;
};

/**
 * Position-based halo matcher.
 *
 * Core algorithm for matching halos between snapshots using position
 * and mass similarity.
 */
class PositionMatcher {
public:
    /**
     * Construct matcher with configuration.
     * @param box_size Simulation box size [Mpc/h]
     * @param config Matching configuration
     */
    PositionMatcher(float box_size, PositionMatchConfig config = {});

    /**
     * Find the best matching group at a snapshot.
     *
     * Uses combined scoring: score = distance + mass_weight * |log10(M/M_target)|
     *
     * NOTE: This is O(N) where N is number of groups. For large catalogs,
     * prefer the overload that takes IndexedHaloCatalog for O(k) performance.
     *
     * @param target_pos Target position [Mpc/h]
     * @param target_mass Target mass [10^10 Msun/h]
     * @param catalog Catalog data to search
     * @return Match result
     */
    PositionMatchResult find_best_match(
        const std::array<float, 3>& target_pos,
        float target_mass,
        const HaloCatalogData& catalog
    ) const;

    /**
     * Find the best matching group using spatial index (fast).
     *
     * Uses pre-built spatial grid for O(k) performance where k is the
     * number of groups in nearby cells, typically << N total groups.
     *
     * @param target_pos Target position [Mpc/h]
     * @param target_mass Target mass [10^10 Msun/h]
     * @param indexed_catalog Indexed catalog with spatial grid
     * @return Match result
     */
    PositionMatchResult find_best_match(
        const std::array<float, 3>& target_pos,
        float target_mass,
        const IndexedHaloCatalog& indexed_catalog
    ) const;

    /**
     * Find best matches for multiple targets (vectorized).
     *
     * @param target_positions Target positions (N x 3)
     * @param target_masses Target masses (N)
     * @param catalog Catalog data to search
     * @return Match results (N)
     */
    std::vector<PositionMatchResult> find_best_matches_batch(
        const std::vector<std::array<float, 3>>& target_positions,
        const std::vector<float>& target_masses,
        const HaloCatalogData& catalog
    ) const;

    /**
     * Compute periodic distance between two positions.
     */
    float periodic_distance(
        const std::array<float, 3>& pos1,
        const std::array<float, 3>& pos2
    ) const;

    /**
     * Get configuration.
     */
    const PositionMatchConfig& config() const { return config_; }

    /**
     * Get box size.
     */
    float box_size() const { return box_size_; }

private:
    float box_size_;
    PositionMatchConfig config_;

    // SIMD-optimized distance computation
    void compute_distances_simd(
        const std::array<float, 3>& target,
        const HaloCatalogData& catalog,
        std::vector<float>& distances
    ) const;
};

/**
 * Position-based branch extraction.
 *
 * Tracks halos across snapshots using position matching instead of
 * merger tree links. Supports two parallelization strategies:
 *
 * - Option A: Parallelize across branches (simple, some memory duplication)
 * - Option B: Preload catalogs, parallelize matching (memory efficient, default)
 */
class PositionBranchExtractor {
public:
    /**
     * Parallelization strategy.
     */
    enum class Strategy {
        PARALLEL_BRANCHES,   ///< Option A: Each thread tracks one branch
        PRELOAD_CATALOGS     ///< Option B: Load all catalogs, parallelize matching (default)
    };

    /**
     * Construct extractor.
     * @param reader Catalog reader (must outlive extractor)
     * @param box_size Simulation box size [Mpc/h]
     * @param config Matching configuration
     */
    PositionBranchExtractor(
        HaloCatalogReader& reader,
        float box_size,
        PositionMatchConfig config = {}
    );

    /**
     * Track a single branch.
     *
     * @param start_snap Starting snapshot number
     * @param start_group_nr FOF group number at starting snapshot
     * @param backward Track backward in time (progenitors)
     * @param forward Track forward in time (descendants)
     * @param snap_min Minimum snapshot to track to
     * @param snap_max Maximum snapshot to track to
     * @return Tracked halos sorted by scale factor (early to late)
     */
    std::vector<TrackedHaloInfo> track_branch(
        int32_t start_snap,
        int32_t start_group_nr,
        bool backward = true,
        bool forward = false,
        std::optional<int32_t> snap_min = std::nullopt,
        std::optional<int32_t> snap_max = std::nullopt
    );

    /**
     * Track a single branch using preloaded catalogs.
     *
     * More efficient when tracking multiple branches with the same catalogs.
     *
     * @param catalogs Preloaded catalog data (snap_num -> data)
     * @param start_snap Starting snapshot number
     * @param start_group_nr FOF group number at starting snapshot
     * @param backward Track backward in time
     * @param forward Track forward in time
     * @return Tracked halos sorted by scale factor
     */
    std::vector<TrackedHaloInfo> track_branch_preloaded(
        const std::map<int32_t, HaloCatalogData>& catalogs,
        int32_t start_snap,
        int32_t start_group_nr,
        bool backward = true,
        bool forward = false
    );

    /**
     * Unwrap periodic coordinates for smooth visualization.
     * @param branch Branch to unwrap (modified in-place)
     */
    void unwrap_coordinates(std::vector<TrackedHaloInfo>& branch);

    /**
     * Get the position matcher.
     */
    const PositionMatcher& matcher() const { return matcher_; }

    /**
     * Preload all catalogs into memory.
     * Call this before extract_all_branches_parallel with PRELOAD_CATALOGS strategy.
     */
    void preload_all_catalogs();

    /**
     * Clear preloaded catalogs to free memory.
     */
    void clear_preloaded_catalogs();

    /**
     * Check if catalogs are preloaded.
     */
    bool catalogs_preloaded() const { return !preloaded_catalogs_.empty(); }

    /**
     * Get preloaded catalogs (for read-only access).
     */
    const std::map<int32_t, HaloCatalogData>& preloaded_catalogs() const {
        return preloaded_catalogs_;
    }

private:
    HaloCatalogReader& reader_;
    PositionMatcher matcher_;
    std::map<int32_t, HaloCatalogData> preloaded_catalogs_;
};

/**
 * Position-matched branch catalog data.
 *
 * Similar to BranchCatalogData but includes match quality information
 * and uses FOF group numbers instead of subhalo numbers.
 */
struct PositionBranchCatalogData {
    // =========================================================================
    // Branch Index
    // =========================================================================
    std::vector<int32_t> branch_id;     ///< Branch identifier
    std::vector<int64_t> start_offset;  ///< Start index in flattened data
    std::vector<int32_t> length;        ///< Number of halos in each branch
    std::vector<int32_t> root_snap;     ///< Root halo snapshot number
    std::vector<int32_t> root_group;    ///< Root halo FOF group index
    std::vector<float> root_m200c;      ///< Root halo M200c
    std::vector<float> a_min;           ///< Earliest scale factor
    std::vector<float> a_max;           ///< Latest scale factor

    // =========================================================================
    // Branch Data (flattened)
    // =========================================================================
    std::vector<int32_t> snap_num;      ///< Snapshot numbers
    std::vector<int32_t> group_nr;      ///< FOF group indices
    std::vector<float> scale_factor;    ///< Scale factors
    std::vector<float> pos_x, pos_y, pos_z;  ///< Positions [Mpc/h]
    std::vector<float> m200c;           ///< M200 critical [10^10 Msun/h]
    std::vector<float> r200c;           ///< R200 critical [Mpc/h]

    // Match quality information
    std::vector<float> match_score;     ///< Match quality score
    std::vector<float> match_distance;  ///< Distance from previous halo

    /**
     * Number of branches.
     */
    size_t n_branches() const { return branch_id.size(); }

    /**
     * Total number of halos.
     */
    size_t n_halos() const { return snap_num.size(); }

    /**
     * Check if empty.
     */
    bool empty() const { return branch_id.empty(); }
};

/**
 * Starting point specification for branch extraction.
 */
struct BranchStartPoint {
    int32_t snap_num;       ///< Starting snapshot
    int32_t group_nr;       ///< FOF group number
    int32_t branch_id = -1; ///< Optional branch ID (-1 = auto-assign)
};

/**
 * Extract position-matched branches in parallel.
 *
 * @param reader Catalog reader
 * @param box_size Simulation box size [Mpc/h]
 * @param start_points Branch starting points (snap, group pairs)
 * @param config Matching configuration
 * @param strategy Parallelization strategy (default: PRELOAD_CATALOGS)
 * @param n_threads Number of threads (0 = use OMP_NUM_THREADS)
 * @param progress_callback Optional progress callback (done, total)
 * @param snap_min Optional lower snapshot bound (inclusive); snapshots below
 *                 this are never loaded/considered. nullopt = no lower bound.
 * @param snap_max Optional upper snapshot bound (inclusive); snapshots above
 *                 this are never loaded/considered. nullopt = no upper bound.
 *                 Capping at the descendant snapshot avoids loading
 *                 future-extended catalogs that no backward branch reaches.
 * @return Position-matched branch catalog
 */
PositionBranchCatalogData extract_position_branches_parallel(
    HaloCatalogReader& reader,
    float box_size,
    const std::vector<BranchStartPoint>& start_points,
    PositionMatchConfig config = {},
    PositionBranchExtractor::Strategy strategy = PositionBranchExtractor::Strategy::PRELOAD_CATALOGS,
    int n_threads = 0,
    std::function<void(size_t, size_t)> progress_callback = nullptr,
    std::optional<int32_t> snap_min = std::nullopt,
    std::optional<int32_t> snap_max = std::nullopt
);

/**
 * Validate a merger tree branch using position matching.
 *
 * Compares merger tree tracking with position-based tracking to identify
 * potential tracking errors.
 *
 * @param tree_snap_nums Snapshot numbers from merger tree branch
 * @param tree_group_nrs Group numbers from merger tree branch
 * @param tree_m200c M200c values from merger tree branch
 * @param reader Catalog reader
 * @param box_size Simulation box size [Mpc/h]
 * @param config Matching configuration
 * @return Map of snap_num -> {tree_group, pos_group, match, mass_ratio}
 */
std::map<int32_t, std::unordered_map<std::string, float>> validate_branch_with_position_matching(
    const std::vector<int32_t>& tree_snap_nums,
    const std::vector<int32_t>& tree_group_nrs,
    const std::vector<float>& tree_m200c,
    HaloCatalogReader& reader,
    float box_size,
    PositionMatchConfig config = {}
);

} // namespace io
} // namespace tessera
