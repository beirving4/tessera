#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "io/headers.h"
#include "halo/radius.h"

namespace asymptotic_tetra {
namespace halo {

/**
 * Rockstar catalog column indices.
 * 
 * These correspond to the columns in Rockstar halo finder output,
 * matching the order in out_*.list files.
 */
enum class Val {
    Scale = 0,
    ID = 1,
    DescScale = 2,
    DescID = 3,
    NumProg = 4,
    PID = 5,
    UPID = 6,
    DescPID = 7,
    Phantom = 8,
    SAMMVir = 9,
    MVir = 10,
    RVir = 11,
    Rs = 12,
    Vrms = 13,
    MMP = 14,
    ScaleOfLastMMP = 15,
    VMax = 16,
    X = 17,
    Y = 18,
    Z = 19,
    Vx = 20,
    Vy = 21,
    Vz = 22,
    Jx = 23,
    Jy = 24,
    Jz = 25,
    Spin = 26,
    BreadthFirstID = 27,
    DepthFirstID = 28,
    TreeRootID = 29,
    OrigHaloID = 30,
    SnapNum = 31,
    NextCoprogenitorDepthFirstID = 32,
    LastProgenitorDepthFirstID = 33,
    RsKylpin = 34,
    MVirAll = 35,
    M200b = 36,
    M200c = 37,
    M500c = 38,
    M2500c = 39,
    XOff = 40,
    Voff = 41,
    SpinBullock = 42,
    BToA = 43,
    CToA = 44,
    Ax = 45,
    Ay = 46,
    Az = 47,
    BToA500c = 48,
    CToA500c = 49,
    Ax500c = 50,
    Ay500c = 51,
    Az500c = 52,
    TU = 53,
    MAcc = 54,
    MPeak = 55,
    VAcc = 56,
    VPeak = 57,
    HalfmassScale = 58,
    AccRateInst = 59,
    AccRate100Myr = 60,
    AccRateTdyn = 61,
    // Derived values (computed, not stored directly)
    RadVir = 100,
    Rad200b = 101,
    Rad200c = 102,
    Rad500c = 103,
    Rad2500c = 104
};

constexpr int VAL_NUM = 62;  // Number of actual stored columns

/**
 * Helper struct for sorting halos by multiple properties simultaneously.
 */
struct HaloData {
    std::vector<int> rids;
    std::vector<double> xs, ys, zs, ms, rs;
    
    size_t size() const { return rs.size(); }
    
    void swap(size_t i, size_t j) {
        std::swap(rids[i], rids[j]);
        std::swap(xs[i], xs[j]);
        std::swap(ys[i], ys[j]);
        std::swap(zs[i], zs[j]);
        std::swap(ms[i], ms[j]);
        std::swap(rs[i], rs[j]);
    }
};

/**
 * Read a Rockstar halo catalog and return positions, masses, and radii.
 * Results are sorted from largest to smallest radius.
 * 
 * @param file Path to Rockstar ASCII catalog (out_*.list)
 * @param r_type Radius type to use
 * @param cosmo Cosmology header for mass-radius conversion
 * @param rids Output: Rockstar halo IDs
 * @param xs Output: X positions [Mpc/h]
 * @param ys Output: Y positions [Mpc/h]
 * @param zs Output: Z positions [Mpc/h]
 * @param ms Output: Masses [M_sun/h]
 * @param rs Output: Radii [Mpc/h]
 */
void read_rockstar(
    const std::string& file,
    RadiusType r_type,
    const io::CosmologyHeader& cosmo,
    std::vector<int>& rids,
    std::vector<double>& xs,
    std::vector<double>& ys,
    std::vector<double>& zs,
    std::vector<double>& ms,
    std::vector<double>& rs
);

/**
 * Read specific columns from a Rockstar ASCII catalog.
 * 
 * @param file Path to Rockstar ASCII catalog
 * @param cosmo Cosmology header (for radius conversions)
 * @param val_flags List of Val columns to read
 * @param ids Output: Rockstar halo IDs
 * @param vals Output: Requested column values (one vector per Val)
 */
void read_rockstar_vals(
    const std::string& file,
    const io::CosmologyHeader& cosmo,
    const std::vector<Val>& val_flags,
    std::vector<int>& ids,
    std::vector<std::vector<double>>& vals
);

/**
 * Read specific columns from a binary Rockstar catalog.
 * 
 * @param file Path to binary Rockstar catalog (created by rockstar_convert)
 * @param cosmo Cosmology header (for radius conversions)
 * @param val_flags List of Val columns to read
 * @param ids Output: Rockstar halo IDs
 * @param vals Output: Requested column values
 */
void read_binary_rockstar_vals(
    const std::string& file,
    const io::CosmologyHeader& cosmo,
    const std::vector<Val>& val_flags,
    std::vector<int>& ids,
    std::vector<std::vector<double>>& vals
);

/**
 * Convert a Rockstar ASCII catalog to binary format for faster loading.
 * 
 * @param in_file Input ASCII catalog path
 * @param out_file Output binary catalog path
 */
void rockstar_convert(const std::string& in_file, const std::string& out_file);

/**
 * Convert a Rockstar ASCII catalog to binary, keeping only the top N halos by M200b.
 * 
 * @param in_file Input ASCII catalog path
 * @param out_file Output binary catalog path
 * @param n Number of top halos to keep
 */
void rockstar_convert_top_n(const std::string& in_file, const std::string& out_file, int n);

/**
 * Read columns from an ASCII table file.
 * Lines starting with '#' are skipped.
 * 
 * @param file Path to ASCII file
 * @param col_idxs Column indices to read (0-based)
 * @return Vector of column vectors
 */
std::vector<std::vector<double>> read_table(
    const std::string& file,
    const std::vector<int>& col_idxs
);

/**
 * Read columns from a binary catalog file.
 * 
 * @param file Path to binary file
 * @param col_idxs Column indices to read (0-based)
 * @return Vector of column vectors
 */
std::vector<std::vector<double>> read_binary_table(
    const std::string& file,
    const std::vector<int>& col_idxs
);

} // namespace halo
} // namespace asymptotic_tetra
