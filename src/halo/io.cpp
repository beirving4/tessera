#include "halo/io.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace tessera {
namespace halo {

std::vector<std::vector<double>> read_table(
    const std::string& file,
    const std::vector<int>& col_idxs
) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + file);
    }
    
    std::vector<std::vector<double>> cols(col_idxs.size());
    std::string line;
    
    // Find max column index needed
    int max_col = *std::max_element(col_idxs.begin(), col_idxs.end());
    
    while (std::getline(in, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::vector<double> row_vals;
        double val;
        
        while (iss >> val) {
            row_vals.push_back(val);
            if (static_cast<int>(row_vals.size()) > max_col) break;
        }
        
        if (static_cast<int>(row_vals.size()) <= max_col) {
            continue; // Skip incomplete rows
        }
        
        for (size_t i = 0; i < col_idxs.size(); ++i) {
            cols[i].push_back(row_vals[col_idxs[i]]);
        }
    }
    
    return cols;
}

std::vector<std::vector<double>> read_binary_table(
    const std::string& file,
    const std::vector<int>& col_idxs
) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + file);
    }
    
    // Read number of rows
    int64_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    
    int64_t jump = n * sizeof(double);
    
    std::vector<std::vector<double>> cols(col_idxs.size());
    for (size_t i = 0; i < col_idxs.size(); ++i) {
        int col_idx = col_idxs[i];
        if (col_idx >= VAL_NUM) {
            throw std::runtime_error("Column index out of range");
        }
        
        cols[i].resize(n);
        in.seekg(sizeof(int64_t) + jump * col_idx, std::ios::beg);
        in.read(reinterpret_cast<char*>(cols[i].data()), n * sizeof(double));
    }
    
    return cols;
}

void rockstar_convert(const std::string& in_file, const std::string& out_file) {
    // Read all columns
    std::vector<int> col_idxs(VAL_NUM);
    std::iota(col_idxs.begin(), col_idxs.end(), 0);
    
    auto cols = read_table(in_file, col_idxs);
    
    // Write binary
    std::ofstream out(out_file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot create file: " + out_file);
    }
    
    int64_t n = cols[0].size();
    out.write(reinterpret_cast<char*>(&n), sizeof(n));
    
    for (const auto& col : cols) {
        out.write(reinterpret_cast<const char*>(col.data()), n * sizeof(double));
    }
}

void rockstar_convert_top_n(const std::string& in_file, const std::string& out_file, int n) {
    // Read all columns
    std::vector<int> col_idxs(VAL_NUM);
    std::iota(col_idxs.begin(), col_idxs.end(), 0);
    
    auto cols = read_table(in_file, col_idxs);
    
    size_t total = cols[0].size();
    if (n > static_cast<int>(total)) n = static_cast<int>(total);
    
    // Sort by M200b (column 36) and get top N indices
    std::vector<size_t> idxs(total);
    std::iota(idxs.begin(), idxs.end(), 0);
    
    const auto& m200b = cols[static_cast<int>(Val::M200b)];
    std::partial_sort(idxs.begin(), idxs.begin() + n, idxs.end(),
        [&m200b](size_t a, size_t b) { return m200b[a] > m200b[b]; });
    
    idxs.resize(n);
    
    // Extract top N from each column
    std::vector<std::vector<double>> out_cols(cols.size());
    for (size_t j = 0; j < cols.size(); ++j) {
        out_cols[j].resize(n);
        for (int i = 0; i < n; ++i) {
            out_cols[j][i] = cols[j][idxs[i]];
        }
    }
    
    // Write binary
    std::ofstream out(out_file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot create file: " + out_file);
    }
    
    int64_t n64 = n;
    out.write(reinterpret_cast<char*>(&n64), sizeof(n64));
    
    for (const auto& col : out_cols) {
        out.write(reinterpret_cast<const char*>(col.data()), n * sizeof(double));
    }
}

// Helper to get actual column index for a Val
static int get_column_index(Val v) {
    int vi = static_cast<int>(v);
    if (vi < VAL_NUM) return vi;
    
    // Derived radius values need mass columns
    switch (v) {
        case Val::RadVir: return static_cast<int>(Val::MVir);
        case Val::Rad200b: return static_cast<int>(Val::M200b);
        case Val::Rad200c: return static_cast<int>(Val::M200c);
        case Val::Rad500c: return static_cast<int>(Val::M500c);
        case Val::Rad2500c: return static_cast<int>(Val::M2500c);
        default: throw std::runtime_error("Unknown Val type");
    }
}

// Apply post-processing for derived values
static void apply_transforms(
    const io::CosmologyHeader& cosmo,
    const std::vector<Val>& val_flags,
    std::vector<std::vector<double>>& vals
) {
    for (size_t i = 0; i < val_flags.size(); ++i) {
        switch (val_flags[i]) {
            case Val::RadVir:
                mass_to_radius(RadiusType::RVirial, cosmo, vals[i], vals[i]);
                break;
            case Val::Rad200b:
                mass_to_radius(RadiusType::R200m, cosmo, vals[i], vals[i]);
                break;
            case Val::Rad200c:
                mass_to_radius(RadiusType::R200c, cosmo, vals[i], vals[i]);
                break;
            case Val::Rad500c:
                mass_to_radius(RadiusType::R500c, cosmo, vals[i], vals[i]);
                break;
            case Val::Rad2500c:
                mass_to_radius(RadiusType::R2500c, cosmo, vals[i], vals[i]);
                break;
            case Val::Rs:
            case Val::RVir:
            case Val::RsKylpin:
                // Convert kpc to Mpc
                for (auto& v : vals[i]) v /= 1000.0;
                break;
            default:
                break;
        }
    }
}

void read_rockstar_vals(
    const std::string& file,
    const io::CosmologyHeader& cosmo,
    const std::vector<Val>& val_flags,
    std::vector<int>& ids,
    std::vector<std::vector<double>>& vals
) {
    // Build column list: always include ID first
    std::vector<int> col_idxs = {static_cast<int>(Val::ID)};
    for (auto v : val_flags) {
        col_idxs.push_back(get_column_index(v));
    }
    
    auto cols = read_table(file, col_idxs);
    
    // Extract IDs
    ids.resize(cols[0].size());
    for (size_t i = 0; i < cols[0].size(); ++i) {
        ids[i] = static_cast<int>(cols[0][i]);
    }
    
    // Extract value columns
    vals.assign(cols.begin() + 1, cols.end());
    
    // Apply transformations
    apply_transforms(cosmo, val_flags, vals);
}

void read_binary_rockstar_vals(
    const std::string& file,
    const io::CosmologyHeader& cosmo,
    const std::vector<Val>& val_flags,
    std::vector<int>& ids,
    std::vector<std::vector<double>>& vals
) {
    // Build column list: always include ID first
    std::vector<int> col_idxs = {static_cast<int>(Val::ID)};
    for (auto v : val_flags) {
        col_idxs.push_back(get_column_index(v));
    }
    
    auto cols = read_binary_table(file, col_idxs);
    
    // Extract IDs
    ids.resize(cols[0].size());
    for (size_t i = 0; i < cols[0].size(); ++i) {
        ids[i] = static_cast<int>(cols[0][i]);
    }
    
    // Extract value columns
    vals.assign(cols.begin() + 1, cols.end());
    
    // Apply transformations
    apply_transforms(cosmo, val_flags, vals);
}

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
) {
    int r_col = rockstar_column(r_type);
    std::vector<int> col_idxs = {
        static_cast<int>(Val::ID),
        static_cast<int>(Val::X),
        static_cast<int>(Val::Y),
        static_cast<int>(Val::Z),
        r_col
    };
    
    auto cols = read_table(file, col_idxs);
    
    size_t n = cols[0].size();
    rids.resize(n);
    for (size_t i = 0; i < n; ++i) {
        rids[i] = static_cast<int>(cols[0][i]);
    }
    
    xs = std::move(cols[1]);
    ys = std::move(cols[2]);
    zs = std::move(cols[3]);
    
    if (rockstar_is_mass(r_type)) {
        // Column contains mass, convert to radius
        ms = std::move(cols[4]);
        rs.resize(n);
        mass_to_radius(r_type, cosmo, ms, rs);
    } else {
        // Column contains radius in kpc
        rs = std::move(cols[4]);
        for (auto& r : rs) r /= 1000.0; // kpc -> Mpc
        ms.resize(n);
        radius_to_mass(r_type, cosmo, rs, ms);
    }
    
    // Sort by radius (largest first)
    std::vector<size_t> idxs(n);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(),
        [&rs](size_t a, size_t b) { return rs[a] > rs[b]; });
    
    // Apply sort order
    HaloData sorted;
    sorted.rids.resize(n);
    sorted.xs.resize(n);
    sorted.ys.resize(n);
    sorted.zs.resize(n);
    sorted.ms.resize(n);
    sorted.rs.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = idxs[i];
        sorted.rids[i] = rids[j];
        sorted.xs[i] = xs[j];
        sorted.ys[i] = ys[j];
        sorted.zs[i] = zs[j];
        sorted.ms[i] = ms[j];
        sorted.rs[i] = rs[j];
    }
    
    rids = std::move(sorted.rids);
    xs = std::move(sorted.xs);
    ys = std::move(sorted.ys);
    zs = std::move(sorted.zs);
    ms = std::move(sorted.ms);
    rs = std::move(sorted.rs);
}

} // namespace halo
} // namespace tessera
