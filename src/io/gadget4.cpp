#include "io/gadget4.h"
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <numeric>

namespace asymptotic_tetra {
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
 * Read string attribute (handles byte strings from Python h5py).
 */
std::string read_string_attr(const HighFive::Group& group, const std::string& name) {
    if (!group.hasAttribute(name)) {
        return "";
    }
    auto attr = group.getAttribute(name);
    
    // Try reading as string first
    try {
        std::string val;
        attr.read(val);
        return val;
    } catch (...) {
        // If that fails, try as fixed-length string
        return "";
    }
}

/**
 * Read array attribute of fixed size.
 */
template<typename T, size_t N>
std::array<T, N> read_array_attr(const HighFive::Group& group, const std::string& name) {
    std::array<T, N> result = {};
    if (group.hasAttribute(name)) {
        std::vector<T> vec;
        group.getAttribute(name).read(vec);
        for (size_t i = 0; i < std::min(N, vec.size()); i++) {
            result[i] = vec[i];
        }
    }
    return result;
}

/**
 * Check if a dataset exists in a group.
 */
bool has_dataset(const HighFive::Group& group, const std::string& name) {
    return group.exist(name) && group.getObjectType(name) == HighFive::ObjectType::Dataset;
}

/**
 * Read a 2D dataset into vector of arrays.
 */
template<typename T, size_t N>
std::vector<std::array<T, N>> read_2d_dataset(const HighFive::DataSet& ds) {
    auto dims = ds.getDimensions();
    if (dims.size() != 2 || dims[1] != N) {
        throw std::runtime_error("Dataset has unexpected dimensions");
    }
    
    std::vector<std::vector<T>> raw;
    ds.read(raw);
    
    std::vector<std::array<T, N>> result(dims[0]);
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < N; j++) {
            result[i][j] = raw[i][j];
        }
    }
    return result;
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

} // anonymous namespace

// =============================================================================
// Gadget4Header
// =============================================================================

uint64_t Gadget4Header::total_particles() const {
    uint64_t total = 0;
    for (auto n : num_part_total) {
        total += n;
    }
    return total;
}

uint64_t Gadget4Header::particles_of_type(int type) const {
    if (type < 0 || type >= 6) {
        throw std::out_of_range("Particle type must be 0-5");
    }
    return num_part_total[type];
}

// =============================================================================
// Gadget4HaloCatalog
// =============================================================================

std::vector<const Gadget4Subhalo*> Gadget4HaloCatalog::subhalos_of_group(size_t group_idx) const {
    std::vector<const Gadget4Subhalo*> result;
    if (group_idx >= groups.size()) {
        return result;
    }
    
    const auto& group = groups[group_idx];
    if (group.first_sub < 0 || group.n_subs <= 0) {
        return result;
    }
    
    for (int32_t i = 0; i < group.n_subs; i++) {
        size_t sub_idx = static_cast<size_t>(group.first_sub) + i;
        if (sub_idx < subhalos.size()) {
            result.push_back(&subhalos[sub_idx]);
        }
    }
    return result;
}

// =============================================================================
// Snapshot reading
// =============================================================================

Gadget4Header read_gadget4_header(const std::string& filename) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    if (!file.exist("Header")) {
        throw std::runtime_error("File does not contain Header group: " + filename);
    }
    
    auto header_group = file.getGroup("Header");
    
    Gadget4Header hd;
    hd.box_size = read_attr<double>(header_group, "BoxSize", 0.0);
    hd.time = read_attr<double>(header_group, "Time", 0.0);
    hd.redshift = read_attr<double>(header_group, "Redshift", 0.0);
    hd.num_files_per_snapshot = read_attr<int>(header_group, "NumFilesPerSnapshot", 1);
    
    // Read mass table
    hd.mass_table = read_array_attr<double, 6>(header_group, "MassTable");
    
    // Read particle counts - GADGET-4 uses NumPart_ThisFile and NumPart_Total
    auto npart_this = read_array_attr<uint64_t, 6>(header_group, "NumPart_ThisFile");
    auto npart_total = read_array_attr<uint64_t, 6>(header_group, "NumPart_Total");
    
    // Some versions use int instead of uint64
    if (npart_this[0] == 0 && npart_total[0] == 0) {
        auto npart_this_int = read_array_attr<int64_t, 6>(header_group, "NumPart_ThisFile");
        auto npart_total_int = read_array_attr<int64_t, 6>(header_group, "NumPart_Total");
        for (int i = 0; i < 6; i++) {
            hd.num_part_this_file[i] = static_cast<uint64_t>(npart_this_int[i]);
            hd.num_part_total[i] = static_cast<uint64_t>(npart_total_int[i]);
        }
    } else {
        hd.num_part_this_file = npart_this;
        hd.num_part_total = npart_total;
    }
    
    // Git info
    hd.git_commit = read_string_attr(header_group, "Git_commit");
    hd.git_date = read_string_attr(header_group, "Git_date");
    
    return hd;
}

Gadget4Snapshot read_gadget4_snapshot_file(
    const std::string& filename,
    uint32_t particle_types,
    bool read_velocities,
    bool read_ids
) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    Gadget4Snapshot snapshot;
    snapshot.header = read_gadget4_header(filename);
    
    // Read each requested particle type
    for (int ptype = 0; ptype < 6; ptype++) {
        if (!(particle_types & (1 << ptype))) {
            continue;
        }
        
        std::string group_name = "PartType" + std::to_string(ptype);
        if (!file.exist(group_name)) {
            continue;
        }
        
        auto pgroup = file.getGroup(group_name);
        auto& particles = snapshot.particles[ptype];
        
        // Read coordinates (required)
        if (has_dataset(pgroup, "Coordinates")) {
            auto ds = pgroup.getDataSet("Coordinates");
            particles.coordinates = read_2d_dataset<float, 3>(ds);
        }
        
        // Read velocities (optional)
        if (read_velocities && has_dataset(pgroup, "Velocities")) {
            auto ds = pgroup.getDataSet("Velocities");
            particles.velocities = read_2d_dataset<float, 3>(ds);
        }
        
        // Read particle IDs (optional)
        if (read_ids && has_dataset(pgroup, "ParticleIDs")) {
            auto ds = pgroup.getDataSet("ParticleIDs");
            // Try uint64 first, fall back to uint32
            try {
                particles.particle_ids = read_1d_dataset<uint64_t>(ds);
            } catch (...) {
                auto ids32 = read_1d_dataset<uint32_t>(ds);
                particles.particle_ids.resize(ids32.size());
                for (size_t i = 0; i < ids32.size(); i++) {
                    particles.particle_ids[i] = ids32[i];
                }
            }
        }
        
        // Read optional datasets
        if (has_dataset(pgroup, "Acceleration")) {
            auto ds = pgroup.getDataSet("Acceleration");
            particles.accelerations = read_2d_dataset<float, 3>(ds);
        }
        
        if (has_dataset(pgroup, "Potential")) {
            auto ds = pgroup.getDataSet("Potential");
            particles.potential = read_1d_dataset<float>(ds);
        }
        
        if (has_dataset(pgroup, "SubfindDensity")) {
            auto ds = pgroup.getDataSet("SubfindDensity");
            particles.subfind_density = read_1d_dataset<float>(ds);
        }
        
        if (has_dataset(pgroup, "SubfindHsml")) {
            auto ds = pgroup.getDataSet("SubfindHsml");
            particles.subfind_hsml = read_1d_dataset<float>(ds);
        }
        
        if (has_dataset(pgroup, "SubfindVelDisp")) {
            auto ds = pgroup.getDataSet("SubfindVelDisp");
            particles.subfind_vel_disp = read_1d_dataset<float>(ds);
        }
    }
    
    return snapshot;
}

std::vector<std::array<float, 3>> read_gadget4_positions(
    const std::string& filename,
    int particle_type
) {
    if (particle_type < 0 || particle_type >= 6) {
        throw std::out_of_range("Particle type must be 0-5");
    }
    
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    std::string group_name = "PartType" + std::to_string(particle_type);
    if (!file.exist(group_name)) {
        return {};
    }
    
    auto pgroup = file.getGroup(group_name);
    if (!has_dataset(pgroup, "Coordinates")) {
        return {};
    }
    
    auto ds = pgroup.getDataSet("Coordinates");
    return read_2d_dataset<float, 3>(ds);
}

// =============================================================================
// Halo catalog reading
// =============================================================================

Gadget4HaloCatalogHeader read_gadget4_halo_header(const std::string& filename) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    if (!file.exist("Header")) {
        throw std::runtime_error("File does not contain Header group: " + filename);
    }
    
    auto header_group = file.getGroup("Header");
    
    Gadget4HaloCatalogHeader hd;
    hd.box_size = read_attr<double>(header_group, "BoxSize", 0.0);
    hd.time = read_attr<double>(header_group, "Time", 0.0);
    hd.redshift = read_attr<double>(header_group, "Redshift", 0.0);
    hd.num_files = read_attr<int>(header_group, "NumFiles", 1);
    
    hd.n_groups_this_file = read_attr<int64_t>(header_group, "Ngroups_ThisFile", 0);
    hd.n_groups_total = read_attr<int64_t>(header_group, "Ngroups_Total", 0);
    hd.n_subhalos_this_file = read_attr<int64_t>(header_group, "Nsubhalos_ThisFile", 0);
    hd.n_subhalos_total = read_attr<int64_t>(header_group, "Nsubhalos_Total", 0);
    hd.n_ids_this_file = read_attr<int64_t>(header_group, "Nids_ThisFile", 0);
    hd.n_ids_total = read_attr<int64_t>(header_group, "Nids_Total", 0);
    
    hd.git_commit = read_string_attr(header_group, "Git_commit");
    hd.git_date = read_string_attr(header_group, "Git_date");
    
    return hd;
}

std::vector<Gadget4Group> read_gadget4_groups(const std::string& filename) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    if (!file.exist("Group")) {
        return {};
    }
    
    auto group = file.getGroup("Group");
    
    // Read all group datasets
    std::vector<std::array<float, 3>> pos, vel;
    std::vector<float> mass, ascale;
    std::vector<int32_t> len, n_subs;
    std::vector<int64_t> first_sub;
    std::vector<std::vector<float>> mass_type;
    std::vector<std::vector<int32_t>> len_type;
    
    // Spherical overdensity
    std::vector<float> m_crit200, r_crit200, m_crit500, r_crit500;
    std::vector<float> m_mean200, r_mean200, m_tophat200, r_tophat200;
    
    if (has_dataset(group, "GroupPos")) {
        pos = read_2d_dataset<float, 3>(group.getDataSet("GroupPos"));
    }
    if (has_dataset(group, "GroupVel")) {
        vel = read_2d_dataset<float, 3>(group.getDataSet("GroupVel"));
    }
    if (has_dataset(group, "GroupMass")) {
        mass = read_1d_dataset<float>(group.getDataSet("GroupMass"));
    }
    if (has_dataset(group, "GroupLen")) {
        len = read_1d_dataset<int32_t>(group.getDataSet("GroupLen"));
    }
    if (has_dataset(group, "GroupNsubs")) {
        n_subs = read_1d_dataset<int32_t>(group.getDataSet("GroupNsubs"));
    }
    if (has_dataset(group, "GroupFirstSub")) {
        first_sub = read_1d_dataset<int64_t>(group.getDataSet("GroupFirstSub"));
    }
    if (has_dataset(group, "GroupAscale")) {
        ascale = read_1d_dataset<float>(group.getDataSet("GroupAscale"));
    }
    
    // Mass/len by type (2D arrays in the file)
    if (has_dataset(group, "GroupMassType")) {
        group.getDataSet("GroupMassType").read(mass_type);
    }
    if (has_dataset(group, "GroupLenType")) {
        group.getDataSet("GroupLenType").read(len_type);
    }
    
    // Spherical overdensity properties
    if (has_dataset(group, "Group_M_Crit200")) {
        m_crit200 = read_1d_dataset<float>(group.getDataSet("Group_M_Crit200"));
    }
    if (has_dataset(group, "Group_R_Crit200")) {
        r_crit200 = read_1d_dataset<float>(group.getDataSet("Group_R_Crit200"));
    }
    if (has_dataset(group, "Group_M_Crit500")) {
        m_crit500 = read_1d_dataset<float>(group.getDataSet("Group_M_Crit500"));
    }
    if (has_dataset(group, "Group_R_Crit500")) {
        r_crit500 = read_1d_dataset<float>(group.getDataSet("Group_R_Crit500"));
    }
    if (has_dataset(group, "Group_M_Mean200")) {
        m_mean200 = read_1d_dataset<float>(group.getDataSet("Group_M_Mean200"));
    }
    if (has_dataset(group, "Group_R_Mean200")) {
        r_mean200 = read_1d_dataset<float>(group.getDataSet("Group_R_Mean200"));
    }
    if (has_dataset(group, "Group_M_TopHat200")) {
        m_tophat200 = read_1d_dataset<float>(group.getDataSet("Group_M_TopHat200"));
    }
    if (has_dataset(group, "Group_R_TopHat200")) {
        r_tophat200 = read_1d_dataset<float>(group.getDataSet("Group_R_TopHat200"));
    }
    
    // Build group vector
    size_t n_groups = pos.size();
    std::vector<Gadget4Group> groups(n_groups);
    
    for (size_t i = 0; i < n_groups; i++) {
        auto& g = groups[i];
        
        if (i < pos.size()) g.pos = pos[i];
        if (i < vel.size()) g.vel = vel[i];
        if (i < mass.size()) g.mass = mass[i];
        if (i < len.size()) g.len = len[i];
        if (i < first_sub.size()) g.first_sub = first_sub[i];
        if (i < n_subs.size()) g.n_subs = n_subs[i];
        if (i < ascale.size()) g.ascale = ascale[i];
        
        // Mass/len by type
        if (i < mass_type.size()) {
            for (size_t j = 0; j < std::min(mass_type[i].size(), size_t(6)); j++) {
                g.mass_type[j] = mass_type[i][j];
            }
        }
        if (i < len_type.size()) {
            for (size_t j = 0; j < std::min(len_type[i].size(), size_t(6)); j++) {
                g.len_type[j] = len_type[i][j];
            }
        }
        
        // Spherical overdensity
        if (i < m_crit200.size()) g.m_crit200 = m_crit200[i];
        if (i < r_crit200.size()) g.r_crit200 = r_crit200[i];
        if (i < m_crit500.size()) g.m_crit500 = m_crit500[i];
        if (i < r_crit500.size()) g.r_crit500 = r_crit500[i];
        if (i < m_mean200.size()) g.m_mean200 = m_mean200[i];
        if (i < r_mean200.size()) g.r_mean200 = r_mean200[i];
        if (i < m_tophat200.size()) g.m_tophat200 = m_tophat200[i];
        if (i < r_tophat200.size()) g.r_tophat200 = r_tophat200[i];
    }
    
    return groups;
}

std::vector<Gadget4Subhalo> read_gadget4_subhalos(const std::string& filename) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    
    if (!file.exist("Subhalo")) {
        return {};
    }
    
    auto subhalo = file.getGroup("Subhalo");
    
    // Read all subhalo datasets
    std::vector<std::array<float, 3>> pos, vel, cm, spin;
    std::vector<float> mass, halfmass_rad, vmax, vmax_rad, vel_disp;
    std::vector<int32_t> len, rank_in_gr, parent_rank;
    std::vector<int64_t> group_nr;
    std::vector<uint64_t> id_mostbound;
    std::vector<std::vector<float>> mass_type, halfmass_rad_type;
    std::vector<std::vector<int32_t>> len_type;
    
    if (has_dataset(subhalo, "SubhaloPos")) {
        pos = read_2d_dataset<float, 3>(subhalo.getDataSet("SubhaloPos"));
    }
    if (has_dataset(subhalo, "SubhaloVel")) {
        vel = read_2d_dataset<float, 3>(subhalo.getDataSet("SubhaloVel"));
    }
    if (has_dataset(subhalo, "SubhaloCM")) {
        cm = read_2d_dataset<float, 3>(subhalo.getDataSet("SubhaloCM"));
    }
    if (has_dataset(subhalo, "SubhaloSpin")) {
        spin = read_2d_dataset<float, 3>(subhalo.getDataSet("SubhaloSpin"));
    }
    if (has_dataset(subhalo, "SubhaloMass")) {
        mass = read_1d_dataset<float>(subhalo.getDataSet("SubhaloMass"));
    }
    if (has_dataset(subhalo, "SubhaloLen")) {
        len = read_1d_dataset<int32_t>(subhalo.getDataSet("SubhaloLen"));
    }
    if (has_dataset(subhalo, "SubhaloHalfmassRad")) {
        halfmass_rad = read_1d_dataset<float>(subhalo.getDataSet("SubhaloHalfmassRad"));
    }
    if (has_dataset(subhalo, "SubhaloVmax")) {
        vmax = read_1d_dataset<float>(subhalo.getDataSet("SubhaloVmax"));
    }
    if (has_dataset(subhalo, "SubhaloVmaxRad")) {
        vmax_rad = read_1d_dataset<float>(subhalo.getDataSet("SubhaloVmaxRad"));
    }
    if (has_dataset(subhalo, "SubhaloVelDisp")) {
        vel_disp = read_1d_dataset<float>(subhalo.getDataSet("SubhaloVelDisp"));
    }
    if (has_dataset(subhalo, "SubhaloGroupNr")) {
        group_nr = read_1d_dataset<int64_t>(subhalo.getDataSet("SubhaloGroupNr"));
    }
    if (has_dataset(subhalo, "SubhaloRankInGr")) {
        rank_in_gr = read_1d_dataset<int32_t>(subhalo.getDataSet("SubhaloRankInGr"));
    }
    if (has_dataset(subhalo, "SubhaloParentRank")) {
        parent_rank = read_1d_dataset<int32_t>(subhalo.getDataSet("SubhaloParentRank"));
    }
    if (has_dataset(subhalo, "SubhaloIDMostbound")) {
        try {
            id_mostbound = read_1d_dataset<uint64_t>(subhalo.getDataSet("SubhaloIDMostbound"));
        } catch (...) {
            auto ids32 = read_1d_dataset<uint32_t>(subhalo.getDataSet("SubhaloIDMostbound"));
            id_mostbound.resize(ids32.size());
            for (size_t i = 0; i < ids32.size(); i++) {
                id_mostbound[i] = ids32[i];
            }
        }
    }
    
    // Mass/len by type
    if (has_dataset(subhalo, "SubhaloMassType")) {
        subhalo.getDataSet("SubhaloMassType").read(mass_type);
    }
    if (has_dataset(subhalo, "SubhaloLenType")) {
        subhalo.getDataSet("SubhaloLenType").read(len_type);
    }
    if (has_dataset(subhalo, "SubhaloHalfmassRadType")) {
        subhalo.getDataSet("SubhaloHalfmassRadType").read(halfmass_rad_type);
    }
    
    // Build subhalo vector
    size_t n_subhalos = pos.size();
    std::vector<Gadget4Subhalo> subhalos(n_subhalos);
    
    for (size_t i = 0; i < n_subhalos; i++) {
        auto& s = subhalos[i];
        
        if (i < pos.size()) s.pos = pos[i];
        if (i < vel.size()) s.vel = vel[i];
        if (i < cm.size()) s.cm = cm[i];
        if (i < spin.size()) s.spin = spin[i];
        if (i < mass.size()) s.mass = mass[i];
        if (i < len.size()) s.len = len[i];
        if (i < halfmass_rad.size()) s.halfmass_rad = halfmass_rad[i];
        if (i < vmax.size()) s.vmax = vmax[i];
        if (i < vmax_rad.size()) s.vmax_rad = vmax_rad[i];
        if (i < vel_disp.size()) s.vel_disp = vel_disp[i];
        if (i < group_nr.size()) s.group_nr = group_nr[i];
        if (i < rank_in_gr.size()) s.rank_in_gr = rank_in_gr[i];
        if (i < parent_rank.size()) s.parent_rank = parent_rank[i];
        if (i < id_mostbound.size()) s.id_mostbound = id_mostbound[i];
        
        // By type
        if (i < mass_type.size()) {
            for (size_t j = 0; j < std::min(mass_type[i].size(), size_t(6)); j++) {
                s.mass_type[j] = mass_type[i][j];
            }
        }
        if (i < len_type.size()) {
            for (size_t j = 0; j < std::min(len_type[i].size(), size_t(6)); j++) {
                s.len_type[j] = len_type[i][j];
            }
        }
        if (i < halfmass_rad_type.size()) {
            for (size_t j = 0; j < std::min(halfmass_rad_type[i].size(), size_t(6)); j++) {
                s.halfmass_rad_type[j] = halfmass_rad_type[i][j];
            }
        }
    }
    
    return subhalos;
}

Gadget4HaloCatalog read_gadget4_halo_catalog_file(
    const std::string& filename,
    bool read_ids
) {
    Gadget4HaloCatalog catalog;
    
    catalog.header = read_gadget4_halo_header(filename);
    catalog.groups = read_gadget4_groups(filename);
    catalog.subhalos = read_gadget4_subhalos(filename);
    
    if (read_ids) {
        HighFive::File file(filename, HighFive::File::ReadOnly);
        if (file.exist("IDs")) {
            auto ids_group = file.getGroup("IDs");
            if (has_dataset(ids_group, "ParticleIDs")) {
                try {
                    catalog.particle_ids = read_1d_dataset<uint64_t>(
                        ids_group.getDataSet("ParticleIDs"));
                } catch (...) {
                    auto ids32 = read_1d_dataset<uint32_t>(
                        ids_group.getDataSet("ParticleIDs"));
                    catalog.particle_ids.resize(ids32.size());
                    for (size_t i = 0; i < ids32.size(); i++) {
                        catalog.particle_ids[i] = ids32[i];
                    }
                }
            }
        }
    }
    
    return catalog;
}

// =============================================================================
// Distributed file reading
// =============================================================================

namespace {

/**
 * Extract file index from a distributed filename.
 * E.g., "snapshot_009.5.hdf5" -> 5
 */
int extract_file_index(const std::string& filename, const std::string& prefix, int snapshot_num) {
    // Build expected pattern: prefix_NNN.X.hdf5
    std::ostringstream pattern_ss;
    pattern_ss << prefix << "_" << std::setfill('0') << std::setw(3) << snapshot_num << "\\.";
    std::regex pattern(pattern_ss.str() + "(\\d+)\\.hdf5$");
    
    std::smatch match;
    std::string fname = std::filesystem::path(filename).filename().string();
    if (std::regex_search(fname, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

/**
 * Generic function to discover files matching a pattern in a directory.
 */
std::vector<std::string> discover_files(
    const std::string& directory,
    const std::string& prefix,
    int snapshot_num
) {
    namespace fs = std::filesystem;
    
    std::vector<std::pair<int, std::string>> indexed_files;
    
    // Build pattern: prefix_NNN.*.hdf5
    std::ostringstream pattern_ss;
    pattern_ss << prefix << "_" << std::setfill('0') << std::setw(3) << snapshot_num << "\\.\\d+\\.hdf5$";
    std::regex pattern(pattern_ss.str());
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        
        std::string fname = entry.path().filename().string();
        if (std::regex_search(fname, pattern)) {
            int idx = extract_file_index(entry.path().string(), prefix, snapshot_num);
            if (idx >= 0) {
                indexed_files.emplace_back(idx, entry.path().string());
            }
        }
    }
    
    // Sort by file index
    std::sort(indexed_files.begin(), indexed_files.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Extract just the paths
    std::vector<std::string> result;
    result.reserve(indexed_files.size());
    for (const auto& [idx, path] : indexed_files) {
        result.push_back(path);
    }
    
    return result;
}

/**
 * Append particles from src to dst.
 */
void append_particles(Gadget4Particles& dst, const Gadget4Particles& src) {
    // Coordinates
    dst.coordinates.insert(dst.coordinates.end(),
                           src.coordinates.begin(), src.coordinates.end());
    
    // Velocities
    dst.velocities.insert(dst.velocities.end(),
                          src.velocities.begin(), src.velocities.end());
    
    // Particle IDs
    dst.particle_ids.insert(dst.particle_ids.end(),
                            src.particle_ids.begin(), src.particle_ids.end());
    
    // Optional fields
    dst.accelerations.insert(dst.accelerations.end(),
                             src.accelerations.begin(), src.accelerations.end());
    dst.potential.insert(dst.potential.end(),
                         src.potential.begin(), src.potential.end());
    dst.subfind_density.insert(dst.subfind_density.end(),
                               src.subfind_density.begin(), src.subfind_density.end());
    dst.subfind_hsml.insert(dst.subfind_hsml.end(),
                            src.subfind_hsml.begin(), src.subfind_hsml.end());
    dst.subfind_vel_disp.insert(dst.subfind_vel_disp.end(),
                                src.subfind_vel_disp.begin(), src.subfind_vel_disp.end());
}

} // anonymous namespace

std::vector<std::string> discover_snapshot_files(
    const std::string& directory,
    int snapshot_num
) {
    return discover_files(directory, "snapshot", snapshot_num);
}

std::vector<std::string> discover_halo_catalog_files(
    const std::string& directory,
    int snapshot_num
) {
    return discover_files(directory, "fof_subhalo_tab", snapshot_num);
}

Gadget4Snapshot read_gadget4_snapshot_distributed(
    const std::string& directory,
    int snapshot_num,
    uint32_t particle_types,
    bool read_velocities,
    bool read_ids
) {
    auto files = discover_snapshot_files(directory, snapshot_num);
    
    if (files.empty()) {
        std::ostringstream ss;
        ss << "No snapshot files found in " << directory 
           << " for snapshot " << std::setfill('0') << std::setw(3) << snapshot_num;
        throw std::runtime_error(ss.str());
    }
    
    // Read first file to get header
    Gadget4Snapshot result = read_gadget4_snapshot_file(files[0], particle_types, 
                                                         read_velocities, read_ids);
    
    // The header from the first file already has NumPart_Total set correctly,
    // but NumPart_ThisFile is only for that file. We'll update it at the end.
    
    // Read and append remaining files
    for (size_t i = 1; i < files.size(); i++) {
        auto partial = read_gadget4_snapshot_file(files[i], particle_types,
                                                   read_velocities, read_ids);
        
        // Append particles from each type
        for (int ptype = 0; ptype < 6; ptype++) {
            append_particles(result.particles[ptype], partial.particles[ptype]);
        }
    }
    
    // Update num_part_this_file to reflect all particles now in the snapshot
    for (int ptype = 0; ptype < 6; ptype++) {
        result.header.num_part_this_file[ptype] = result.particles[ptype].size();
    }
    
    return result;
}

Gadget4HaloCatalog read_gadget4_halo_catalog_distributed(
    const std::string& directory,
    int snapshot_num,
    bool read_ids
) {
    auto files = discover_halo_catalog_files(directory, snapshot_num);
    
    if (files.empty()) {
        std::ostringstream ss;
        ss << "No halo catalog files found in " << directory 
           << " for snapshot " << std::setfill('0') << std::setw(3) << snapshot_num;
        throw std::runtime_error(ss.str());
    }
    
    // Read all files first to collect groups and subhalos
    struct FileData {
        std::vector<Gadget4Group> groups;
        std::vector<Gadget4Subhalo> subhalos;
        std::vector<uint64_t> particle_ids;
        int64_t group_offset = 0;   // Global offset for groups in this file
        int64_t subhalo_offset = 0; // Global offset for subhalos in this file
    };
    
    std::vector<FileData> file_data(files.size());
    
    // Track global offsets
    int64_t total_groups = 0;
    int64_t total_subhalos = 0;
    int64_t total_ids = 0;
    
    // Read header from first file
    Gadget4HaloCatalogHeader header = read_gadget4_halo_header(files[0]);
    
    // Read each file
    for (size_t i = 0; i < files.size(); i++) {
        file_data[i].groups = read_gadget4_groups(files[i]);
        file_data[i].subhalos = read_gadget4_subhalos(files[i]);
        file_data[i].group_offset = total_groups;
        file_data[i].subhalo_offset = total_subhalos;
        
        total_groups += file_data[i].groups.size();
        total_subhalos += file_data[i].subhalos.size();
        
        if (read_ids) {
            HighFive::File hfile(files[i], HighFive::File::ReadOnly);
            if (hfile.exist("IDs")) {
                auto ids_group = hfile.getGroup("IDs");
                if (has_dataset(ids_group, "ParticleIDs")) {
                    try {
                        file_data[i].particle_ids = read_1d_dataset<uint64_t>(
                            ids_group.getDataSet("ParticleIDs"));
                    } catch (...) {
                        auto ids32 = read_1d_dataset<uint32_t>(
                            ids_group.getDataSet("ParticleIDs"));
                        file_data[i].particle_ids.resize(ids32.size());
                        for (size_t j = 0; j < ids32.size(); j++) {
                            file_data[i].particle_ids[j] = ids32[j];
                        }
                    }
                    total_ids += file_data[i].particle_ids.size();
                }
            }
        }
    }
    
    // Build result catalog
    Gadget4HaloCatalog result;
    result.header = header;
    result.header.n_groups_this_file = total_groups;
    result.header.n_subhalos_this_file = total_subhalos;
    result.header.n_ids_this_file = total_ids;
    
    // Reserve space
    result.groups.reserve(total_groups);
    result.subhalos.reserve(total_subhalos);
    if (read_ids) {
        result.particle_ids.reserve(total_ids);
    }
    
    // Merge groups, updating first_sub pointers
    for (size_t i = 0; i < files.size(); i++) {
        for (auto& grp : file_data[i].groups) {
            // Update first_sub to global index
            if (grp.first_sub >= 0) {
                grp.first_sub += file_data[i].subhalo_offset;
            }
            result.groups.push_back(std::move(grp));
        }
    }
    
    // Merge subhalos, updating group_nr pointers
    for (size_t i = 0; i < files.size(); i++) {
        for (auto& sub : file_data[i].subhalos) {
            // group_nr is already a global index in GADGET-4, no need to update
            result.subhalos.push_back(std::move(sub));
        }
    }
    
    // Merge particle IDs
    if (read_ids) {
        for (size_t i = 0; i < files.size(); i++) {
            result.particle_ids.insert(result.particle_ids.end(),
                                       file_data[i].particle_ids.begin(),
                                       file_data[i].particle_ids.end());
        }
    }
    
    // Sort groups by their original file order (they should already be in order)
    // GADGET-4 writes groups in order across files, so simple concatenation works.
    // However, if sorting is needed for subhalos by group_nr:
    
    // Create indices for sorting subhalos by (group_nr, rank_in_gr)
    std::vector<size_t> subhalo_order(result.subhalos.size());
    std::iota(subhalo_order.begin(), subhalo_order.end(), 0);
    
    std::sort(subhalo_order.begin(), subhalo_order.end(),
              [&result](size_t a, size_t b) {
                  const auto& sa = result.subhalos[a];
                  const auto& sb = result.subhalos[b];
                  if (sa.group_nr != sb.group_nr) {
                      return sa.group_nr < sb.group_nr;
                  }
                  return sa.rank_in_gr < sb.rank_in_gr;
              });
    
    // Reorder subhalos according to sorted order
    std::vector<Gadget4Subhalo> sorted_subhalos(result.subhalos.size());
    for (size_t i = 0; i < subhalo_order.size(); i++) {
        sorted_subhalos[i] = std::move(result.subhalos[subhalo_order[i]]);
    }
    result.subhalos = std::move(sorted_subhalos);
    
    // Update first_sub in groups to point to correct positions after sorting
    // Build a map from group_nr to first subhalo index
    std::vector<int64_t> group_first_sub(result.groups.size(), -1);
    for (size_t i = 0; i < result.subhalos.size(); i++) {
        int64_t gn = result.subhalos[i].group_nr;
        if (gn >= 0 && static_cast<size_t>(gn) < group_first_sub.size()) {
            if (group_first_sub[gn] < 0) {
                group_first_sub[gn] = static_cast<int64_t>(i);
            }
        }
    }
    
    // Update groups with correct first_sub
    for (size_t i = 0; i < result.groups.size(); i++) {
        result.groups[i].first_sub = group_first_sub[i];
    }
    
    return result;
}

// =============================================================================
// Unified interface functions
// =============================================================================

namespace {

/**
 * Extract snapshot number from a directory name like "snapdir_009" or "snapdir_123".
 * Returns -1 if unable to parse.
 */
int extract_snapshot_num_from_dir(const std::string& dirname) {
    std::regex pattern("snapdir_(\\d+)$");
    std::smatch match;
    if (std::regex_search(dirname, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

/**
 * Extract snapshot number from a filename like "snapshot_034.hdf5" or "fof_subhalo_tab_034.hdf5".
 * Returns -1 if unable to parse.
 */
int extract_snapshot_num_from_file(const std::string& filename) {
    // Try snapshot pattern
    std::regex snap_pattern("snapshot_(\\d+)\\.hdf5$");
    std::smatch match;
    std::string fname = std::filesystem::path(filename).filename().string();
    if (std::regex_search(fname, match, snap_pattern)) {
        return std::stoi(match[1].str());
    }
    
    // Try halo catalog pattern
    std::regex halo_pattern("fof_subhalo_tab_(\\d+)\\.hdf5$");
    if (std::regex_search(fname, match, halo_pattern)) {
        return std::stoi(match[1].str());
    }
    
    return -1;
}

} // anonymous namespace

Gadget4Snapshot read_gadget4_snapshot(
    const std::string& path,
    int snapshot_num,
    uint32_t particle_types,
    bool read_velocities,
    bool read_ids
) {
    namespace fs = std::filesystem;
    
    // Check if path is a file or directory
    if (fs::is_regular_file(path)) {
        // Single file - use direct read
        return read_gadget4_snapshot_file(path, particle_types, read_velocities, read_ids);
    }
    
    if (fs::is_directory(path)) {
        // Directory - use distributed read
        int snap_num = snapshot_num;
        
        // Try to infer snapshot number from directory name if not provided
        if (snap_num < 0) {
            std::string dirname = fs::path(path).filename().string();
            snap_num = extract_snapshot_num_from_dir(dirname);
        }
        
        if (snap_num < 0) {
            throw std::runtime_error(
                "Cannot infer snapshot number from directory name. "
                "Please provide snapshot_num explicitly.");
        }
        
        return read_gadget4_snapshot_distributed(path, snap_num, particle_types,
                                                  read_velocities, read_ids);
    }
    
    throw std::runtime_error("Path does not exist or is not a file/directory: " + path);
}

Gadget4HaloCatalog read_gadget4_halo_catalog(
    const std::string& path,
    int snapshot_num,
    bool read_ids
) {
    namespace fs = std::filesystem;
    
    // Check if path is a file or directory
    if (fs::is_regular_file(path)) {
        // Single file - use direct read
        return read_gadget4_halo_catalog_file(path, read_ids);
    }
    
    if (fs::is_directory(path)) {
        // Directory - use distributed read
        int snap_num = snapshot_num;
        
        // Try to infer snapshot number from directory name if not provided
        if (snap_num < 0) {
            std::string dirname = fs::path(path).filename().string();
            snap_num = extract_snapshot_num_from_dir(dirname);
        }
        
        if (snap_num < 0) {
            throw std::runtime_error(
                "Cannot infer snapshot number from directory name. "
                "Please provide snapshot_num explicitly.");
        }
        
        return read_gadget4_halo_catalog_distributed(path, snap_num, read_ids);
    }
    
    throw std::runtime_error("Path does not exist or is not a file/directory: " + path);
}

} // namespace io
} // namespace asymptotic_tetra
