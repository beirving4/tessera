#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "io/headers.h"

#ifdef TESSERA_HAS_HDF5
#include "io/gadget4.h"
#include "io/merger_tree.h"
#endif

namespace py = pybind11;
using namespace tessera::io;

void bind_io(py::module& m) {
    // ByteOrder enum
    py::enum_<ByteOrder>(m, "ByteOrder", "Byte order for binary I/O")
        .value("Little", ByteOrder::Little)
        .value("Big", ByteOrder::Big);

    // CosmologyHeader
    py::class_<CosmologyHeader>(m, "CosmologyHeader", "Cosmological parameters")
        .def(py::init<>())
        .def_readwrite("z", &CosmologyHeader::z, "Redshift")
        .def_readwrite("omega_m", &CosmologyHeader::omega_m, "Matter density")
        .def_readwrite("omega_l", &CosmologyHeader::omega_l, "Dark energy density")
        .def_readwrite("h100", &CosmologyHeader::h100, "Hubble parameter (H0/100)")
        .def("__repr__", [](const CosmologyHeader& h) {
            return "CosmologyHeader(z=" + std::to_string(h.z) + 
                   ", omega_m=" + std::to_string(h.omega_m) +
                   ", omega_l=" + std::to_string(h.omega_l) + ")";
        });

    // CatalogHeader
    py::class_<CatalogHeader>(m, "CatalogHeader", "Particle catalog metadata")
        .def(py::init<>())
        .def_readwrite("cosmo", &CatalogHeader::cosmo)
        .def_readwrite("mass", &CatalogHeader::mass)
        .def_readwrite("count", &CatalogHeader::count)
        .def_readwrite("total_count", &CatalogHeader::total_count)
        .def_readwrite("count_width", &CatalogHeader::count_width)
        .def_readwrite("total_width", &CatalogHeader::total_width);

    // SheetHeader
    py::class_<SheetHeader>(m, "SheetHeader", "Phase sheet file header")
        .def(py::init<>())
        .def_readwrite("cosmo", &SheetHeader::cosmo)
        .def_readwrite("count", &SheetHeader::count)
        .def_readwrite("count_width", &SheetHeader::count_width)
        .def_readwrite("segment_width", &SheetHeader::segment_width)
        .def_readwrite("grid_width", &SheetHeader::grid_width)
        .def_readwrite("grid_count", &SheetHeader::grid_count)
        .def_readwrite("idx", &SheetHeader::idx)
        .def_readwrite("cells", &SheetHeader::cells)
        .def_readwrite("mass", &SheetHeader::mass)
        .def_readwrite("total_width", &SheetHeader::total_width)
        .def_readwrite("origin", &SheetHeader::origin)
        .def_readwrite("width", &SheetHeader::width)
        .def("cell_bounds", &SheetHeader::cell_bounds, py::arg("cells"));

    // GadgetHeader
    py::class_<GadgetHeader>(m, "GadgetHeader", "Gadget-2 file header")
        .def(py::init<>())
        .def_readonly("box_size", &GadgetHeader::box_size)
        .def_readonly("redshift", &GadgetHeader::redshift)
        .def_readonly("omega0", &GadgetHeader::omega0)
        .def_readonly("omega_lambda", &GadgetHeader::omega_lambda)
        .def_readonly("hubble_param", &GadgetHeader::hubble_param)
        .def("standardize", &GadgetHeader::standardize)
        .def("wrap_distance", &GadgetHeader::wrap_distance, py::arg("x"));

    // Utility functions
    m.def("is_system_little_endian", &is_system_little_endian,
          "Check if system uses little-endian byte order");
    m.def("system_byte_order", &system_byte_order,
          "Get system's native byte order");

#ifdef TESSERA_HAS_HDF5
    // =========================================================================
    // GADGET-4 HDF5 I/O
    // =========================================================================

    // Gadget4Header
    py::class_<Gadget4Header>(m, "Gadget4Header",
        R"pbdoc(
        GADGET-4 snapshot header.
        
        GADGET-4 Internal Code Units
        ----------------------------
        All values are in GADGET-4 internal code units:
        
        - Mass: 10^10 M_sun/h
        - Length: kpc/h or Mpc/h (depends on simulation)
        - Velocity: km/s
        
        To convert to physical units:
        - mass_physical = mass_code * 1e10 / h  [M_sun]
        - length_physical = length_code / h     [kpc or Mpc]
        
        where h = H0 / 100 km/s/Mpc (typically ~0.67-0.70)
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("box_size", &Gadget4Header::box_size, "Box size in code length units (kpc/h or Mpc/h)")
        .def_readwrite("time", &Gadget4Header::time, "Scale factor a (cosmological) or time")
        .def_readwrite("redshift", &Gadget4Header::redshift, "Redshift z = 1/a - 1")
        .def_readwrite("mass_table", &Gadget4Header::mass_table, "Mass per particle type (10^10 M_sun/h)")
        .def_readwrite("num_part_this_file", &Gadget4Header::num_part_this_file)
        .def_readwrite("num_part_total", &Gadget4Header::num_part_total)
        .def_readwrite("num_files_per_snapshot", &Gadget4Header::num_files_per_snapshot)
        .def_readwrite("git_commit", &Gadget4Header::git_commit)
        .def_readwrite("git_date", &Gadget4Header::git_date)
        .def("total_particles", &Gadget4Header::total_particles,
             "Get total particle count across all types")
        .def("particles_of_type", &Gadget4Header::particles_of_type, py::arg("type"),
             "Get particle count for a specific type (0-5)")
        .def("__repr__", [](const Gadget4Header& h) {
            return "Gadget4Header(box_size=" + std::to_string(h.box_size) +
                   ", z=" + std::to_string(h.redshift) +
                   ", total=" + std::to_string(h.total_particles()) + ")";
        });

    // Gadget4Particles
    py::class_<Gadget4Particles>(m, "Gadget4Particles", "Particle data from GADGET-4 snapshot")
        .def(py::init<>())
        .def_readwrite("coordinates", &Gadget4Particles::coordinates)
        .def_readwrite("velocities", &Gadget4Particles::velocities)
        .def_readwrite("particle_ids", &Gadget4Particles::particle_ids)
        .def_readwrite("accelerations", &Gadget4Particles::accelerations)
        .def_readwrite("potential", &Gadget4Particles::potential)
        .def_readwrite("subfind_density", &Gadget4Particles::subfind_density)
        .def_readwrite("subfind_hsml", &Gadget4Particles::subfind_hsml)
        .def_readwrite("subfind_vel_disp", &Gadget4Particles::subfind_vel_disp)
        .def("size", &Gadget4Particles::size, "Number of particles")
        .def("has_coordinates", &Gadget4Particles::has_coordinates)
        .def("has_velocities", &Gadget4Particles::has_velocities)
        .def("__len__", &Gadget4Particles::size);

    // Gadget4Snapshot
    py::class_<Gadget4Snapshot>(m, "Gadget4Snapshot", "Complete GADGET-4 snapshot data")
        .def(py::init<>())
        .def_readwrite("header", &Gadget4Snapshot::header)
        .def_readwrite("particles", &Gadget4Snapshot::particles)
        .def("dark_matter", py::overload_cast<>(&Gadget4Snapshot::dark_matter),
             py::return_value_policy::reference_internal,
             "Get dark matter particles (type 1)")
        .def("gas", py::overload_cast<>(&Gadget4Snapshot::gas),
             py::return_value_policy::reference_internal,
             "Get gas particles (type 0)");

    // Gadget4Group
    py::class_<Gadget4Group>(m, "Gadget4Group",
        R"pbdoc(
        FOF Group (halo) from GADGET-4 halo catalog.
        
        Units (GADGET-4 internal code units):
        - Positions: kpc/h or Mpc/h
        - Velocities: km/s
        - Masses: 10^10 M_sun/h
        - Radii: kpc/h or Mpc/h
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("pos", &Gadget4Group::pos, "Center of mass position (code length units)")
        .def_readwrite("vel", &Gadget4Group::vel, "Center of mass velocity (km/s)")
        .def_readwrite("mass", &Gadget4Group::mass, "Total FOF mass (10^10 M_sun/h)")
        .def_readwrite("len", &Gadget4Group::len, "Number of particles")
        .def_readwrite("mass_type", &Gadget4Group::mass_type, "Mass by particle type")
        .def_readwrite("len_type", &Gadget4Group::len_type, "Particles by type")
        .def_readwrite("m_crit200", &Gadget4Group::m_crit200, "M200 critical")
        .def_readwrite("r_crit200", &Gadget4Group::r_crit200, "R200 critical")
        .def_readwrite("m_crit500", &Gadget4Group::m_crit500, "M500 critical")
        .def_readwrite("r_crit500", &Gadget4Group::r_crit500, "R500 critical")
        .def_readwrite("m_mean200", &Gadget4Group::m_mean200, "M200 mean")
        .def_readwrite("r_mean200", &Gadget4Group::r_mean200, "R200 mean")
        .def_readwrite("m_tophat200", &Gadget4Group::m_tophat200, "M_tophat")
        .def_readwrite("r_tophat200", &Gadget4Group::r_tophat200, "R_tophat")
        .def_readwrite("first_sub", &Gadget4Group::first_sub, "Index of first subhalo")
        .def_readwrite("n_subs", &Gadget4Group::n_subs, "Number of subhalos")
        .def_readwrite("ascale", &Gadget4Group::ascale, "Scale factor when identified")
        .def("__repr__", [](const Gadget4Group& g) {
            return "Gadget4Group(mass=" + std::to_string(g.mass) +
                   ", len=" + std::to_string(g.len) +
                   ", n_subs=" + std::to_string(g.n_subs) + ")";
        });

    // Gadget4Subhalo
    py::class_<Gadget4Subhalo>(m, "Gadget4Subhalo",
        R"pbdoc(
        Subhalo from GADGET-4 halo catalog.
        
        Units (GADGET-4 internal code units):
        - Positions: kpc/h or Mpc/h
        - Velocities: km/s
        - Masses: 10^10 M_sun/h
        - Radii: kpc/h or Mpc/h
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("pos", &Gadget4Subhalo::pos, "Position (code length units)")
        .def_readwrite("vel", &Gadget4Subhalo::vel, "Velocity (km/s)")
        .def_readwrite("cm", &Gadget4Subhalo::cm, "Center of mass (code length units)")
        .def_readwrite("spin", &Gadget4Subhalo::spin, "Angular momentum")
        .def_readwrite("mass", &Gadget4Subhalo::mass, "Total bound mass (10^10 M_sun/h)")
        .def_readwrite("len", &Gadget4Subhalo::len, "Number of bound particles")
        .def_readwrite("mass_type", &Gadget4Subhalo::mass_type)
        .def_readwrite("len_type", &Gadget4Subhalo::len_type)
        .def_readwrite("halfmass_rad", &Gadget4Subhalo::halfmass_rad, "Half-mass radius")
        .def_readwrite("halfmass_rad_type", &Gadget4Subhalo::halfmass_rad_type)
        .def_readwrite("vmax", &Gadget4Subhalo::vmax, "Maximum circular velocity")
        .def_readwrite("vmax_rad", &Gadget4Subhalo::vmax_rad, "Radius of Vmax")
        .def_readwrite("vel_disp", &Gadget4Subhalo::vel_disp, "Velocity dispersion")
        .def_readwrite("group_nr", &Gadget4Subhalo::group_nr, "Parent group index")
        .def_readwrite("rank_in_gr", &Gadget4Subhalo::rank_in_gr, "Rank within parent group")
        .def_readwrite("parent_rank", &Gadget4Subhalo::parent_rank, "Parent subhalo rank")
        .def_readwrite("id_mostbound", &Gadget4Subhalo::id_mostbound, "ID of most bound particle")
        .def("__repr__", [](const Gadget4Subhalo& s) {
            return "Gadget4Subhalo(mass=" + std::to_string(s.mass) +
                   ", len=" + std::to_string(s.len) +
                   ", vmax=" + std::to_string(s.vmax) + ")";
        });

    // Gadget4HaloCatalogHeader
    py::class_<Gadget4HaloCatalogHeader>(m, "Gadget4HaloCatalogHeader",
        "GADGET-4 halo catalog header")
        .def(py::init<>())
        .def_readwrite("box_size", &Gadget4HaloCatalogHeader::box_size)
        .def_readwrite("time", &Gadget4HaloCatalogHeader::time)
        .def_readwrite("redshift", &Gadget4HaloCatalogHeader::redshift)
        .def_readwrite("n_groups_this_file", &Gadget4HaloCatalogHeader::n_groups_this_file)
        .def_readwrite("n_groups_total", &Gadget4HaloCatalogHeader::n_groups_total)
        .def_readwrite("n_subhalos_this_file", &Gadget4HaloCatalogHeader::n_subhalos_this_file)
        .def_readwrite("n_subhalos_total", &Gadget4HaloCatalogHeader::n_subhalos_total)
        .def_readwrite("n_ids_this_file", &Gadget4HaloCatalogHeader::n_ids_this_file)
        .def_readwrite("n_ids_total", &Gadget4HaloCatalogHeader::n_ids_total)
        .def_readwrite("num_files", &Gadget4HaloCatalogHeader::num_files)
        .def_readwrite("git_commit", &Gadget4HaloCatalogHeader::git_commit)
        .def_readwrite("git_date", &Gadget4HaloCatalogHeader::git_date);

    // Gadget4HaloCatalog
    py::class_<Gadget4HaloCatalog>(m, "Gadget4HaloCatalog",
        "Complete GADGET-4 halo catalog")
        .def(py::init<>())
        .def_readwrite("header", &Gadget4HaloCatalog::header)
        .def_readwrite("groups", &Gadget4HaloCatalog::groups)
        .def_readwrite("subhalos", &Gadget4HaloCatalog::subhalos)
        .def_readwrite("particle_ids", &Gadget4HaloCatalog::particle_ids)
        .def("num_groups", &Gadget4HaloCatalog::num_groups, "Number of groups")
        .def("num_subhalos", &Gadget4HaloCatalog::num_subhalos, "Number of subhalos")
        .def("__repr__", [](const Gadget4HaloCatalog& c) {
            return "Gadget4HaloCatalog(groups=" + std::to_string(c.num_groups()) +
                   ", subhalos=" + std::to_string(c.num_subhalos()) + ")";
        });

    // =========================================================================
    // Read functions - single file (low-level)
    // =========================================================================
    
    m.def("read_gadget4_header", &read_gadget4_header, py::arg("filename"),
          "Read GADGET-4 snapshot header");

    m.def("read_gadget4_snapshot_file", &read_gadget4_snapshot_file,
          py::arg("filename"),
          py::arg("particle_types") = 0x3F,
          py::arg("read_velocities") = true,
          py::arg("read_ids") = true,
          "Read complete GADGET-4 snapshot from a single file");

    m.def("read_gadget4_positions", &read_gadget4_positions,
          py::arg("filename"),
          py::arg("particle_type") = 1,
          "Read only positions from GADGET-4 snapshot");

    m.def("read_gadget4_halo_header", &read_gadget4_halo_header, py::arg("filename"),
          "Read GADGET-4 halo catalog header");

    m.def("read_gadget4_halo_catalog_file", &read_gadget4_halo_catalog_file,
          py::arg("filename"),
          py::arg("read_ids") = false,
          "Read complete GADGET-4 halo catalog from a single file");

    m.def("read_gadget4_groups", &read_gadget4_groups, py::arg("filename"),
          "Read only FOF groups from GADGET-4 halo catalog");

    m.def("read_gadget4_subhalos", &read_gadget4_subhalos, py::arg("filename"),
          "Read only subhalos from GADGET-4 halo catalog");

    // =========================================================================
    // Distributed file reading (multi-file snapshots/catalogs)
    // =========================================================================
    
    m.def("discover_snapshot_files", &discover_snapshot_files,
          py::arg("directory"),
          py::arg("snapshot_num"),
          "Discover distributed snapshot files in a directory.\n"
          "Returns sorted list of file paths for snapshot_NNN.*.hdf5 files.");

    m.def("discover_halo_catalog_files", &discover_halo_catalog_files,
          py::arg("directory"),
          py::arg("snapshot_num"),
          "Discover distributed halo catalog files in a directory.\n"
          "Returns sorted list of file paths for fof_subhalo_tab_NNN.*.hdf5 files.");

    m.def("read_gadget4_snapshot_distributed", &read_gadget4_snapshot_distributed,
          py::arg("directory"),
          py::arg("snapshot_num"),
          py::arg("particle_types") = 0x3F,
          py::arg("read_velocities") = true,
          py::arg("read_ids") = true,
          "Read a distributed GADGET-4 snapshot from multiple files.\n"
          "Particle data from all files is concatenated.");

    m.def("read_gadget4_halo_catalog_distributed", &read_gadget4_halo_catalog_distributed,
          py::arg("directory"),
          py::arg("snapshot_num"),
          py::arg("read_ids") = false,
          "Read a distributed GADGET-4 halo catalog from multiple files.\n"
          "Groups and subhalos are merged and sorted by their global indices.");

    // =========================================================================
    // Unified interface functions (auto-detect single vs distributed)
    // =========================================================================
    
    m.def("read_gadget4_snapshot", &read_gadget4_snapshot,
          py::arg("path"),
          py::arg("snapshot_num") = -1,
          py::arg("particle_types") = 0x3F,
          py::arg("read_velocities") = true,
          py::arg("read_ids") = true,
          "Read a GADGET-4 snapshot, auto-detecting single file vs distributed.\n"
          "\n"
          "Parameters:\n"
          "  path: Path to a single HDF5 file or a directory containing distributed files\n"
          "  snapshot_num: Snapshot number (inferred from directory name if -1)\n"
          "  particle_types: Bitmask of particle types to read (default: all)\n"
          "  read_velocities: Whether to read velocity data\n"
          "  read_ids: Whether to read particle IDs");

    m.def("read_gadget4_halo_catalog", &read_gadget4_halo_catalog,
          py::arg("path"),
          py::arg("snapshot_num") = -1,
          py::arg("read_ids") = false,
          "Read a GADGET-4 halo catalog, auto-detecting single file vs distributed.\n"
          "\n"
          "Parameters:\n"
          "  path: Path to a single HDF5 file or a directory containing distributed files\n"
          "  snapshot_num: Snapshot number (inferred from directory name if -1)\n"
          "  read_ids: Whether to read particle IDs");

    // =========================================================================
    // Merger Tree I/O
    // =========================================================================

    // MergerTreeHeader
    py::class_<MergerTreeHeader>(m, "MergerTreeHeader",
        R"pbdoc(
        Merger tree file header information.

        Loaded immediately when MergerTree is constructed (small data).

        Units (GADGET-4 internal code units):
        - box_size: Mpc/h
        - Masses: 10^10 M_sun/h
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("n_trees", &MergerTreeHeader::n_trees, "Number of trees (z=0 halos)")
        .def_readwrite("n_halos", &MergerTreeHeader::n_halos, "Total number of halos")
        .def_readwrite("n_files", &MergerTreeHeader::n_files, "Number of distributed files")
        .def_readwrite("last_snap", &MergerTreeHeader::last_snap, "Last snapshot number")
        .def_readwrite("snap_times", &MergerTreeHeader::snap_times, "Scale factor at each snapshot")
        .def_readwrite("snap_redshifts", &MergerTreeHeader::snap_redshifts, "Redshift at each snapshot")
        .def_readwrite("box_size", &MergerTreeHeader::box_size, "Simulation box size [Mpc/h]")
        .def_readwrite("hubble", &MergerTreeHeader::hubble, "Hubble parameter h")
        .def_readwrite("omega_matter", &MergerTreeHeader::omega_matter, "Matter density Omega_m")
        .def_readwrite("omega_lambda", &MergerTreeHeader::omega_lambda, "Dark energy density Omega_Lambda")
        .def("n_snapshots", &MergerTreeHeader::n_snapshots, "Number of snapshots")
        .def("get_scale_factor", &MergerTreeHeader::get_scale_factor, py::arg("snap_num"),
             "Get scale factor for a snapshot number")
        .def("get_redshift", &MergerTreeHeader::get_redshift, py::arg("snap_num"),
             "Get redshift for a snapshot number")
        .def("__repr__", [](const MergerTreeHeader& h) {
            return "MergerTreeHeader(n_trees=" + std::to_string(h.n_trees) +
                   ", n_halos=" + std::to_string(h.n_halos) +
                   ", box_size=" + std::to_string(h.box_size) + ")";
        });

    // HaloInfo
    py::class_<HaloInfo>(m, "HaloInfo",
        R"pbdoc(
        Information about a single halo at one snapshot.

        Units (GADGET-4 internal code units):
        - Positions: Mpc/h
        - Velocities: km/s
        - Masses: 10^10 M_sun/h
        - Radii: Mpc/h
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("tree_index", &HaloInfo::tree_index, "Index in TreeHalos arrays")
        .def_readwrite("snap_num", &HaloInfo::snap_num, "Snapshot number")
        .def_readwrite("subhalo_nr", &HaloInfo::subhalo_nr, "Subhalo index in catalog")
        .def_readwrite("group_nr", &HaloInfo::group_nr, "FOF group index")
        .def_readwrite("scale_factor", &HaloInfo::scale_factor, "Scale factor a = 1/(1+z)")
        .def_readwrite("redshift", &HaloInfo::redshift, "Redshift z")
        .def_readwrite("position", &HaloInfo::position, "Comoving position [Mpc/h]")
        .def_readwrite("velocity", &HaloInfo::velocity, "Velocity [km/s]")
        .def_readwrite("mass", &HaloInfo::mass, "Subhalo mass [10^10 M_sun/h]")
        .def_readwrite("m200c", &HaloInfo::m200c, "M200 critical [10^10 M_sun/h]")
        .def_readwrite("r200c", &HaloInfo::r200c, "R200 critical [Mpc/h]")
        .def("is_valid", &HaloInfo::is_valid, "Check if this is a valid halo")
        .def("__repr__", [](const HaloInfo& h) {
            return "HaloInfo(snap=" + std::to_string(h.snap_num) +
                   ", a=" + std::to_string(h.scale_factor) +
                   ", mass=" + std::to_string(h.mass) + ")";
        });

    // TreeTableEntry
    py::class_<TreeTableEntry>(m, "TreeTableEntry",
        "Tree table entry for locating halos within a specific tree")
        .def(py::init<>())
        .def_readwrite("start_offset", &TreeTableEntry::start_offset, "Start index in TreeHalos")
        .def_readwrite("length", &TreeTableEntry::length, "Number of halos in this tree")
        .def_readwrite("tree_id", &TreeTableEntry::tree_id, "Tree identifier");

    // MergerTree
    py::class_<MergerTree>(m, "MergerTree",
        R"pbdoc(
        GADGET-4 Merger Tree Reader.

        Efficient reader for GADGET-4 merger tree HDF5 files with lazy loading.
        Header information is loaded immediately; TreeTable and TreeHalos are
        loaded on first access to minimize memory usage.

        Uses SIMD-optimized search when AVX2 is available (8x parallel comparison).

        Examples
        --------
        >>> tree = ts.io.MergerTree("trees.hdf5")
        >>> print(f"Found {tree.n_trees()} trees")
        >>>
        >>> # Find a halo
        >>> idx = tree.find_halo_in_tree(snap_num=74, subhalo_nr=0)
        >>> if idx is not None:
        ...     info = tree.make_halo_info(idx)
        ...     print(f"Mass: {info.mass}")
        )pbdoc")
        .def(py::init<const std::string&>(), py::arg("filename"),
             "Construct merger tree reader from HDF5 file")
        .def("header", &MergerTree::header, py::return_value_policy::reference_internal,
             "Get header information")
        .def("n_trees", &MergerTree::n_trees, "Number of trees (z=0 halos)")
        .def("n_halos", &MergerTree::n_halos, "Total number of halos")
        .def("n_snapshots", &MergerTree::n_snapshots, "Number of snapshots")
        .def("box_size", &MergerTree::box_size, "Simulation box size [Mpc/h]")
        .def("get_scale_factor", &MergerTree::get_scale_factor, py::arg("snap_num"),
             "Get scale factor for a snapshot number")
        .def("get_redshift", &MergerTree::get_redshift, py::arg("snap_num"),
             "Get redshift for a snapshot number")
        .def("find_nearest_snapshot", &MergerTree::find_nearest_snapshot,
             py::arg("scale_factor"),
             "Find snapshot number nearest to given scale factor")
        .def("get_tree_entry", &MergerTree::get_tree_entry, py::arg("tree_id"),
             py::return_value_policy::reference_internal,
             "Get tree table entry for a specific tree")
        .def("get_tree_indices", &MergerTree::get_tree_indices, py::arg("tree_id"),
             "Get (start_index, end_index) for halos in a specific tree")
        .def("get_root_halo_index", &MergerTree::get_root_halo_index, py::arg("tree_id"),
             "Get TreeHalos index of the root halo for a tree")
        .def("find_halo_in_tree", &MergerTree::find_halo_in_tree,
             py::arg("snap_num"),
             py::arg("subhalo_nr"),
             py::arg("tree_id") = std::nullopt,
             "Find TreeHalos index for a snapshot/subhalo combination.\n"
             "Returns None if not found. Uses SIMD-optimized search on AVX2 systems.")
        .def("make_halo_info", &MergerTree::make_halo_info, py::arg("tree_index"),
             "Create HaloInfo from a TreeHalos index")
        .def("search_data_loaded", &MergerTree::search_data_loaded,
             "Check if search data (snap_num, subhalo_nr) is loaded")
        .def("halos_loaded", &MergerTree::halos_loaded,
             "Check if full TreeHalos data is loaded")
        .def("table_loaded", &MergerTree::table_loaded,
             "Check if TreeTable data is loaded")
        .def("clear_cache", &MergerTree::clear_cache,
             "Clear cached TreeTable and TreeHalos data to free memory")
        .def("memory_usage", &MergerTree::memory_usage,
             "Get approximate memory usage in bytes")
        .def("__repr__", [](MergerTree& t) {
            return "MergerTree(n_trees=" + std::to_string(t.n_trees()) +
                   ", n_halos=" + std::to_string(t.n_halos()) +
                   ", loaded=" + (t.halos_loaded() ? "True" : "False") + ")";
        });

    // HaloTracker
    py::class_<HaloTracker>(m, "HaloTracker",
        R"pbdoc(
        Halo Branch Tracer.

        Traces progenitor and descendant branches through merger trees.
        Uses prefetching for efficient pointer-chasing through tree links.

        Examples
        --------
        >>> tree = ts.io.MergerTree("trees.hdf5")
        >>> tracker = ts.io.HaloTracker(tree)
        >>>
        >>> # Trace from z=0 most massive halo
        >>> branch = tracker.trace_main_branch(
        ...     snap_num=74,
        ...     subhalo_nr=0,
        ...     backward=True,
        ...     forward=False,
        ...     a_min=0.1
        ... )
        >>>
        >>> for halo in branch:
        ...     print(f"a={halo.scale_factor:.3f}: M200={halo.m200c:.2e}")
        )pbdoc")
        .def(py::init<MergerTree&>(), py::arg("tree"),
             py::keep_alive<1, 2>(),  // Keep tree alive as long as tracker exists
             "Construct halo tracker from merger tree")
        .def("trace_main_branch", &HaloTracker::trace_main_branch,
             py::arg("snap_num"),
             py::arg("subhalo_nr"),
             py::arg("backward") = true,
             py::arg("forward") = true,
             py::arg("a_min") = std::nullopt,
             py::arg("a_max") = std::nullopt,
             "Trace the main branch through a halo.\n\n"
             "Parameters:\n"
             "  snap_num: Reference snapshot number\n"
             "  subhalo_nr: Subhalo index at reference snapshot\n"
             "  backward: Trace progenitors (earlier times)\n"
             "  forward: Trace descendants (later times)\n"
             "  a_min: Minimum scale factor (optional)\n"
             "  a_max: Maximum scale factor (optional)\n\n"
             "Returns:\n"
             "  List of HaloInfo sorted by scale factor (early to late)")
        .def("trace_from_tree_id", &HaloTracker::trace_from_tree_id,
             py::arg("tree_id"),
             py::arg("backward") = true,
             py::arg("forward") = true,
             py::arg("a_min") = std::nullopt,
             py::arg("a_max") = std::nullopt,
             "Trace main branch starting from a tree's root halo.\n"
             "tree_id=0 is the most massive z=0 halo.")
        .def("trace_all_progenitors", &HaloTracker::trace_all_progenitors,
             py::arg("snap_num"),
             py::arg("subhalo_nr"),
             py::arg("a_min") = std::nullopt,
             "Trace ALL progenitor branches (not just main branch).\n\n"
             "Returns dict mapping snap_num -> list of HaloInfo at that snapshot.\n"
             "Useful for visualizing mergers.")
        .def("unwrap_coordinates", &HaloTracker::unwrap_coordinates,
             py::arg("branch"),
             py::arg("box_size") = std::nullopt,
             "Unwrap periodic coordinates for smooth tracking.\n"
             "Modifies positions in-place.");

    // Free function
    m.def("read_merger_tree_header", &read_merger_tree_header, py::arg("filename"),
          "Read merger tree header only (small, fast)");

    // Flag to indicate HDF5 support is available
    m.attr("HAS_HDF5") = true;
#else
    m.attr("HAS_HDF5") = false;
#endif
}
