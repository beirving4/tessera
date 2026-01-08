#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "halo/radius.h"
#include "halo/grid.h"
#include "halo/finder.h"
#include "halo/io.h"

namespace py = pybind11;
using namespace asymptotic_tetra::halo;
using namespace asymptotic_tetra;

void bind_halo(py::module& m) {
    // RadiusType enum
    py::enum_<RadiusType>(m, "RadiusType", "Halo radius definition type")
        .value("RVirial", RadiusType::RVirial, "Virial radius (Bryan & Norman 1998)")
        .value("R200c", RadiusType::R200c, "Radius enclosing 200x critical density")
        .value("R200m", RadiusType::R200m, "Radius enclosing 200x mean density")
        .value("R500c", RadiusType::R500c, "Radius enclosing 500x critical density")
        .value("R2500c", RadiusType::R2500c, "Radius enclosing 2500x critical density")
        .export_values();
    
    // Radius utility functions
    m.def("radius_from_string", [](const std::string& s) {
        auto [r, ok] = radius_from_string(s);
        if (!ok) throw std::runtime_error("Unknown radius type: " + s);
        return r;
    }, py::arg("s"),
    "Parse a radius type from string (e.g., '200m', 'vir', '500c')");
    
    m.def("radius_to_string", &radius_to_string, py::arg("r"),
          "Convert radius type to string representation");
    
    m.def("mass_to_radius", [](RadiusType r, const io::CosmologyHeader& cosmo,
                               const std::vector<double>& masses) {
        std::vector<double> radii(masses.size());
        mass_to_radius(r, cosmo, masses, radii);
        return radii;
    }, py::arg("radius_type"), py::arg("cosmo"), py::arg("masses"),
    "Convert masses to radii for given radius definition");
    
    m.def("radius_to_mass", [](RadiusType r, const io::CosmologyHeader& cosmo,
                               const std::vector<double>& radii) {
        std::vector<double> masses(radii.size());
        radius_to_mass(r, cosmo, radii, masses);
        return masses;
    }, py::arg("radius_type"), py::arg("cosmo"), py::arg("radii"),
    "Convert radii to masses for given radius definition");
    
    m.def("rockstar_column", &rockstar_column, py::arg("radius_type"),
          "Get Rockstar catalog column index for radius type");
    
    m.def("rockstar_is_mass", &rockstar_is_mass, py::arg("radius_type"),
          "Check if Rockstar stores mass (true) or radius (false) for this type");
    
    // Bounds struct
    py::class_<Bounds>(m, "Bounds", "Cell-aligned bounding box for spatial queries")
        .def(py::init<>())
        .def_readwrite("origin", &Bounds::origin)
        .def_readwrite("span", &Bounds::span)
        .def("sphere_bounds", &Bounds::sphere_bounds,
             py::arg("pos"), py::arg("r"), py::arg("cw"), py::arg("width"),
             "Create bounds around a sphere")
        .def("inside", &Bounds::inside,
             py::arg("val"), py::arg("width"), py::arg("dim"),
             "Check if value is inside bounds in given dimension");
    
    // HaloGrid class
    py::class_<HaloGrid>(m, "HaloGrid", "Spatial hash grid for halo neighbor lookups")
        .def(py::init<int, double, int>(),
             py::arg("num_cells"), py::arg("width"), py::arg("data_len"),
             "Create a spatial grid")
        .def_readwrite("cells", &HaloGrid::cells)
        .def_readwrite("cell_width", &HaloGrid::cell_width)
        .def_readwrite("box_width", &HaloGrid::box_width)
        .def("length", &HaloGrid::length, py::arg("idx"),
             "Get number of elements in a cell")
        .def("insert", &HaloGrid::insert,
             py::arg("xs"), py::arg("ys"), py::arg("zs"),
             "Insert elements based on positions")
        .def("total_cells", &HaloGrid::total_cells,
             "Get total number of cells")
        .def("max_length", &HaloGrid::max_length,
             "Get maximum elements in any cell")
        .def("average_length", &HaloGrid::average_length,
             "Get average elements per cell")
        .def("read_indices", &HaloGrid::read_indices,
             py::arg("idx"), py::arg("buf"),
             "Read element indices in a cell");
    
    // SubhaloFinder class
    py::class_<SubhaloFinder>(m, "SubhaloFinder", 
        "Find subhalo relationships based on position and radius overlap")
        .def(py::init<HaloGrid*>(), py::arg("grid"),
             py::keep_alive<1, 2>(),  // Keep grid alive while finder exists
             "Create finder for given spatial grid")
        .def("find_subhalos", &SubhaloFinder::find_subhalos,
             py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("rs"), py::arg("mult"),
             "Find all subhalo relationships")
        .def("start_end", &SubhaloFinder::start_end, py::arg("ih"),
             "Get start/end indices for halo's relationships")
        .def("intersect_count", &SubhaloFinder::intersect_count, py::arg("ih"),
             "Get number of intersecting halos")
        .def("host_count", &SubhaloFinder::host_count, py::arg("ih"),
             "Get number of host halos")
        .def("subhalo_count", &SubhaloFinder::subhalo_count, py::arg("ih"),
             "Get number of subhalos")
        .def("intersects", &SubhaloFinder::intersects, py::arg("ih"),
             "Get all intersecting halos")
        .def("hosts", &SubhaloFinder::hosts, py::arg("ih"),
             "Get all host halos")
        .def("subhalos", &SubhaloFinder::subhalos, py::arg("ih"),
             "Get all subhalos");
    
    // Utility function
    m.def("cumulative_max", &cumulative_max, py::arg("xs"),
          "Compute cumulative maximum from end to start");
    
    // Val enum for Rockstar columns
    py::enum_<Val>(m, "Val", "Rockstar catalog column indices")
        .value("Scale", Val::Scale)
        .value("ID", Val::ID)
        .value("DescScale", Val::DescScale)
        .value("DescID", Val::DescID)
        .value("NumProg", Val::NumProg)
        .value("PID", Val::PID)
        .value("UPID", Val::UPID)
        .value("DescPID", Val::DescPID)
        .value("Phantom", Val::Phantom)
        .value("SAMMVir", Val::SAMMVir)
        .value("MVir", Val::MVir)
        .value("RVir", Val::RVir)
        .value("Rs", Val::Rs)
        .value("Vrms", Val::Vrms)
        .value("MMP", Val::MMP)
        .value("ScaleOfLastMMP", Val::ScaleOfLastMMP)
        .value("VMax", Val::VMax)
        .value("X", Val::X)
        .value("Y", Val::Y)
        .value("Z", Val::Z)
        .value("Vx", Val::Vx)
        .value("Vy", Val::Vy)
        .value("Vz", Val::Vz)
        .value("Jx", Val::Jx)
        .value("Jy", Val::Jy)
        .value("Jz", Val::Jz)
        .value("Spin", Val::Spin)
        .value("BreadthFirstID", Val::BreadthFirstID)
        .value("DepthFirstID", Val::DepthFirstID)
        .value("TreeRootID", Val::TreeRootID)
        .value("OrigHaloID", Val::OrigHaloID)
        .value("SnapNum", Val::SnapNum)
        .value("NextCoprogenitorDepthFirstID", Val::NextCoprogenitorDepthFirstID)
        .value("LastProgenitorDepthFirstID", Val::LastProgenitorDepthFirstID)
        .value("RsKylpin", Val::RsKylpin)
        .value("MVirAll", Val::MVirAll)
        .value("M200b", Val::M200b)
        .value("M200c", Val::M200c)
        .value("M500c", Val::M500c)
        .value("M2500c", Val::M2500c)
        .value("XOff", Val::XOff)
        .value("Voff", Val::Voff)
        .value("SpinBullock", Val::SpinBullock)
        .value("BToA", Val::BToA)
        .value("CToA", Val::CToA)
        .value("Ax", Val::Ax)
        .value("Ay", Val::Ay)
        .value("Az", Val::Az)
        .value("BToA500c", Val::BToA500c)
        .value("CToA500c", Val::CToA500c)
        .value("Ax500c", Val::Ax500c)
        .value("Ay500c", Val::Ay500c)
        .value("Az500c", Val::Az500c)
        .value("TU", Val::TU)
        .value("MAcc", Val::MAcc)
        .value("MPeak", Val::MPeak)
        .value("VAcc", Val::VAcc)
        .value("VPeak", Val::VPeak)
        .value("HalfmassScale", Val::HalfmassScale)
        .value("AccRateInst", Val::AccRateInst)
        .value("AccRate100Myr", Val::AccRate100Myr)
        .value("AccRateTdyn", Val::AccRateTdyn)
        // Derived values
        .value("RadVir", Val::RadVir)
        .value("Rad200b", Val::Rad200b)
        .value("Rad200c", Val::Rad200c)
        .value("Rad500c", Val::Rad500c)
        .value("Rad2500c", Val::Rad2500c)
        .export_values();
    
    // HaloData struct
    py::class_<HaloData>(m, "HaloData", "Container for halo properties")
        .def(py::init<>())
        .def_readwrite("rids", &HaloData::rids, "Rockstar IDs")
        .def_readwrite("xs", &HaloData::xs, "X positions [Mpc/h]")
        .def_readwrite("ys", &HaloData::ys, "Y positions [Mpc/h]")
        .def_readwrite("zs", &HaloData::zs, "Z positions [Mpc/h]")
        .def_readwrite("ms", &HaloData::ms, "Masses [M_sun/h]")
        .def_readwrite("rs", &HaloData::rs, "Radii [Mpc/h]")
        .def("size", &HaloData::size, "Get number of halos");
    
    // Rockstar I/O functions
    m.def("read_rockstar", [](const std::string& file, RadiusType r_type,
                              const io::CosmologyHeader& cosmo) {
        std::vector<int> rids;
        std::vector<double> xs, ys, zs, ms, rs;
        read_rockstar(file, r_type, cosmo, rids, xs, ys, zs, ms, rs);
        HaloData data;
        data.rids = std::move(rids);
        data.xs = std::move(xs);
        data.ys = std::move(ys);
        data.zs = std::move(zs);
        data.ms = std::move(ms);
        data.rs = std::move(rs);
        return data;
    }, py::arg("file"), py::arg("radius_type"), py::arg("cosmo"),
    R"pbdoc(
        Read a Rockstar halo catalog.
        
        Results are sorted from largest to smallest radius.
        
        Parameters
        ----------
        file : str
            Path to Rockstar ASCII catalog (out_*.list)
        radius_type : RadiusType
            Which radius definition to use
        cosmo : io.CosmologyHeader
            Cosmology for mass-radius conversion
            
        Returns
        -------
        HaloData
            Container with IDs, positions, masses, and radii
    )pbdoc");
    
    m.def("read_rockstar_vals", [](const std::string& file,
                                   const io::CosmologyHeader& cosmo,
                                   const std::vector<Val>& val_flags) {
        std::vector<int> ids;
        std::vector<std::vector<double>> vals;
        read_rockstar_vals(file, cosmo, val_flags, ids, vals);
        return std::make_tuple(ids, vals);
    }, py::arg("file"), py::arg("cosmo"), py::arg("val_flags"),
    R"pbdoc(
        Read specific columns from a Rockstar ASCII catalog.
        
        Parameters
        ----------
        file : str
            Path to Rockstar ASCII catalog
        cosmo : io.CosmologyHeader
            Cosmology for derived value conversions
        val_flags : list of Val
            Columns to read (e.g., [Val.X, Val.Y, Val.Z, Val.MVir])
            
        Returns
        -------
        tuple(list[int], list[list[float]])
            (halo_ids, column_values)
    )pbdoc");
    
    m.def("read_binary_rockstar_vals", [](const std::string& file,
                                          const io::CosmologyHeader& cosmo,
                                          const std::vector<Val>& val_flags) {
        std::vector<int> ids;
        std::vector<std::vector<double>> vals;
        read_binary_rockstar_vals(file, cosmo, val_flags, ids, vals);
        return std::make_tuple(ids, vals);
    }, py::arg("file"), py::arg("cosmo"), py::arg("val_flags"),
    "Read specific columns from a binary Rockstar catalog");
    
    m.def("rockstar_convert", &rockstar_convert,
          py::arg("in_file"), py::arg("out_file"),
          "Convert Rockstar ASCII catalog to binary format");
    
    m.def("rockstar_convert_top_n", &rockstar_convert_top_n,
          py::arg("in_file"), py::arg("out_file"), py::arg("n"),
          "Convert Rockstar ASCII catalog to binary, keeping top N by M200b");
    
    m.def("read_table", &read_table,
          py::arg("file"), py::arg("col_idxs"),
          "Read columns from an ASCII table file (skips # comments)");
    
    m.def("read_binary_table", &read_binary_table,
          py::arg("file"), py::arg("col_idxs"),
          "Read columns from a binary table file");
    
    m.doc() = R"pbdoc(
        Halo finding and analysis module.
        
        This module provides tools for:
        - Converting between halo masses and radii (R200c, R200m, R500c, RVirial)
        - Spatial hashing for efficient neighbor searches
        - Finding subhalo relationships based on sphere overlaps
        
        Example
        -------
        >>> from asymptotic_tetra import halo, io
        >>> 
        >>> # Set up cosmology
        >>> cosmo = io.CosmologyHeader()
        >>> cosmo.z = 0.0
        >>> cosmo.omega_m = 0.3
        >>> cosmo.omega_l = 0.7
        >>> cosmo.h100 = 0.7
        >>> 
        >>> # Convert masses to R200m radii
        >>> masses = [1e12, 1e13, 1e14]  # M_sun/h
        >>> radii = halo.mass_to_radius(halo.RadiusType.R200m, cosmo, masses)
        >>> 
        >>> # Find subhalos
        >>> grid = halo.HaloGrid(num_cells=100, width=500.0, data_len=len(xs))
        >>> grid.insert(xs, ys, zs)
        >>> finder = halo.SubhaloFinder(grid)
        >>> finder.find_subhalos(xs, ys, zs, radii, mult=1.0)
        >>> 
        >>> # Query relationships
        >>> for i in range(len(xs)):
        ...     subs = finder.subhalos(i)
        ...     if subs:
        ...         print(f"Halo {i} has {len(subs)} subhalos")
    )pbdoc";
}
