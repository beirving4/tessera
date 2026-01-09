#!/usr/bin/env python3
"""
Example: Reading GADGET-4 HDF5 files

This script demonstrates how to read GADGET-4 snapshot and halo catalog files
using the asymptotic_tetra library.
"""

import asymptotic_tetra as at

# Check if HDF5 support is available
print("HDF5 support:", at.io.HAS_HDF5)

if not at.io.HAS_HDF5:
    print("HDF5 support not available. Rebuild with -DBUILD_WITH_HDF5=ON")
    exit(1)

# =============================================================================
# Reading a GADGET-4 Snapshot
# =============================================================================

# Path to your snapshot file
snapshot_file = "path/to/snapshot_034.hdf5"

# Read just the header (fast)
header = at.io.read_gadget4_header(snapshot_file)
print(f"\n=== Snapshot Header ===")
print(f"Box size: {header.box_size}")
print(f"Redshift: {header.redshift}")
print(f"Time (scale factor): {header.time}")
print(f"Total particles: {header.total_particles():,}")
print(f"DM particles (type 1): {header.particles_of_type(1):,}")

# Read only positions (efficient for large files)
positions = at.io.read_gadget4_positions(snapshot_file, particle_type=1)
print(f"\nLoaded {len(positions):,} DM positions")
print(f"First 3 positions: {positions[:3]}")

# Read full snapshot with velocities and IDs
# particle_types is a bitmask: 0x02 = type 1 (DM only)
snap = at.io.read_gadget4_snapshot(
    snapshot_file,
    particle_types=0x02,  # Only DM (type 1)
    read_velocities=True,
    read_ids=True
)

dm = snap.dark_matter()
print(f"\n=== Dark Matter Particles ===")
print(f"Count: {len(dm):,}")
print(f"Has coordinates: {dm.has_coordinates()}")
print(f"Has velocities: {dm.has_velocities()}")
print(f"First 3 coordinates: {dm.coordinates[:3]}")
print(f"First 3 velocities: {dm.velocities[:3]}")
print(f"First 5 IDs: {dm.particle_ids[:5]}")

# =============================================================================
# Reading a GADGET-4 Halo Catalog (FOF/Subfind)
# =============================================================================

# Path to your halo catalog file
halo_file = "path/to/fof_subhalo_tab_034.hdf5"

# Read the catalog header
halo_header = at.io.read_gadget4_halo_header(halo_file)
print(f"\n=== Halo Catalog Header ===")
print(f"Number of groups: {halo_header.n_groups_total:,}")
print(f"Number of subhalos: {halo_header.n_subhalos_total:,}")

# Read the full catalog
catalog = at.io.read_gadget4_halo_catalog(halo_file, read_ids=False)
print(f"\n=== Halo Catalog ===")
print(f"Groups: {catalog.num_groups():,}")
print(f"Subhalos: {catalog.num_subhalos():,}")

# Access group (halo) properties
print(f"\n=== First 3 FOF Groups ===")
for i, g in enumerate(catalog.groups[:3]):
    print(f"Group {i}:")
    print(f"  Position: {g.pos}")
    print(f"  Velocity: {g.vel}")
    print(f"  Mass (FOF): {g.mass:.2e}")
    print(f"  M200c: {g.m_crit200:.2e}")
    print(f"  R200c: {g.r_crit200:.3f}")
    print(f"  N_particles: {g.len}")
    print(f"  N_subhalos: {g.n_subs}")

# Access subhalo properties
print(f"\n=== First 3 Subhalos ===")
for i, s in enumerate(catalog.subhalos[:3]):
    print(f"Subhalo {i}:")
    print(f"  Position: {s.pos}")
    print(f"  Mass: {s.mass:.2e}")
    print(f"  Vmax: {s.vmax:.1f}")
    print(f"  Half-mass radius: {s.halfmass_rad:.3f}")
    print(f"  Parent group: {s.group_nr}")

# =============================================================================
# Reading only groups or subhalos (more memory efficient)
# =============================================================================

# Read only FOF groups
groups = at.io.read_gadget4_groups(halo_file)
print(f"\nLoaded {len(groups)} groups")

# Read only subhalos
subhalos = at.io.read_gadget4_subhalos(halo_file)
print(f"Loaded {len(subhalos)} subhalos")
