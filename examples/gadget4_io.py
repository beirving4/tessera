#!/usr/bin/env python3
"""
Example: Reading GADGET-4 HDF5 files

This script demonstrates how to read GADGET-4 snapshot and halo catalog files
using the tessera library. The unified interface functions
`read_gadget4_snapshot` and `read_gadget4_halo_catalog` automatically detect
whether you're reading a single file or a distributed multi-file output.
"""

import tessera as ts

# Check if HDF5 support is available
print("HDF5 support:", at.io.HAS_HDF5)

if not at.io.HAS_HDF5:
    print("HDF5 support not available. Rebuild with -DBUILD_WITH_HDF5=ON")
    exit(1)

# =============================================================================
# Reading a Single-File GADGET-4 Snapshot
# =============================================================================

# Path to your snapshot file (single file)
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

# Read full snapshot - unified interface auto-detects single file
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
# Reading a Distributed (Multi-File) GADGET-4 Snapshot
# =============================================================================

# For distributed snapshots, pass the directory path.
# The snapshot number is inferred from directory name "snapdir_NNN"
# or can be specified explicitly.
distributed_dir = "path/to/snapdir_009"

# Option 1: Auto-infer snapshot number from directory name
snap_dist = at.io.read_gadget4_snapshot(
    distributed_dir,
    particle_types=0x02,  # DM only
    read_velocities=False,
    read_ids=False
)
print(f"\n=== Distributed Snapshot ===")
print(f"Total DM particles: {len(snap_dist.dark_matter()):,}")

# Option 2: Explicitly provide snapshot number (useful for non-standard naming)
snap_dist2 = at.io.read_gadget4_snapshot(
    distributed_dir,
    snapshot_num=9,  # Explicitly set snapshot number
    particle_types=0x02
)

# =============================================================================
# Reading a Single-File GADGET-4 Halo Catalog (FOF/Subfind)
# =============================================================================

# Path to your halo catalog file
halo_file = "path/to/fof_subhalo_tab_034.hdf5"

# Read the catalog header
halo_header = at.io.read_gadget4_halo_header(halo_file)
print(f"\n=== Halo Catalog Header ===")
print(f"Number of groups: {halo_header.n_groups_total:,}")
print(f"Number of subhalos: {halo_header.n_subhalos_total:,}")

# Read the full catalog - unified interface auto-detects single file
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
# Reading a Distributed Halo Catalog
# =============================================================================

# For distributed halo catalogs, pass the directory path.
# Subhalos are automatically sorted by (group_nr, rank_in_gr)
# and first_sub pointers are updated correctly.
catalog_dist = at.io.read_gadget4_halo_catalog(
    distributed_dir,  # Same directory as snapshots
    read_ids=False
)
print(f"\n=== Distributed Halo Catalog ===")
print(f"Groups: {catalog_dist.num_groups():,}")
print(f"Subhalos: {catalog_dist.num_subhalos():,}")

# =============================================================================
# Reading only groups or subhalos (more memory efficient)
# =============================================================================

# Read only FOF groups from a single file
groups = at.io.read_gadget4_groups(halo_file)
print(f"\nLoaded {len(groups)} groups")

# Read only subhalos from a single file
subhalos = at.io.read_gadget4_subhalos(halo_file)
print(f"Loaded {len(subhalos)} subhalos")

# =============================================================================
# Low-Level Functions (explicit single-file or distributed)
# =============================================================================

# If you want to explicitly use single-file reading:
snap_single = at.io.read_gadget4_snapshot_file(snapshot_file, particle_types=0x02)

# If you want to explicitly use distributed reading:
snap_multi = at.io.read_gadget4_snapshot_distributed(
    distributed_dir,
    snapshot_num=9,
    particle_types=0x02
)

# Discover files in a directory (useful for custom processing)
snap_files = at.io.discover_snapshot_files(distributed_dir, snapshot_num=9)
halo_files = at.io.discover_halo_catalog_files(distributed_dir, snapshot_num=9)
print(f"\nFound {len(snap_files)} snapshot files")
print(f"Found {len(halo_files)} halo catalog files")
