#!/usr/bin/env python3
"""
Example: Computing density histograms

This script demonstrates how to use the histogram module to compute
mass-weighted density histograms from phase-space sheet data.
"""

import tessera as ts

# =============================================================================
# Histogram Configuration
# =============================================================================

# Create histogram info with logarithmic binning
info = at.render.HistInfo()
info.min = 0.1      # Minimum density
info.max = 100.0    # Maximum density
info.bins = 50      # Number of bins
info.scale = "log"  # "log" or "linear"

print(f"Histogram config: min={info.min}, max={info.max}, bins={info.bins}")
print(f"Using logarithmic scale: {info.is_log()}")

# =============================================================================
# Define Spatial Regions (HistBox)
# =============================================================================

# Create a histogram box for a specific spatial region
# Parameters: x, y, z, x_width, y_width, z_width, bins
box1 = at.render.HistBox.from_config(
    x=0, y=0, z=0,
    x_width=100, y_width=100, z_width=100,
    bins=50
)

print(f"\nHistBox 1:")
print(f"  Origin: {box1.origin}")
print(f"  Span: {box1.span}")
print(f"  Number of bins: {len(box1.counts)}")

# Create another box for a different region
box2 = at.render.HistBox.from_config(
    x=100, y=100, z=100,
    x_width=50, y_width=50, z_width=50,
    bins=50
)

# =============================================================================
# Grid I/O (for density interpolation)
# =============================================================================

# Read a density grid file header
grid_file = "path/to/density_grid.dat"

# Read grid header
# grid_header = at.render.read_grid_header(grid_file)
# print(f"\nGrid header:")
# print(f"  Box width: {grid_header.cosmo.box_width}")
# print(f"  Pixel width: {grid_header.loc.pixel_width}")

# Read the full grid
# grid_data = at.render.read_grid(grid_file)
# print(f"  Grid size: {len(grid_data)} values")

# =============================================================================
# Using HistManager (Full workflow)
# =============================================================================

# NOTE: The full HistManager workflow requires:
# 1. Sheet segment files (.gtet files from gotetra)
# 2. A density grid file for interpolation

# Example setup (uncomment when you have the necessary files):
#
# sheet_files = [
#     "path/to/sheet_000.gtet",
#     "path/to/sheet_001.gtet",
#     # ... more sheet files
# ]
#
# boxes = [box1, box2]
# points_per_tetra = 100  # Monte Carlo sample points
#
# manager = at.render.HistManager(
#     files=sheet_files,
#     boxes=boxes,
#     points=points_per_tetra,
#     quantity="Density",
#     grid_file=grid_file
# )
#
# # Optional: subsample for faster computation
# manager.subsample(2)  # Must be power of 2
#
# # Compute histograms
# manager.hist(info)
#
# # Access results
# for i, box in enumerate(manager.boxes()):
#     print(f"\nBox {i} results:")
#     print(f"  Bin centers: {box.centers[:5]}...")
#     print(f"  Counts: {box.counts[:5]}...")
#     total = sum(box.counts)
#     print(f"  Total counts: {total}")

print("\nHistogram module loaded successfully!")
print("See comments in script for full workflow example.")
