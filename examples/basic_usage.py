#!/usr/bin/env python3
"""
Example: Basic usage of tessera

This script demonstrates the core functionality of the library including
vectors, tetrahedra, random number generation, and interpolation.
"""

import tessera as ts

# =============================================================================
# Vectors (Vec3f and Vec3d)
# =============================================================================

print("=== Vectors ===")

# Create vectors
v1 = at.Vec3f(1.0, 2.0, 3.0)
v2 = at.Vec3f(4.0, 5.0, 6.0)

print(f"v1 = {v1}")
print(f"v2 = {v2}")

# Access components
print(f"v1.x = {v1.x}, v1.y = {v1.y}, v1.z = {v1.z}")

# Vector operations
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 2 = {v1 * 2}")
print(f"v1.dot(v2) = {v1.dot(v2)}")
print(f"v1.cross(v2) = {v1.cross(v2)}")
print(f"v1.norm() = {v1.norm()}")

# Periodic boundary subtraction (for cosmological simulations)
box_width = 100.0
v3 = at.Vec3f(95.0, 5.0, 50.0)
v4 = at.Vec3f(5.0, 95.0, 50.0)
diff = v3.sub(v4, box_width)  # Handles periodic wrapping
print(f"\nPeriodic subtraction (box={box_width}):")
print(f"  v3 = {v3}")
print(f"  v4 = {v4}")
print(f"  v3.sub(v4, box_width) = {diff}")

# =============================================================================
# Random Number Generation
# =============================================================================

print("\n=== Random Number Generation ===")

# Create generators with different algorithms
gen_taus = at.Generator.new_time_seed(at.GeneratorType.Tausworthe)
gen_xor = at.Generator.new_time_seed(at.GeneratorType.Xorshift)
gen_mt = at.Generator.new_time_seed(at.GeneratorType.Mersenne)

# Generate uniform random numbers
print(f"Tausworthe uniform(0,1): {gen_taus.uniform(0, 1)}")
print(f"Xorshift uniform(0,1): {gen_xor.uniform(0, 1)}")
print(f"Mersenne uniform(0,1): {gen_mt.uniform(0, 1)}")

# Generate arrays of random numbers
arr = gen_taus.uniform_array(0, 1, 5)
print(f"Array of 5 uniform randoms: {arr}")

# =============================================================================
# Tetrahedra
# =============================================================================

print("\n=== Tetrahedra ===")

# Create corner points
c0 = at.Vec3f(0, 0, 0)
c1 = at.Vec3f(1, 0, 0)
c2 = at.Vec3f(0, 1, 0)
c3 = at.Vec3f(0, 0, 1)

# Create tetrahedron
tet = at.Tetra(c0, c1, c2, c3)

print(f"Tetrahedron corners: {tet.corners}")
print(f"Volume: {tet.volume()}")
print(f"Barycenter: {tet.barycenter()}")

# Check if point is inside
test_point = at.Vec3f(0.1, 0.1, 0.1)
print(f"Point {test_point} inside? {tet.contains(test_point)}")

test_point2 = at.Vec3f(0.5, 0.5, 0.5)
print(f"Point {test_point2} inside? {tet.contains(test_point2)}")

# =============================================================================
# Density Quantities
# =============================================================================

print("\n=== Density Quantities ===")

# Available quantities for density field computation
quantities = [
    at.Quantity.Density,
    at.Quantity.DensityGradient,
    at.Quantity.Velocity,
    at.Quantity.VelocityDivergence,
    at.Quantity.VelocityCurl
]

for q in quantities:
    requires_vel = at.requires_velocity(q)
    can_proj = at.can_project(q)
    print(f"{q.name}: requires_velocity={requires_vel}, can_project={can_proj}")

# =============================================================================
# Grid and Cell Bounds
# =============================================================================

print("\n=== Grid and Cell Bounds ===")

# Create cell bounds (for domain decomposition)
origin = [0, 0, 0]
width = [10, 10, 10]
cb = at.CellBounds(origin, width)

print(f"CellBounds: origin={cb.origin}, width={cb.width}")

# Check intersection
cb2 = at.CellBounds([5, 5, 5], [10, 10, 10])
cells = 100
intersects = cb.intersect(cb2, cells)
print(f"Intersects with cb2? {intersects}")

# =============================================================================
# Cosmology Functions
# =============================================================================

print("\n=== Cosmology ===")

# Physical constants
print(f"G (MKS): {at.cosmo.G_MKS}")
print(f"Mpc (MKS): {at.cosmo.MPC_MKS}")
print(f"Msun (MKS): {at.cosmo.MSUN_MKS}")
print(f"c (MKS): {at.cosmo.C_MKS}")

# Density calculations
H0 = 70.0  # km/s/Mpc
omega_m = 0.3
omega_l = 0.7
z = 0.0

h_frac = at.cosmo.hubble_frac(H0, omega_m, omega_l, z)
rho_crit = at.cosmo.rho_critical(H0, omega_m, omega_l, z)
rho_avg = at.cosmo.rho_average(H0, omega_m, omega_l, z)

print(f"\nAt z={z} with H0={H0}, Omega_m={omega_m}, Omega_L={omega_l}:")
print(f"  H(z)/H0 = {h_frac}")
print(f"  rho_critical = {rho_crit:.3e} Msun/Mpc^3")
print(f"  rho_average = {rho_avg:.3e} Msun/Mpc^3")

print("\n=== Basic usage complete! ===")
