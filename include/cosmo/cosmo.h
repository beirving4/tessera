#pragma once

#include <cmath>

namespace asymptotic_tetra {
namespace cosmo {

// Physical constants in MKS units
constexpr double G_MKS = 6.67259e-11;      // Gravitational constant [m^3 kg^-1 s^-2]
constexpr double MPC_MKS = 3.08560e+22;    // Megaparsec in meters
constexpr double MSUN_MKS = 1.98900e+30;   // Solar mass in kg
constexpr double C_MKS = 2.99792e+08;      // Speed of light [m/s]

/**
 * Calculate h(z) = H(z)/H0, the Hubble parameter ratio.
 * 
 * From Hubble's Law:
 * H(z)^2 + k(c/a)^2 = H0^2 h100^2 (OmegaR a^-4 + OmegaM a^-3 + OmegaL)
 * 
 * Assumes k = 0 (flat universe) and OmegaR = 0 (no radiation).
 * 
 * @param omega_m Matter density parameter
 * @param omega_l Dark energy density parameter
 * @param z Redshift
 * @return H(z)/H0
 */
inline double hubble_frac(double omega_m, double omega_l, double z) {
    return std::sqrt(omega_m * std::pow(1.0 + z, 3.0) + omega_l);
}

/**
 * Internal: Calculate critical density in MKS units (with h factors).
 */
inline double rho_critical_mks(double H0, double omega_m, double omega_l, double z) {
    double H0_mks = (H0 * 1000.0) / MPC_MKS;
    double H100 = H0 / 100.0;
    double H0_mks_h = H0_mks / H100;

    double H = hubble_frac(omega_m, omega_l, z) * H0_mks_h;
    return 3.0 * H * H / (8.0 * M_PI * G_MKS);
}

/**
 * Calculate the critical density of the universe.
 * 
 * The critical density appears in halo definitions and in the definitions
 * of the Omega parameters (Omega_X = rho_X / rho_critical).
 * 
 * @param H0 Hubble constant in km/s/Mpc (e.g., 70)
 * @param omega_m Matter density parameter
 * @param omega_l Dark energy density parameter  
 * @param z Redshift
 * @return Critical density in cosmological units [M_sun/Mpc^3 * h^2]
 */
inline double rho_critical(double H0, double omega_m, double omega_l, double z) {
    return rho_critical_mks(H0, omega_m, omega_l, z) * std::pow(MPC_MKS, 3) / MSUN_MKS;
}

/**
 * Calculate the average matter density of the universe.
 * 
 * @param H0 Hubble constant in km/s/Mpc
 * @param omega_m Matter density parameter
 * @param omega_l Dark energy density parameter
 * @param z Redshift
 * @return Average matter density in cosmological units [M_sun/Mpc^3 * h^2]
 */
inline double rho_average(double H0, double omega_m, double omega_l, double z) {
    return rho_critical(H0, omega_m, omega_l, 0.0) * omega_m * std::pow(1.0 + z, 3.0);
}

} // namespace cosmo
} // namespace asymptotic_tetra
