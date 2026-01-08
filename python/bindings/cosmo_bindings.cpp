#include <pybind11/pybind11.h>
#include "cosmo/cosmo.h"

namespace py = pybind11;
using namespace asymptotic_tetra::cosmo;

void bind_cosmo(py::module& m) {
    // Physical constants
    m.attr("G_MKS") = G_MKS;
    m.attr("MPC_MKS") = MPC_MKS;
    m.attr("MSUN_MKS") = MSUN_MKS;
    m.attr("C_MKS") = C_MKS;
    
    // Functions
    m.def("hubble_frac", &hubble_frac,
          py::arg("omega_m"), py::arg("omega_l"), py::arg("z"),
          R"pbdoc(
              Calculate h(z) = H(z)/H0, the Hubble parameter ratio.
              
              Assumes flat universe (k=0) and no radiation (OmegaR=0).
              
              Parameters
              ----------
              omega_m : float
                  Matter density parameter
              omega_l : float
                  Dark energy density parameter
              z : float
                  Redshift
                  
              Returns
              -------
              float
                  H(z)/H0
          )pbdoc");
    
    m.def("rho_critical", &rho_critical,
          py::arg("H0"), py::arg("omega_m"), py::arg("omega_l"), py::arg("z"),
          R"pbdoc(
              Calculate the critical density of the universe.
              
              Parameters
              ----------
              H0 : float
                  Hubble constant in km/s/Mpc (e.g., 70)
              omega_m : float
                  Matter density parameter
              omega_l : float
                  Dark energy density parameter
              z : float
                  Redshift
                  
              Returns
              -------
              float
                  Critical density in M_sun/Mpc^3 * h^2
          )pbdoc");
    
    m.def("rho_average", &rho_average,
          py::arg("H0"), py::arg("omega_m"), py::arg("omega_l"), py::arg("z"),
          R"pbdoc(
              Calculate the average matter density of the universe.
              
              Parameters
              ----------
              H0 : float
                  Hubble constant in km/s/Mpc
              omega_m : float
                  Matter density parameter
              omega_l : float
                  Dark energy density parameter
              z : float
                  Redshift
                  
              Returns
              -------
              float
                  Average matter density in M_sun/Mpc^3 * h^2
          )pbdoc");
    
    m.doc() = R"pbdoc(
        Cosmology module with physical constants and density calculations.
        
        Constants
        ---------
        G_MKS : float
            Gravitational constant in MKS units [m^3 kg^-1 s^-2]
        MPC_MKS : float
            Megaparsec in meters
        MSUN_MKS : float
            Solar mass in kg
        C_MKS : float
            Speed of light in m/s
            
        Example
        -------
        >>> from asymptotic_tetra import cosmo
        >>> # Calculate critical density at z=0 for Planck cosmology
        >>> rho_c = cosmo.rho_critical(H0=67.4, omega_m=0.315, omega_l=0.685, z=0)
        >>> print(f"Critical density: {rho_c:.2e} M_sun/Mpc^3 h^2")
    )pbdoc";
}
