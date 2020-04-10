import numpy as np
import fastpt as fpt
from scipy.integrate import quad

#? The function which will be integrated over when calculating multipole power spectra.
def multipole_integrand(mu: float, sigma_v, k, H0, f, l: int, Pk_dd, Pk_dt, Pk_tt, element: int, lorentzianDamping):
    if l == 0:
        legendre_polynomial = 1
    elif l == 2:
        legendre_polynomial = 1/2*(3*mu**2 - 1)
    elif l == 4:
        legendre_polynomial = 1/8*(35*mu**4 - 30*mu**2 + 3)
    else:
        print("Invalid l in multipole_integrand.")

    #* Calculate the damping function.
    if lorentzianDamping:
        damping_function = 1/(1 + (mu*k*sigma_v*f/H0)**2)
    else:
        damping_function = np.exp(-1*(mu*k*sigma_v*f/H0)**2)

    return (2*l + 1)/2 * damping_function * legendre_polynomial * (Pk_dd[element] - 2*mu**2*Pk_dt[element] + mu**4*Pk_tt[element])

#? Function which will convert the power spectrum into the multipole power spectrum through numerical integration.
def convert_to_Pl(sigma_v, H0, f, kLinear, Pk_dd, Pk_dt, Pk_tt, lorentzianDamping):
    #* List of one-loop multipole power spectra in order of l=0,2,4.
    oneloop_pl = []
    #*Calculate the one-loop multipole power spectrum.
    for l in [0,2,4]:
        Pl = []
        for element in range(len(Pk_dd)):
            k = kLinear[element]
            Pl.append(quad(multipole_integrand, -1.0, 1.0, args=(sigma_v, k, H0, f, l, Pk_dd, Pk_dt, Pk_tt, element, lorentzianDamping))[0])
        oneloop_pl.append(Pl)

    return np.asarray(oneloop_pl)

#! The function which will be called from main. It takes in the Quijote data wavenumber and linear pk calculated by CAMB in the kaiser effect code.
def get_oneloop_multipoles_with_damping(sigma_v, H0, f, kLinear, Pk_dd, Pk_dt, Pk_tt, lorentzianDamping = False):
    
    #* Calculate and return the one-loop multipole power spectra as a list of the power spectra in the order l=0,2,4.
    return convert_to_Pl(sigma_v, H0, f, kLinear, Pk_dd, Pk_dt, Pk_tt, lorentzianDamping)

    