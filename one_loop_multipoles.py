import numpy as np
import fastpt as fpt
from scipy.integrate import quad

#? The function which will be integrated over when calculating multipole power spectra.
def multipole_integrand(mu: float, l: int, Pk_dd, Pk_dt, Pk_tt, element: int):
    if l == 0:
        legendre_polynomial = 1
    elif l == 2:
        legendre_polynomial = 1/2*(3*mu**2 - 1)
    elif l == 4:
        legendre_polynomial = 1/8*(35*mu**4 - 30*mu**2 + 3)
    else:
        print("Invalid l in multipole_integrand.")

    return (2*l + 1)/2 * legendre_polynomial * (Pk_dd[element] - 2*mu**2*Pk_dt[element] + mu**4*Pk_tt[element])

#? Function which will convert the power spectrum into the multipole power spectrum through numerical integration.
def convert_to_Pl(Pk_dd, Pk_dt, Pk_tt, args=(multipole_integrand)):
    #* List of one-loop multipole power spectra in order of l=0,2,4.
    oneloop_pl = []
    #*Calculate the one-loop multipole power spectrum.
    for l in [0,2,4]:
        Pl = []
        for element in range(len(Pk_dd)):
            Pl.append(quad(multipole_integrand, -1.0, 1.0, args=(l, Pk_dd, Pk_dt, Pk_tt, element))[0])
        oneloop_pl.append(Pl)

    return np.asarray(oneloop_pl)

#! The function which will be called from main. It takes in the Quijote data wavenumber and linear pk calculated by CAMB in the kaiser effect code.
def get_oneloop_multipoles(ftp_instance, linear_Pk, f, args=(convert_to_Pl)):
    
    #* Calculate the one-loop power spectrum.
    Pk_dd = np.asarray(linear_Pk) + ftp_instance.one_loop_dd(linear_Pk)[0]

    #* Calculate the P_dt power spectra.
    Pk_dt = -f*np.asarray(linear_Pk) + ftp_instance.one_loop_dt(linear_Pk)[0]

    #* Calculate the P_tt power spectra.
    Pk_tt = f**2*np.asarray(linear_Pk) + ftp_instance.one_loop_tt(linear_Pk)[0]

    #* Calculate and return the one-loop multipole power spectra as a list of the power spectra in the order l=0,2,4.
    return convert_to_Pl(Pk_dd, Pk_dt, Pk_tt)

    