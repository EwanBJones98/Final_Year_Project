import numpy as np 
from scipy.integrate import quad

#* Function which will return the value of the TNS multipole power spectrum for a given l at a value of k.
def TNSModel(sigma_v, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt, ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8, LorentzianDamping = False):

    #* Determine Pl for the TNS model.
    TNS_Pl = []
    for l in [0,2,4]:
        Pl = []
        for index in range(len(kLinear)):
            #? Things that we want to do once because they remain constant with mu and so should not appear in the integrand.
            k = kLinear[index]
            Pk_dd_ind = Pk_dd[index]
            Pk_dt_ind = Pk_dt[index]
            Pk_tt_ind = Pk_tt[index]
            ABsum_mu2_ind = ABsum_mu2[index]
            ABsum_mu4_ind = ABsum_mu4[index]
            ABsum_mu6_ind = ABsum_mu6[index]
            ABsum_mu8_ind = ABsum_mu8[index]

            Pl.append(quad(integrand, -1.0, 1.0, args=(index, sigma_v, l, k, H0, f, Pk_dd_ind, Pk_dt_ind, Pk_tt_ind, ABsum_mu2_ind, ABsum_mu4_ind, ABsum_mu6_ind, ABsum_mu8_ind, LorentzianDamping))[0])

        TNS_Pl.append(Pl)
    
    #* Return the power spectrum as (l0,l2,l4)
    return np.asarray(TNS_Pl)

def integrand(mu, index, sigma_v, l, k, H0, f, Pk_dd_ind, Pk_dt_ind, Pk_tt_ind, ABsum_mu2_ind, ABsum_mu4_ind, ABsum_mu6_ind, ABsum_mu8_ind, LorentzianDamping):
    #* Determine legendre polynomial.
    if l == 0:
        legendre_polynomial = 1
    elif l == 2:
        legendre_polynomial = 1/2*(3*mu**2 - 1)
    elif l == 4:
        legendre_polynomial = 1/8*(35*mu**4 - 30*mu**2 + 3)
    else:
        print("Invalid l in multipole_integrand.")

    #* Calculate the damping function.
    if LorentzianDamping:
        damping_function = 1/(1 + (mu*k*sigma_v*f/H0)**2)
    else:
        damping_function = np.exp(-1*(mu*k*sigma_v*f/H0)**2)

    #* Calculate TNS power spectrum,
    TNS_Pk = damping_function*(Pk_dd_ind - 2*mu**2*Pk_dt_ind + mu**4*Pk_tt_ind + ABsum_mu2_ind*mu**2 + ABsum_mu4_ind*mu**4 + ABsum_mu6_ind*mu**6 + ABsum_mu8_ind*mu**8)

    return (2*l + 1)/2 * legendre_polynomial * TNS_Pk