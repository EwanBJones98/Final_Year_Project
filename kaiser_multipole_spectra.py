import camb
import numpy as np
from scipy.integrate import solve_ivp
from Extract_quijote_data import extract_quijote_data

#! The function which will be called in the main file. It returns the kaiser multipole power spectra components.
def get_kaiser_multipole_spectra(Pk_linear, z):

    #? Functions which decompose the power spectra into multipoles.
    def P_0(Pk,f):
        return Pk*(1.0 + (2/3)*f + (1/5)*f**2.0)

    def P_2(Pk,f):
        return Pk*(4/21)*f*(7.0 + 3.0*f)

    def P_4(Pk,f):
        return Pk*(8/35)*f**2.0

    multipoleAnalyticList = [P_0, P_2, P_4] # Placing these functions into a list so that they can easily be iterated through.

    #? Calculate the linear growth function using the code that I wrote before.
    #* These are values that David gave me for to be used when calculating growth factor when we did it before.
    rho_m0 = 1.1523E-11
    Lambda4 = 2.52354E-11
    Dstar = 1
    Zstar = 50 
    Estar = -1.0/(1.0 + Zstar)
    initialConditions = [Dstar, Estar]
    #* Function that is passed into the ODE solver to obtain the linear growth function.
    def odeFunc(z, equationParameters):
        #* These are the parameters present in the equations.
        D = equationParameters[0]
        E = equationParameters[1] 

        #* Calculating all of the variables that I need to do the calculation.
        rho_m = (1.0+z)*(1.0+z)*(1.0+z)*rho_m0 #Equation for rho_m given z and rho_m0
        Omega_m = rho_m / (rho_m + Lambda4) #Equation for Omega_m given rho_m and Lambda4
        epsilon = 3.0*Omega_m/2.0 #Equation for epsilon given Omega_m

        #* The two differential equations.
        Dprime = E
        Eprime = (1.0-epsilon)*E/(1.0+z) + (3.0/2.0)*Omega_m*D/((1.0+z)*(1.0+z))

        return np.asarray([Dprime, Eprime])

    #* Calculate f.
    ode_soln = solve_ivp(odeFunc, (Zstar, 0), [Dstar, Estar], method='RK45', t_eval=[z])
    zSol = ode_soln['t']
    DSol = ode_soln['y'][0]
    ESol = ode_soln['y'][1]
    f=-(1.0+z)*ESol/DSol

    #? Calculate the multipole power spectra for the Kaiser component.
    kaiser_Pl = []
    for n, l in enumerate([0, 2, 4]):
        kaiser_Pl.append(multipoleAnalyticList[n](Pk_linear, f))

    return f, kaiser_Pl #! Returns the linear: growth factor, k & power spectra, and the list of multipole power spectra for the kaiser component of the power spectra as [l=0,l=2,l=4]