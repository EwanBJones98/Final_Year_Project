import numpy as np 
from scipy.integrate import quad
from tqdm import tqdm

#? Function that the correct Legendre polynomial given l.
def legendre_polynomial(mu, l):
    if l == 0:
        return 1
    elif l == 2:
        return 1/2*(3*mu**2 - 1)
    elif l == 4:
        return 1/8*(35*mu**4 - 30*mu**2 + 3)
    else:
        print("Invalid l in multipole_integrand. None returned.")
        return None

#? Function which will multiply four legendre polynomials given four l values. This is the integrand.
def integrand(mu, l, lP, lPP, lPPP):
    return legendre_polynomial(mu, l)*legendre_polynomial(mu, lP)*legendre_polynomial(mu,lPP)*legendre_polynomial(mu,lPPP)

#? Function that integrates four legendre polynomails given four different values of l.
def integrate_polynomials(l, lP, lPP, lPPP):
    return quad(integrand, -1.0, 1.0, args=(l, lP, lPP, lPPP))[0]

#? Create a four index tensor and calculate the above integrate for every combination of l, l', l'' and l'''.
legendre_tensor = np.zeros((3,3,3,3))
for l in [0,2,4]:
    for lP in [0,2,4]:
        for lPP in [0,2,4]:
            for lPPP in [0,2,4]:
                element = integrate_polynomials(l, lP, lPP, lPPP)
                if np.abs(element) <= 0.0000000001:
                    print("fixed")
                    element = 0
                legendre_tensor[int(l/2),int(lP/2),int(lPP/2),int(lPPP/2)] = element

#? Save this tensor as a text file so that it does not have to be calculated each time.
np.save('Legendre_tensor', legendre_tensor)

l