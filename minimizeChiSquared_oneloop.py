import numpy as np 
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import UnivariateSpline
from one_loop_multipoles_withDamping import get_oneloop_multipoles_with_damping
from Inverse_covariance_matrix import get_inverse_covariance_matrix, get_l0l0_submatrix_inverse, get_l_and_l2_covarianceMatrix_inverse

#* Function which will minimize the chisquared. 
def minimize_chisquared_oneloop(Pl_data, kData, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt, minTrimIndex, maxTrimIndex, lorentzianDamping = False):
    #* Obtain the covariance matrix.
    #!invCov = get_inverse_covariance_matrix()
    invCov = get_l_and_l2_covarianceMatrix_inverse()
    #!invCov = get_l0l0_submatrix_inverse()

    minimizedChisquared = minimize(chisquared, 350, args=(Pl_data, kData, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt,  invCov, minTrimIndex, maxTrimIndex, lorentzianDamping))
    return minimizedChisquared

#* Function for the chisquared.
def chisquared(sigma_v, Pl_data, kData, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt, invCov, minTrimIndex, maxTrimIndex, lorentzianDamping):
    #* Calculate the residual vector.
    delta = deltaVector(sigma_v, Pl_data, kData, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt, minTrimIndex, maxTrimIndex, lorentzianDamping)

    #* Calculate the chisquared.
    chisq = np.matmul(np.matmul(delta.copy().T, invCov), delta)

    #print(sigma_v, chisq)

    return chisq

#* Function for the residual at a given index.
def deltaVector(sigma_v, Pl_data, kData, kLinear, H0, f, Pk_dd, Pk_dt, Pk_tt, minTrimIndex, maxTrimIndex, lorentzianDamping):
    #* Get the TNS power spectrum.
    OneloopPl_untrimmed = get_oneloop_multipoles_with_damping(sigma_v, H0, f, kLinear, Pk_dd, Pk_dt, Pk_tt, lorentzianDamping)

    #* Trim the TNS power spectrum and kLinear
    OneloopPl = (OneloopPl_untrimmed[0][minTrimIndex:maxTrimIndex], OneloopPl_untrimmed[1][minTrimIndex:maxTrimIndex], OneloopPl_untrimmed[2][minTrimIndex:maxTrimIndex])
    kLinear = kLinear[minTrimIndex:maxTrimIndex]

    #? Spine the TNS power spectrum so that it can be evaluated at the same k as the data.
    #* These need to be concatenated so that the chisquared can be easily calculated.
    #!PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1], Pl_data[2]))
    PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1]))
    #!PlDataConcatenated = Pl_data[0]

    #* Calculate splines of the model data so that we can evaluate the model at the k data values.
    l0Spline = UnivariateSpline(kLinear, OneloopPl[0])
    l2Spline = UnivariateSpline(kLinear, OneloopPl[1])
    #!l4Spline = UnivariateSpline(kLinear, TNSPl[2])

    #* Create arrays of the model power spectra at same k as the data using the splines. Concatenate all of the results so that the chisquared can be easily calculated.
    l0_oneloop_Data = l0Spline(kData)
    l2_oneloop_Data = l2Spline(kData)
    #!l4TNSData = l4Spline(kData)
    #!PlTNSConcatenated = np.concatenate((l0TNSData, l2TNSData, l4TNSData))
    Pl_oneloop_Concatenated = np.concatenate((l0_oneloop_Data, l2_oneloop_Data))
    #!PlTNSConcatenated = l0TNSData

    return PlDataConcatenated - Pl_oneloop_Concatenated
