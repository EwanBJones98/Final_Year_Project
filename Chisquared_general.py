import numpy as np 
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2
from Inverse_covariance_matrix import get_inverse_covariance_matrix_nochecks, get_l0l0_submatrix_inverse, get_l_and_l2_covarianceMatrix_inverse

#? Main will call this function. The Pl should be given as (lo, l2, l4).
def chi_squared_general(Pl_model, k_model, Pl_data, k_data, printDelta=False):
    #* These need to be concatenated so that the chisquared can be easily calculated.
    #!PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1], Pl_data[2]))
    PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1]))
    #!PlDataConcatenated = Pl_data[0]

    #* Calculate splines of the model data so that we can evaluate the model at the k data values.
    #print(len(k_model), len(Pl_model[0]))
    l0Spline = UnivariateSpline(k_model, Pl_model[0])
    l2Spline = UnivariateSpline(k_model, Pl_model[1])
    #!l4Spline = UnivariateSpline(k_model, Pl_model[2])

    #* Create arrays of the model power spectra at same k as the data using the splines. Concatenate all of the results so that the chisquared can be easily calculated.
    l0ModelData = l0Spline(k_data)
    l2ModelData = l2Spline(k_data)
    #!l4ModelData = l4Spline(k_data)
    #!PlModelConcatenated = np.concatenate((l0ModelData, l2ModelData, l4ModelData))
    PlModelConcatenated = np.concatenate((l0ModelData, l2ModelData))
    #!PlModelConcatenated = l0ModelData

    #* The inverse covariance matrix needed in the calculation of the chisquared.
    #!invCov = get_inverse_covariance_matrix_nochecks()
    invCov = get_l_and_l2_covarianceMatrix_inverse()
    #!invCov = get_l0l0_submatrix_inverse()

    #* Get the array of deltas.
    delta = deltaVector(PlModelConcatenated, PlDataConcatenated)
   
    #* Compute the chi squared.
    chiSquared = np.matmul(np.matmul(delta.copy().T, invCov), delta)
    
    #* Return the chisquared.
    return chiSquared

#? Alternative function which calculates chi-square term-by-term in order to be able to analyse intermediate results. Returns a list of chi-square components by k.
def chi_squared_general_components(Pl_model, k_model, Pl_data, k_data, printDelta=False):
    #* These need to be concatenated so that the chisquared can be easily calculated.
    #!PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1], Pl_data[2]))
    PlDataConcatenated = np.concatenate((Pl_data[0], Pl_data[1]))
    #!PlDataConcatenated = Pl_data[0]

    #* Calculate splines of the model data so that we can evaluate the model at the k data values.
    l0Spline = UnivariateSpline(k_model, Pl_model[0])
    l2Spline = UnivariateSpline(k_model, Pl_model[1])
    #!l4Spline = UnivariateSpline(k_model, Pl_model[2])

    #* Create arrays of the model power spectra at same k as the data using the splines. Concatenate all of the results so that the chisquared can be easily calculated.
    l0ModelData = l0Spline(k_data)
    l2ModelData = l2Spline(k_data)
    #!l4ModelData = l4Spline(k_data)
    #!PlModelConcatenated = np.concatenate((l0ModelData, l2ModelData, l4ModelData))
    PlModelConcatenated = np.concatenate((l0ModelData, l2ModelData))
    #!PlModelConcatenated = l0ModelData

    #* The inverse covariance matrix needed in the calculation of the chisquared.
    #!invCov = get_inverse_covariance_matrix_nochecks()
    #!invCov = get_l0l0_submatrix_inverse()

    #* Get the array of deltas.
    delta = deltaVector(PlModelConcatenated, PlDataConcatenated)
   
   
    #* Compute the chi squared.
    chiSquareComponents = []
    invCovComponents = invCov.diagonal()
    for i in range(len(k_data)):
        for j in range(len(k_data)):
            if i != j and invCov[i,j] != 0:
                print('not diagonal')
            if i == j:
                chiSquareComponents.append(delta[i]*invCov[i,j]*delta[j])
    
    #* Return the chisquared.
    return chiSquareComponents, invCovComponents

#? The function which calculates the delta component of the chi-squared.
def deltaVector(PlModelConcatenated, PlDataConcatenated):

    return PlDataConcatenated - PlModelConcatenated




