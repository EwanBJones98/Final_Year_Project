import numpy as np 
import time
from scipy.integrate import quad
from tqdm import tqdm

#* Average the power spectrum of 1000 realisations k-by-k
power_spectra_realisations = {}
k_realisations = {}
averagedPl0 = []
averagedPl2 = []
averagedPl4 = []
averagedK = []
#* Read in all the power spectra and save them in a dictionary.
for realisation in range(1000,2000):
    data = np.loadtxt("/System/Volumes/Data/Volumes/Ewan_External/Quijote_data/fiducial/" + str(realisation) + "/Pk_m_RS0_z=0.5.txt")
    power_spectra_realisations[str(realisation)] = (data[:,1], data[:,2], data[:,3])
    k_realisations[str(realisation)] = data[:,0]
#* Average all of the power spectra.
for kIndex in tqdm(range(len(power_spectra_realisations['1000'][0]))):
    Pl0ToBeAveraged = []
    Pl2ToBeAveraged = []
    Pl4ToBeAveraged = []
    kToBeAveraged = []
    for realisation in range(1000,2000):
        Pl0ToBeAveraged.append(power_spectra_realisations[str(realisation)][0][kIndex])
        Pl2ToBeAveraged.append(power_spectra_realisations[str(realisation)][1][kIndex])
        Pl4ToBeAveraged.append(power_spectra_realisations[str(realisation)][2][kIndex])
        kToBeAveraged.append(k_realisations[str(realisation)][kIndex])
    averagedPl0.append(np.mean(Pl0ToBeAveraged))
    averagedPl2.append(np.mean(Pl2ToBeAveraged))
    averagedPl4.append(np.mean(Pl4ToBeAveraged))
    averagedK.append(np.mean(kToBeAveraged))

#? Cut off the spectra at some maximum k-value as error bars become arbitrarily small at large values of k.
h = 0.6711
kMax = 0.4
maxIndex = (np.abs(np.asarray(averagedK) - kMax).argmin()) #Index of data point closest to kmax
#* fastpt needs an even number of elements so fix that here too.
if maxIndex%2 != 0:
    maxIndex -= 1
print('Max index:', maxIndex)
#* Clip all power spectra.
averagedK = averagedK[:maxIndex]
averagedPl0 = averagedPl0[:maxIndex]
averagedPl2 = averagedPl2[:maxIndex]
averageaveragedPl4 =averagedPl4[:maxIndex]

#* Save the index of max k to a text file so that it can be obtained by the other codes.
with open('maxIndex.txt', 'w') as f:
  f.write('%d' % maxIndex)

#* Length of power spectra.
spectraLength = len(averagedK)

#* Length of each axis of the covariance matrix.
axisLength = len(3*averagedK)

#* Average difference between k values.
deltaKToBeAveraged = []
for index in range(len(averagedK) - 1):
    deltaK = averagedK[index + 1] - averagedK[index]
    deltaKToBeAveraged.append(deltaK)
averageDeltaK = np.mean(deltaKToBeAveraged)

def get_l(kIndex, args=(spectraLength)):
    if kIndex < spectraLength:
        return 0
    elif kIndex < 2*spectraLength:
        return 2
    elif kIndex < 3*spectraLength:
        return 4
    else:
        print("get_l index out of bounds.")

def get_legendre(mu, l):
    if l == 0:
        return 1
    elif l == 2:
        return 1/2*(3*mu**2 - 1)
    elif l == 4:
        return 1/8*(35*mu**4 - 30*mu**2 + 3)
    else:
        print("Invalid l in multipole_integrand.")

#? Read in the matrix calculated in legendre_integrals.py. These elements are the integrals referenced by l, l', l'', l'''.
Legendre_integral_tensor = np.load('Legendre_tensor.npy')

#? Function which calculates each element of the covariance matrix given k and k' indices.
def matrix_element(kIndex, kPrimeIndex, concatAveragedK, concat_averaged_Pl, args = (averageDeltaK, Legendre_integral_tensor)):
    #* Get the value of l given the k index.
    l = get_l(kIndex)
    lPrime = get_l(kPrimeIndex)

    #* Calculate N_mode.
    V = 1e9
    N_mode = (4*np.pi*(concatAveragedK[kIndex]**2)*averageDeltaK*V)/((2*np.pi)**3)

    #* Because of the Kroenecker-delta, we only want to compute the integral when we know the element is non-zero. This if statement represents the Kroenecker-delta.
    if concatAveragedK[kIndex] == concatAveragedK[kPrimeIndex]:
        #* The prefactor to the sum.
        prefactor = (2*l + 1)*(2*lPrime + 1)/N_mode
        #* Computing the sum over the l'' and l''' power spectra with the integral tensor element.
        individual_terms = []
        for lPP in [0,2,4]:
            P_lPP = concat_averaged_Pl[int(lPP/2)][kIndex]
            for lPPP in [0,2,4]:
                P_lPPP = concat_averaged_Pl[int(lPPP/2)][kIndex]
                individual_terms.append(Legendre_integral_tensor[int(l/2),int(lPrime/2),int(lPP/2),int(lPPP/2)]*P_lPP*P_lPPP)
        summedIntegralTerm = np.sum(individual_terms)
        #* Return the prefactor multiplied by the summed integral term.
        return prefactor*summedIntegralTerm
    else:
        return 0

    
def get_covariance_matrix(args=(averagedPl0, averagedPl2, averagedPl4, averagedK, axisLength)):
    #* Create a blank covariance matrix.
    covarianceMatrix = np.zeros((axisLength, axisLength))

    #* Concatenated spectras for ease.
    concatAveragedK = np.concatenate((averagedK, averagedK, averagedK))
    concatAveragedPl0 = np.concatenate((averagedPl0, averagedPl0, averagedPl0))
    concatAveragedPl2 = np.concatenate((averagedPl2, averagedPl2, averagedPl2))
    concatAveragedPl4 = np.concatenate((averagedPl4, averagedPl4, averagedPl4))
    concat_averaged_Pl = (concatAveragedPl0, concatAveragedPl2, concatAveragedPl4)

    #* Populate the covariance matrix.
    for kIndex in tqdm(range(axisLength)):
        for kPrimeIndex in range(axisLength):
            element = matrix_element(kIndex, kPrimeIndex, concatAveragedK, concat_averaged_Pl)
            covarianceMatrix[kIndex, kPrimeIndex] = element

    return covarianceMatrix


#*Obtain the covariance matrix.
print("Calculating the covariance matrix.")
t1 = time.time()
covarianceMatrix = get_covariance_matrix()
t2 = time.time()
print("Time to calculate covariance matrix: {}".format(t2-t1))

#* Obtain l0l0 submatrix only.
l0l0_submatrix = covarianceMatrix[0:spectraLength, 0:spectraLength]

#* Obtain l0l0,l2l2,l0l2 submatrix.
l0_and_l2_covarianceMatrix = covarianceMatrix[0:2*spectraLength, 0:2*spectraLength]

diagonal = covarianceMatrix.diagonal()
for element in diagonal:
    if element < 0:
        print(element)
print(len(diagonal))

#* Computing and saving the 1sigma errors for the l0l0 submatrix.
l0l0_diagonal = l0l0_submatrix.diagonal()
oneSigmaErrors = np.sqrt(l0l0_diagonal)
np.savetxt("oneSigmaErrors.txt", oneSigmaErrors, delimiter='    ')

#* Save covariance matrix as a text file.
np.savetxt("Covariance_matrix.txt", covarianceMatrix, delimiter='   ')

#* Save l0l0 submatrix.
np.savetxt("l0l0_submatrix.txt", l0l0_submatrix, delimiter='    ')

#* Save l0 and l2 covariance matrix.
np.savetxt("l0_and_l2_covarianceMatrix.txt", l0_and_l2_covarianceMatrix, delimiter='    ')