import numpy as np 
import matplotlib.pyplot as plt 
import fastpt as ftp 
import camb
import time
import seaborn as sns
from fastpt.myFASTPT import FASTPT
import scipy.stats as stats
from scipy.stats import chi2
from kaiser_multipole_spectra import get_kaiser_multipole_spectra
from Extract_quijote_data import extract_quijote_data
from one_loop_multipoles import get_oneloop_multipoles
#from TNS_multipoles import get_TNS_multipoles
#from sigma_v_constraint import constrain_velocity_dispersion
from TNSPl import TNSModel
from minimizeChiSquared import minimize_chisquared, chisquared
from Chisquared_general import chi_squared_general, chi_squared_general_components
from check_covariance_with_TNS_paper import compare_covariances
from Inverse_covariance_matrix import get_l0l0_submatrix_inverse, get_inverse_covariance_matrix_nochecks
from minimizeChiSquared_oneloop import minimize_chisquared_oneloop
from one_loop_multipoles_withDamping import get_oneloop_multipoles_with_damping

#? For each realisation of the Quijote simulations we will calculate the chisquare for: TNS model (gaussian), TNS model(lorentzian), Kaiser model, oneloop model, Oneloop+damping (Gaussian) model, and oneloop+damping (Lorentzian) model. We will also calculate the power spectra for each of these.
#? All of this data will be stored in the following dictionary. The keys are the realisation (the integer) and stored is ((chisquare), (Pl)). Where chisquare = (kaiser, oneloop, oneloop+gaussian, oneloop+lorentzian, TNS gaussian, TNS lorentzian). Pl is a tuple of (Pl=0, Pl=2) tuples for each model in the same order as the chisquare.
realisationResults = {}
realisationRange = range(2000,2300)

overall_start_time = time.time()
for realisation in realisationRange:
    initial_time = time.time()
    #TODO Set the cosmological parameters.
    Omega_b = 0.049
    h = 0.6711 #! Make sure to change this value in covariance_matrix.py if you change it here.
    H0 = 100*h
    ns = 0.9624

    #TODO Set the filepath for the Quijote data that you want to use.
    Quijote_filepath = "/System/Volumes/Data/Volumes/Ewan_External/Quijote_data/fiducial/" + str(realisation) + "/Pk_m_RS0_z=0.5"
    #realisation = float(Quijote_filepath.split('(')[1].split(')')[0])

    #! 1) Obtain the Quijote data. 
    Quijote_k, Quijote_Pl = extract_quijote_data(Quijote_filepath) #wavenumber and tuple of l=0,2,4 power spectra
    #Quijote_k *= h #Change the units of k so that it agrees with camb and works properly inside the functions.

    #* Clip the data in accordance with the covariance matrix. This is because the error becomes arbitrarily small at large k.
    maxIndex = int(np.loadtxt('maxIndex.txt'))
    #print("Number of data points used:", maxIndex, '\n')
    Quijote_k = Quijote_k[:maxIndex]
    Quijote_Pl = (Quijote_Pl[0][:maxIndex], Quijote_Pl[1][:maxIndex], Quijote_Pl[2][:maxIndex])

    #? Obtain the redshift from the Quijote data filename.
    z = float(Quijote_filepath.split("=")[1])

    #? Setting the range of k and value of z for CAMB to use based off of the range used in the Quijote data. 
    minKValue = min(Quijote_k)*(1 - 0.1)   #Subtract 10% so that I can trim this off later to get rid of the boundary effect which makes the power spectrum diverge.
    maxKValue = max(Quijote_k)*(1 + 0.1) #Add 10% so that I can trim this off later to get rid of the boundary effect which makes the power spectrum diverge.
    redshiftValues = [z]

    #? Create a set of parameters and set up the cosmology.
    pars = camb.CAMBparams()
    pars.set_cosmology(ombh2=Omega_b*h**2, H0=H0) 
    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=redshiftValues, kmax=maxKValue)
    results = camb.get_results(pars)

    #! 2) Calcualte the linear power spectrum using CAMB
    kh_linear, zlist, Pk_linear_array = results.get_matter_power_spectrum(minkh=minKValue, maxkh=maxKValue, npoints=len(Quijote_k))

    #? The actual power spectrum must be extracted from Pk_linear.
    Pk_linear = Pk_linear_array[0]

    #? This has even spacing and so has to be used in the calculation of the one-loop power spectrum.
    #k_linear = h*kh_linear
    k_linear = kh_linear

    #* Get the indices in k_linear which we need to trim to in order to obtain the desired power spectrum.
    minTrimIndex = np.abs(k_linear - min(Quijote_k)).argmin()
    maxTrimIndex = np.abs(k_linear - max(Quijote_k)).argmin()

    #! 3) Obtain the linear growth factor and power spectrum to pass to the one loop code and the kaiser effect multipole power spectra.
    f, Kaiser_Pl_untrimmed = get_kaiser_multipole_spectra(Pk_linear, z) #linear: growth factor, k & power spectrum, and multipole in list l=0,2,4
    Kaiser_Pl = (Kaiser_Pl_untrimmed[0][minTrimIndex:maxTrimIndex], Kaiser_Pl_untrimmed[1][minTrimIndex:maxTrimIndex], Kaiser_Pl_untrimmed[2][minTrimIndex:maxTrimIndex]) #Trim the kaiser multipole power spectrum.

    #? Set the padding width for the fastpt calculations.
    padWidth = 96

    #? Create an instance of the fastPT class here so that it does not need to be done multiple times.
    ftp_inst = FASTPT(k_linear, to_do=["one_loop_dd","RSD", "one_loop_dt", "one_loop_tt"], n_pad = padWidth) #I pass k_linear as it needs to be evenly intervaled in log space, Quijote data is not.

    #! 4) Obtain the one-loop multipole power spectra.
    OneLoop_Pl_untrimmed = get_oneloop_multipoles(ftp_inst, Pk_linear, f)
    OneLoop_Pl = (OneLoop_Pl_untrimmed[0][minTrimIndex:maxTrimIndex], OneLoop_Pl_untrimmed[1][minTrimIndex:maxTrimIndex], OneLoop_Pl_untrimmed[2][minTrimIndex:maxTrimIndex]) #Trim the one-loop multipole power spectrum.

    #? Obtain sigmav-independent TNS power power spectra components here so that they do not have to be found multiple times. 
    Pk_dd = np.asarray(Pk_linear) + ftp_inst.one_loop_dd(Pk_linear)[0]
    Pk_dt = -f*np.asarray(Pk_linear) + ftp_inst.one_loop_dt(Pk_linear)[0]
    Pk_tt = f**2*np.asarray(Pk_linear) + ftp_inst.one_loop_tt(Pk_linear)[0]
    ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8 = ftp_inst.RSD_ABsum_components(Pk_linear, f, 1) #The 1 is just the bias.

    #! 5) Constrain the velocity dispersion so that we can use this value in the calculation of the TNS and oneloop+dampings multipole power spectrum. Do this for both Gaussian and Lorentzian damping functions.
    t1 = time.time()
    #print("Beginning to constrain the velocity dispersion for both Gaussian and Lorentzian damping functions, for both the TNS on oneloop+damping models.")
    sigma_v_lorentzian = minimize_chisquared(Quijote_Pl, Quijote_k, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8, minTrimIndex, maxTrimIndex, LorentzianDamping=True).x
    #print("Lorentzian TNS constraint complete")
    sigma_v_gaussian = minimize_chisquared(Quijote_Pl, Quijote_k, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8, minTrimIndex, maxTrimIndex).x
    #print("Gaussian  TNS constraint complete.")
    sigma_v_oneloop_lorentzian = minimize_chisquared_oneloop(Quijote_Pl, Quijote_k, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, minTrimIndex, maxTrimIndex, lorentzianDamping=True).x
    #print("Lorentzian one-loop constraint complete.")
    sigma_v_oneloop_gaussian = minimize_chisquared_oneloop(Quijote_Pl, Quijote_k, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, minTrimIndex, maxTrimIndex).x
    #print("Guassian one-loop constaint complete.")
    t2 = time.time()
    #print("Finished constraining velocity dispersions in {} seconds".format(t2-t1))
    #print("\n\n")
    #print(" The velocity dispersion for the TNS model with Gaussian damping functions is:", sigma_v_gaussian)
    #print("The velocity dispersion for the TNS model with Lorentzian damping functions is:", sigma_v_lorentzian)
    #print(" The velocity dispersion for the one-loop+damping model with Gaussian damping functions is:", sigma_v_oneloop_gaussian)
    #print("The velocity dispersion for the one-loop+damping model with Lorentzian damping functions is:", sigma_v_oneloop_lorentzian)

    #! 6) Obtain the multipole power spectra using the sigma_v constraints.
    #print("Beginning to obtain the multipole power spectrum for the TNS model, with both Gaussian and Lorentzian damping functions.")
    t1 = time.time()
    TNS_Pl_lorentzian = TNSModel(sigma_v_lorentzian, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8, LorentzianDamping=True)
    TNS_Pl_gaussian = TNSModel(sigma_v_gaussian, k_linear, H0, f, Pk_dd, Pk_dt, Pk_tt, ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8)
    TNS_Pl_gaussian = (TNS_Pl_gaussian[0][minTrimIndex:maxTrimIndex], TNS_Pl_gaussian[1][minTrimIndex:maxTrimIndex], TNS_Pl_gaussian[2][minTrimIndex:maxTrimIndex]) #Trim the gaussian TNS power spectrum.
    TNS_Pl_lorentzian = (TNS_Pl_lorentzian[0][minTrimIndex:maxTrimIndex], TNS_Pl_lorentzian[1][minTrimIndex:maxTrimIndex], TNS_Pl_lorentzian[2][minTrimIndex:maxTrimIndex]) #Trim the lorentzian TNS power spectrum.
    t2 = time.time()
    #print("Obtained multipole power spectrum for the TNS model in {} seconds".format(t2 - t1))
    #print("\n")
    #print("Beginning to obtain the multipole power spectrum for the one-loop+damping model, with both Gaussian and Lorentzian damping functions.")
    t1 = time.time()
    oneloop_Pl_lorentzian = get_oneloop_multipoles_with_damping(sigma_v_oneloop_lorentzian, H0, f, k_linear, Pk_dd, Pk_dt, Pk_tt, lorentzianDamping=True)
    oneloop_Pl_gaussian = get_oneloop_multipoles_with_damping(sigma_v_oneloop_gaussian, H0, f, k_linear, Pk_dd, Pk_dt, Pk_tt)
    oneloop_Pl_lorentzian = (oneloop_Pl_lorentzian[0][minTrimIndex:maxTrimIndex], oneloop_Pl_lorentzian[1][minTrimIndex:maxTrimIndex], oneloop_Pl_lorentzian[2][minTrimIndex:maxTrimIndex]) #Trim the gaussian oneloop+damping power spectrum.
    oneloop_Pl_gaussian = (oneloop_Pl_gaussian[0][minTrimIndex:maxTrimIndex], oneloop_Pl_gaussian[1][minTrimIndex:maxTrimIndex], oneloop_Pl_gaussian[2][minTrimIndex:maxTrimIndex]) #Trim the lorentzian oneloop+damping power spectrum.
    t2 = time.time()
    #print("Obtained multipole power spectrum for the one-loop+damping model in {} seconds".format(t2 - t1))

    #* Trim k_linear for plotting.
    k_linear = k_linear[minTrimIndex:maxTrimIndex]

    #! 7) Calculate the chi_squared for all of the models.
    #* Calculate chisquared for oneloop and Kaiser models.
    chisquared_TNS_gaussian = chi_squared_general(TNS_Pl_gaussian, k_linear, Quijote_Pl, Quijote_k)
    chisquared_TNS_lorentzian = chi_squared_general(TNS_Pl_lorentzian, k_linear, Quijote_Pl, Quijote_k)
    chisquared_oneloop_gaussian = chi_squared_general(oneloop_Pl_gaussian, k_linear, Quijote_Pl, Quijote_k)
    chisquared_oneloop_lorentzian = chi_squared_general(oneloop_Pl_lorentzian, k_linear, Quijote_Pl, Quijote_k)
    chisquared_oneLoop = chi_squared_general(OneLoop_Pl, k_linear, Quijote_Pl, Quijote_k)
    chisquared_kasier = chi_squared_general(Kaiser_Pl, k_linear, Quijote_Pl, Quijote_k)
    #* Print the results for all three models.
    #print("The chi-squared for the TNS model with a Gaussian damping function is:", chisquared_TNS_gaussian)
    #print("The chi-squared for the TNS model with a Lorentzian damping function is:", chisquared_TNS_lorentzian)
    #print("The chi-squared for the one-loop model with a Gaussian damping function is:", chisquared_oneloop_gaussian)
    #print("The chi-squared for the one-loop model with a Lorentzian damping function is:", chisquared_oneloop_lorentzian)
    #print("The chi-squared for the one-loop model is:", chisquared_oneLoop)
    #print("The chi-squared for the Kaiser model is:", chisquared_kasier)

    #! 8) Save the results of the realisation to the dictionary.
    chisquareTuple = (chisquared_kasier, chisquared_oneLoop, chisquared_oneloop_gaussian, chisquared_oneloop_lorentzian, chisquared_TNS_gaussian, chisquared_TNS_lorentzian)
    PlTuple = (Kaiser_Pl, OneLoop_Pl, oneloop_Pl_gaussian, oneloop_Pl_lorentzian, TNS_Pl_gaussian, TNS_Pl_lorentzian)
    realisationResults[str(realisation)] = (chisquareTuple, PlTuple)

    #* The time to run the realisation.
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print('\n Time to run the realisation was {} minutes and {} seconds'.format(int(elapsed_time/60), elapsed_time%60))


overall_end_time = time.time()
overall_elapsed_time = overall_end_time - overall_start_time
print("\n Time to run the entire code was {} minutes and {} seconds".format(int(overall_elapsed_time/60), overall_elapsed_time%60))

#! 9) Plot the distribution of the chisquares.
for modelIndex, model in enumerate(["Kaiser", "one-loop", "one-loop + Gaussian damping", "one-loop + Lorentzian damping", "TNS with Gaussian damping", "TNS with Lorentzian damping"]):
    #* Set up the figure.
    #plt.figure()
    #plt.title('Chisquare distribution over all realisations for the {} model'.format(model))
    #plt.xlabel('realisation')
    #plt.ylabel('chisquare')
    #* Place all of the chisquares into a list so that it can be plotted.
    chisquares = [realisationResults[str(realisation)][0][modelIndex] for realisation in realisationRange]
    #* Plot the chisquares against the realisation
    #plt.plot(realisationRange, chisquares, 'ro')
    #* Print a few statistical results for each model.
    meanChisq = np.mean(chisquares)
    stdevChisq = np.std(chisquares)
    medianChisq = np.median(chisquares)
    maxChisq = np.max(chisquares)
    minChisq = np.min(chisquares)
    print("\n\n")
    print("For the {} model".format(model))
    print("     Mean = ", meanChisq)
    print("     Standard Deviation =", stdevChisq)
    print("     Median =", medianChisq)
    print("     Maximum =", maxChisq)
    print("     Minimum =", minChisq)

    #? Plot a histogram for the chisquare for each model. We over plot the x^2 pdf with the number of degrees of freedom equal to the number of datapoints.
    pdf_xaxis = np.linspace(min(chisquares), max(chisquares))
    plt.figure()
    sns.set_style('darkgrid')
    sns.distplot(chisquares)
    #plt.hist(chisquares, bins='auto', color='#0504aa', alpha=0.7)
    #plt.plot(chi2.pdf(pdf_xaxis, df=maxIndex), color='r', lw=2)
    plt.title('Histogram of the chisquared for the {} model over {} realisations'.format(model, len(chisquares)))
    plt.xlabel('chisquare', kde=False)
    plt.grid()
    plt.plot()

plt.show()
 
"""
#* Reading in standard deviation for Quijote data. 
stdev_Pl0 = np.loadtxt('oneSigmaErrors.txt', delimiter='    ')
#? Plots for the multipole power spectra.
for n, l in enumerate([0,2]):
    plt.figure()
    plt.axes(xscale = 'log', yscale ='log')
    plt.title("Multipole power spectrum for realisation {}, l={} and z={}".format(realisation, l, z))
    plt.xlabel("k")
    plt.ylabel("P_l(K)")
    plt.plot(Quijote_k, np.abs(Quijote_Pl[n]), 'b', label="Quijote data")
    plt.fill_between(Quijote_k, Quijote_Pl[n] - stdev_Pl0, Quijote_Pl[n] + stdev_Pl0, facecolor='b', edgecolor='None', alpha=0.2)
    plt.plot(k_linear, np.abs(Kaiser_Pl[n]), 'k', label="P_kaiser")
    plt.plot(k_linear, np.abs(OneLoop_Pl[n]), 'r', label="One-loop")
    plt.plot(k_linear, np.abs(TNS_Pl_gaussian[n]), 'g', label="TNS (Gaussian)")
    plt.plot(k_linear, np.abs(TNS_Pl_lorentzian[n]), label="TNS (Lorentzian)") 
    plt.plot(k_linear, np.abs(oneloop_Pl_gaussian[n]), label="one-loop + damping (Gaussian)") 
    plt.plot(k_linear, np.abs(oneloop_Pl_lorentzian[n]), label="one-loop + damping (lorentzian)") 
    plt.legend()

plt.show()


#? Plot of the chi-square component-by-component as a function of k.
chiSquareComponents, invCovComponents = chi_squared_general_components(TNS_Pl_gaussian, k_linear, Quijote_Pl, Quijote_k)
plt.figure()
plt.plot(Quijote_k, chiSquareComponents, 'bx', label='Chi Squared Components')
plt.plot(Quijote_k, invCovComponents, 'kx', label='Inverse Covariance Matrix Elements')
plt.xlabel('k')
plt.title('Chi-square and InvCov distribution for l=0')
plt.legend()
plt.show()
"""
