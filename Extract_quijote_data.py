import numpy as np

#? Get the data from the Quijote file and return it to the user in a convenient manner.
def extract_quijote_data(filepath: str):
    #* Read in and unpack data.
    data_array = np.loadtxt(filepath + ".txt", dtype = np.float64)
    k = data_array[:,0]
    P_0_Quijote = data_array[:,1]
    P_2_Quijote = data_array[:,2]
    P_4_Quijote = data_array[:,3]

    #* Package multipole power specta into a tuple and return it alongside k.
    return k, (P_0_Quijote, P_2_Quijote, P_4_Quijote)