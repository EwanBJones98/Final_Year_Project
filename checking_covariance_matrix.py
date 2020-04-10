import numpy as np 

cMatrix = np.loadtxt("Covariance_matrix.txt", dtype=float,  delimiter='   ')

spectraLength = int(len(cMatrix[:,])/3)

#* Check if the matrix is symmetric, it should be in theory.
print("The covariance matrix is symmetric:", np.allclose(cMatrix, cMatrix.T))

inverseMatrix = np.linalg.inv(cMatrix)
cond = np.linalg.cond(inverseMatrix)
print("Condition Number:", cond, "Inverted: ", np.allclose(np.dot(inverseMatrix, cMatrix), np.eye(spectraLength*3)))

#* Check if the submatrices are symmetric and diagonal.
for l in [0,2,4]:
    for lPrime in [0,2,4]: 
        if l == 0:
            initL = 0 
        elif l == 2:
            initL = spectraLength
        elif l == 4:
            initL = 2*spectraLength
        finalL = initL + spectraLength

        if lPrime == 0:
            initLPrime = 0 
        elif lPrime == 2:
            initLPrime = spectraLength
        elif lPrime == 4:
            initLPrime = 2*spectraLength
        finalLPrime = initLPrime + spectraLength

        subMatrix = cMatrix[initL:finalL, initLPrime:finalLPrime]
        submatrixCopy = subMatrix.copy()
        removedDiagonal = subMatrix.copy() - np.diag(np.diagonal(submatrixCopy))
        
        if np.count_nonzero(removedDiagonal) == 0:
            isDiagonal = True
        else:
            isDiagonal = False

        determinant = np.linalg.det(subMatrix)



        #print("l = {}, lPrime = {}. Submatrix is symmetric:".format(l, lPrime), np.allclose(subMatrix, subMatrix.T), "Submatrix is diagonal:", isDiagonal)
        print("l = {}, lPrime = {}".format(l, lPrime))
        print("Determinant of the submatrix: {}".format(determinant))
        print("Number of 0 components on the diagonal:", np.count_nonzero(np.diagonal(subMatrix==0)))
        print(np.diagonal(subMatrix))
        #print(np.diagonal(subMatrix))



