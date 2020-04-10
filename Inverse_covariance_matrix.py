import numpy as np 

def get_inverse_covariance_matrix():

    #? Read in the covariance matric which has already been calculated.
    covarianceMatrix = np.loadtxt("Covariance_matrix.txt", delimiter='  ')

    #? Invert the matrix and check that it has inverted properly.
    inverseCMatrix = np.linalg.inv(covarianceMatrix.copy())
    goodInverse = np.allclose(np.dot(covarianceMatrix, inverseCMatrix), np.eye(len(covarianceMatrix[:,0])))
    conditionNumber = np.linalg.cond(covarianceMatrix)

    print("The condition number of the covariance matrix: {}".format(conditionNumber))
    print("The covariance matrix has inverted properly: {}".format(goodInverse))

    return inverseCMatrix

def get_inverse_covariance_matrix_nochecks():

    #? Read in the covariance matric which has already been calculated.
    covarianceMatrix = np.loadtxt("Covariance_matrix.txt", delimiter='  ')

    #? Invert the matrix and check that it has inverted properly.
    inverseCMatrix = np.linalg.inv(covarianceMatrix.copy())
    
    return inverseCMatrix

def get_l0l0_submatrix_inverse():
    matrix = np.loadtxt("l0l0_submatrix.txt", delimiter='   ')

    inverse = np.linalg.inv(matrix.copy())

    goodInverse = np.allclose(np.dot(matrix, inverse), np.eye(len(matrix[:,0])))
    conditionNumber = np.linalg.cond(matrix)

    if not goodInverse:
        print('Matrix inverse is bad')
    #print("The condition number of the covariance matrix: {}".format(conditionNumber))
    #print("The covariance matrix has inverted properly: {}".format(goodInverse))

    return inverse

def get_l_and_l2_covarianceMatrix_inverse():
    matrix = np.loadtxt("l0_and_l2_covarianceMatrix.txt", delimiter='   ')

    inverse = np.linalg.inv(matrix.copy())

    goodInverse = np.allclose(np.dot(matrix, inverse), np.eye(len(matrix[:,0])))
    conditionNumber = np.linalg.cond(matrix)

    if not goodInverse:
        print('Matrix inverse is bad')
    #print("The condition number of the covariance matrix: {}".format(conditionNumber))
    #print("The covariance matrix has inverted properly: {}".format(goodInverse))

    return inverse




