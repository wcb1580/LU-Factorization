# ENGSCI233: Lab - Numerical Errors

# PURPOSE:
# To IMPLEMENT LU factorisation with partial pivoting and solve matrix equation.

# SUBMISSION:
# - YOU MUST submit this file.

# TO DO:
# - READ all the comments in a function before modifying it
# - COMPLETE the functions lu_factor(), lu_forward_sub() and lu_backward_sub()
# - TEST each function is working correctly by writing test functions in test_errlab.py
# - DO NOT modify the other function(s)

# imports
from pprint import pprint

import fraction as fraction
import numpy as np
from copy import copy
from fractions import Fraction


# this function is complete
def lu_read(filename):
    """
	This function will read in matrix data from an external file that adheres to an appropriate format. It will then
	return both the coefficient matrix A and vector of constants b for a system of linear equations, Ax=b.

	"""
    with open(filename, 'r') as fp:
        # Row dimension
        nums = fp.readline().strip()
        row = int(nums)

        A = []
        for i in range(row):
            nums = fp.readline().rstrip().split()
            A.append([float(num) for num in nums])
        A = np.array(A)

        b = []
        nums = fp.readline().rstrip().split()
        b.append([float(num) for num in nums])
        b = np.array(b)

    return A, b.T


# **this function is incomplete**
def lu_factor(A, pivot=False):
    ''' The function lu_factor() is to decompose matrix A in L U , with options to determine if
    	partial pivoting would be used or not. It retunrs a stitch matrix of L and U

		Parameters
        ----------
    	A : a square and full rank matrix.

    	pivot: A boolean variable that indicate whether partial pivoting is applied(True) or not(False).

        Returns
        -------
        A : a stitch matrix of L and U .

        Notes
        -----
        The functionality of the method is independent on whether partial pivoting is applied or not.

        The return is a stitch matrix of L and U, it requires further unpack outside this function if individual matrix of L and U is required


        Examples
        -----
        When partial pivoting is not applied:
        >>> A=np.array([[2.0,  3.0, -4.0,  2.0],[-4.0,-5.0,6.0,-3.0],[2.0,2.0,1.0,0.0],[-6.0,-7.0,14.0,-4.0]])
		>>> A = lu_factor(A, pivot=False)
        >>> A
        np.array([ [ 2.0,3.0 ,-4.0 ,2.0 ], [-2.0 ,1.0 , -2.0, 1.0], [ 1.0, -1.0,3.0 , -1.0], [-3.0,2.0,2.0,2.0]])

        When partial pivoting is applied:
        >>> A=np.array([[2.0,  3.0, -4.0,  2.0],[-4.0,-5.0,6.0,-3.0],[2.0,2.0,1.0,0.0],[-6.0,-7.0,14.0,-4.0]])
		>>> A = lu_factor(A, pivot=True)
        >>> A
        np.array([ [ 2.0,3.0 ,-4.0 ,2.0 ], [-2.0 ,1.0 , -2.0, 1.0], [ 1.0, -1.0,3.0, -1.0], [-3.0,2.0,2.0,2.0]])

    '''

    # get dimensions of square matrix
    n = np.shape(A)[0]

    # create initial row swap vector: p = [0, 1, 2, ... n]
    p = np.arange(n)
    # loop over each row in the matrix
    # **hint** what is the pivot index, row and column?
    L = np.zeros([n, n])
    for i in range(n):
        # Step 2: Row swaps for partial pivoting
        #    DO NOT attempt until Steps 0 and 1 below are confirmed to be working.

        if pivot:
            loc = np.where(abs(A[i:, i]) == np.max(abs(A[i:,i])))# Obtain the position of the largest absolute value in the column below the diagonial line
            if np.shape(loc[0])==1:
                index = loc[0] + i
            else:
                index=loc[0][0]+i
            temp = copy(A[index, :])  # Record relevant row that has the largest absolute value
            temp1 = copy(A[i, :])
            A[i, :] = temp  # switch the relevant row in matrix A
            A[index, :] = temp1
            p[i] = index
            t = copy(L[i, :])  # swap the L matrix if there is a swap in the U matrix
            t1 = copy(L[index, :])
            L[i, :] = t1
            L[index, :] = t
            P = (A[i, i])  # pefrom factorisation
            for row in range(i + 1, n):
                L[row, i] = (A[row, i] / P)
                A[row, :] = A[row, :] - (A[row, i] / P) * A[i, :]
        else:
            P = (A[i, i])  # peform LU factorisation if partial pivoting is not applied
            for row in range(i + 1, n):
                L[row, i] = (A[row, i] / P)
                A[row, :] = A[row, :] - (A[row, i] / P) * A[i, :]

    for i in range(1, n):  # record the value of L into stitch matrix A
        A[i, 0:i] = L[i, 0:i]
    return A, p


def lu_forward_sub(L, b, p=None):
	
    # check shape of L consistent with shape of b (for matrix multiplication L^T*b)
    assert np.shape(L)[0] == len(b), 'incompatible dimensions of L and b'
    '''The function lu_forward_sub() returns the result of forward substution of Ly=b in an array, with or without partial pivoting applied when calculating the stitch matrix from lu_factor()

		Parameters
        ----------
    	L : a stitich matrix of L and U from LU decompostion 
     
        p : a variable indicates whether partial pivoting is applied on calculating the stitich matrix or not.
        
        b : an array of result of Ux = b

        Returns
        -------
        b : the result for Ux = b

        Notes
        -----
        The functions should return different array value when partial pivoting is applied or not

        Examples
        -----
        When partial pivoting is not applied:
        >>> A= np.array([ [ 2.0,3.0 ,-4.0 ,2.0 ], [-2.0 ,1.0 , -2.0, 1.0], [ 1.0, -1.0,3.0, -1.0], [-3.0,2.0,2.0,2.0]])
        >>> b = np.array([4.0,0.0,5.0,8.0])
        >>> b = lu_forward_sub(A, b, p=None)
        >>> b
        np.array([4.0, 8.0, 9.0, -14.0])
        
        When partial pivoting is applied:
        >>> A= np.array([ [ 2.0,3.0 ,-4.0 ,2.0 ], [-2.0 ,1.0 , -2.0, 1.0], [ 1.0, -1.0,3.0, -1.0], [-3.0,2.0,2.0,2.0]])
        >>> b = np.array([4.0,0.0,5.0,8.0])
        >>> b = lu_forward_sub(A, b, p=not None)
        >>> b
        np.array([6.0, 6.0, 14.0, -2.0])

    '''
    # Step 0: Get matrix dimension
    # **hint** See how this is done in lu_factor()
    n = np.shape(L)[0]
    # Step 2: Perform partial pivoting row swaps on RHS
    if p is not None: # if partial pivoting is applied on LU decompostion 
        for i in range(len(p)): 
            loc = p[i]
            swap = copy(b[i]) # swap the relevant value in b if its row in LU decomposition is swapped.
            swap1 = copy(b[loc])
            b[loc] = swap
            b[i] = swap1
            i += 1
    y = np.zeros(len(b)) # construct the answer array for Ux = b 
    for i in range(len(b)):
        if i == 0:
            y[0] = b[0] 
        else:
            index = 0 # preform forward substitution for Ux = b
            longpart = 0
            while index <= i:
                longpart = longpart - L[i][index] * y[index]
                index += 1
            y[i] = b[i] + longpart 
    b = y
    return b 


# **this function is incomplete**
def lu_backward_sub(U, y):
    # check shape consistency
    x = np.zeros(len(y))
    # check shape consistency
    assert np.shape(U)[0] == len(y), 'incompatible dimensions of U and y'
    ''' The function lu_backward_sub() performs and returns the result of backward substitution Ux=y 
        ----------
    	U : a stitich matrix of L and U from LU decompostion 
        
        y : an array of result of Ux = b

        Returns
        -------
        x : the result for Ux = b

        Notes
        -----
        The functions should return the same array no matter partial pivoting is applied or not.

        Examples
        -----
        >>> A = np.array([[2.0,3.0,-4.0,2.0],[-2.0,1.0,-2.0,1.0],[1.0,-1.0,3.0,-1.0],[-3.0,2.0,2.0,2.0]])
        >>> y = np.array([4.0, 8.0, 9.0, -14.0])
        >>> x = lu_backward_sub(A, y)
        >>> x
        np.array([1.0,2.0,3.0,4.0])

    '''
    n = np.shape(U)[0]
    # Perform backward substitution operations
    for i in range(len(y) - 1, -1, -1):
        if i == len(y) - 1:
            x[i] = y[i] / U[i, i]
        else:
            total = 0
            index = i + 1
            while index <= len(y) - 1: # use a while loop to calculate the x values in the result array
                total = total - U[i][index] * x[index]
                index += 1
            x[i] = (y[i] + total) / U[i][i]
    return x
