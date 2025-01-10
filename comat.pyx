
# cython: c_string_type=str, c_string_encoding=ascii
# Define this directive in your .pyx file or in `Extension` to suppress the deprecated NumPy API warning.
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from numpy import zeros, float64, dot
from numpy.linalg import LinAlgError
from cython.parallel import prange
from numpy cimport float64_t
from libc.math cimport log

def logdet_quad(const float64_t[::1] a, const float64_t[::1] b):
    """Calculate the log determinant of the real symmetric Toeplitz matrix M, 
       and the quadratic form, b.T M^{-1} b.
       This algorithm is adapted from the Levinson-Durbin algorithm.

       The Hermitian version has yet to be implemented.

    Parameters
    ----------
    a : array, dtype=double, shape=(n,)
        The first row of the matrix  (see below)
    b : array, dtype=double, shape=(n,)
        The vector in the quadratic form. 
    Both a and y must have the same type (double or complex128).

    Notes
    -----
    For example, the 5x5 toeplitz matrix below should be represented as
    the linear array ``a`` on the right ::

        [ a0   a1   a2  a3  a4 ]
        [ a1   a0   a1  a2  a3 ]
        [ a2   a1   a0  a1  a2 ] -> [a0  a1  a2  a3  a4]
        [ a3   a2   a1  a0  a1 ]
        [ a4   a3   a2  a1  a0 ]

    Returns
    -------
    logdet : double, 
        The log determinant of M.
    quad : double,
        The quadratic form, b.T M^{-1} b
    """
    
    cdef ssize_t n, m, j, k
    n = b.shape[0]
    cdef float64_t alpha, logdet, aux, quad, inv_a0, inv_beta, mu
    cdef float64_t[:] r = zeros(n-1, dtype=float64)  # aux vector
    cdef float64_t[:] x = zeros(n, dtype=float64)  # Tx = b
    cdef float64_t[:] y = zeros(n, dtype=float64)  # Ty = r
    cdef float64_t[:] aux_mu=zeros(n, dtype=float64)
    cdef float64_t[:] aux_det=zeros(n, dtype=float64)
    cdef float64_t[:] aux_A=zeros(n, dtype=float64)
    

    inv_a0 = 1.0 / a[0]
    for k in prange(n-1, nogil=True):
        r[k] = a[k+1] * inv_a0

    x[0] = b[0]
    #y[0] = -r[0]
    y[0] = 1.0
    inv_beta = 1.0
    logdet = 0.0
    alpha = -r[0]

    for k in range(1, n):
        #beta = (1 - alpha * alpha) * beta 
        #inv_beta = 1.0 / beta 
        inv_beta = inv_beta / (1 - alpha * alpha) 

        for j in range(k-1):
            y[j] = y[j] + alpha * y[k-j-2]
        y[k-1] = alpha  

        mu = b[k]

        for j in prange(k, nogil=True):
            aux_mu[j] = r[j] * x[k-j-1]
            aux_det[j] = r[j] * y[j]
        mu = (mu-np.sum(aux_mu[:k])) * inv_beta
        aux = 1.0 + np.sum(aux_det[:k])
        logdet += log(aux)
        
        for j in prange(k, nogil=True):
            m = k - j - 1
            x[j] = x[j] + mu * y[m] 
            aux_A[j] = r[j] * y[m]
        x[k] = mu
        alpha = - (r[k] + np.sum(aux_A[:k])) * inv_beta

    logdet = logdet + n*log(a[0])
    quad = dot(x, b) * inv_a0
    return logdet, quad


