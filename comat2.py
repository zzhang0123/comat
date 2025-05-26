import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.linalg._solve_toeplitz import levinson


def log_det_symmetric_toeplitz(r):
    r = np.asarray(r)
    n = len(r)  
    a0 = r[0] 
    
    b = np.zeros(n, dtype=r.dtype)
    b[:n-1] = -r[1:] 
    
    a = np.concatenate((r[::-1], r[1:]))
    x, reflection_coeff = levinson(a, b)
    
    k = reflection_coeff[1:n]  
    k = np.clip(k, -0.999999, 0.999999)  # Clip values close to 1 to avoid numerical issues
    
    factors = np.arange(n-1, 0, -1)
    terms = np.log(1 - k**2)
    result=n * np.log(a0) + np.dot(factors, terms) 
    if np.isnan(result):
        return -np.inf
    else:
        return result

def logdet_quad(corr_list, data):
    # Add parameter validation
    if np.any(np.isnan(corr_list)) or np.any(np.isinf(corr_list)):
        return -np.inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return -np.inf
        
    try:
        result = np.dot(data, solve_toeplitz(corr_list, data)) + log_det_symmetric_toeplitz(corr_list)
        return -0.5 * result
    except:
        return -np.inf
