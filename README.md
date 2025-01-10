# comat

This repository contains a highly optimized implementation of real symmetric **co**variance **mat**rix (comat) operations.

---
## Usage

### 1. logdet_quad(a, b):
The Levinson-Durbin algorithm in Cython, designed for efficient computation of the logrithmic determinant of the Toeplitz matrix, as well as the quadratic form with the inverse of the matrix.

```python
import numpy as np
from comat import logdet_quad

# Toeplitz matrix coefficients:
a = np.array([4.0, -1.0, -0.5, -0.25], dtype=np.float64)

# Vector in the quadractic form of interest, b.T M^{-1} b:
b = np.array([2.0, 1.0, 0.5, 0.25], dtype=np.float64)

# Solve using the adapted Levinson-Durbin
logdet, quad = logdet_quad(a, b)
print(f"logdet: {logdet}, quad: {quad}")
```


---

## Requirements
 - Python: 3.6 or newer
 - Cython: 0.29.21 or newer
 - NumPy: 1.18 or newer
 - Compiler:
	 - GCC with OpenMP (preferred)
	 - Clang with libomp (for macOS)

---

## Installation

1. Clone the Repository:
```shell
git clone https://github.com/zzhang0123/comat.git
cd comat
```
2. Install Dependencies: **numpy** and **cython**.

3. Build the Module:
```shell
python setup.py build_ext --inplace
```




