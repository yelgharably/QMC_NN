from numba import jit
import numpy as np
from scipy.special import sph_harm, eval_genlaguerre

@jit(nopython=True)
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# def eval_genlaguerre_f(n, alpha, x):
#     result = 0.0
#     fact_n_alpha = factorial(n + alpha)  # Precompute factorials
#     for k in range(n + 1):
#         fact_n_k = factorial(n - k)
#         fact_k = factorial(k)
#         coeff = fact_n_alpha / (fact_n_k * fact_k)
#         result += coeff * ((-1)**k * x**k)
#     return result

@jit(nopython=True)
def normalization_constant(l, m=0):
    return np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))

@jit(nopython=True)
def legendre_polynomial(l, x):
    if l == 0:
        return 1.0
    elif l == 1:
        return x
    else:
        P_prev = 1.0  # P_0(x)
        P_curr = x    # P_1(x)
        for i in range(2, l + 1):
            P_next = ((2 * i - 1) * x * P_curr - (i - 1) * P_prev) / i
            P_prev, P_curr = P_curr, P_next
        return P_curr
    
@jit(nopython=True)
def spherical_harmonic(theta, l):
    """Simplified spherical harmonic for m=0, phi=0"""
    N = normalization_constant(l)
    P_lm = legendre_polynomial(l, np.cos(theta))
    return N * P_lm

@jit(nopython=True)
def spherical_harmonic_magnitude_squared(theta, l, m=0):
    N = normalization_constant(l)
    P_lm = legendre_polynomial(l, np.cos(theta))
    return (N * P_lm)**2
    
def radial_part(r, n, l):
    # Correct normalization constant
    term_1 = np.sqrt((2 / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    term_2 = np.exp(-r / n)
    term_3 = (2 * r / n)**l
    term_4 = eval_genlaguerre(n - l - 1, 2 * l + 1, 2 * r / n)
    return term_1 * term_2 * term_3 * term_4

def main():
    l = 1
    theta = np.pi / 2
    print(spherical_harmonic(theta, l))
    print(spherical_harmonic_magnitude_squared(theta, l))

if __name__ == "__main__":
    main()

