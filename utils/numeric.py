import numpy as np
import tensorflow as tf
from numba import jit
from scipy.special import sph_harm, eval_genlaguerre


def finite_difference_derivative(f, dx=1e-5):
    """Calculate first derivative using central difference"""
    f_plus = tf.roll(f, shift=-1, axis=0)
    f_minus = tf.roll(f, shift=1, axis=0)
    return (f_plus - f_minus) / (2 * dx)

def finite_difference_second_derivative(f, dx=1e-5):
    """Calculate second derivative using central difference"""
    f_plus = tf.roll(f, shift=-1, axis=0)
    f_minus = tf.roll(f, shift=1, axis=0)
    return (f_plus - 2*f + f_minus) / (dx**2)

def finite_difference_laplacian(r, psi):
    """Approximate the Laplacian using finite differences with improved stability."""
    dr = 1e-6
    r = tf.cast(r, tf.complex64)
    # Use central difference with appropriate spacing
    psi_plus = tf.roll(psi, shift=-1, axis=0)
    psi_minus = tf.roll(psi, shift=1, axis=0)
    
    # Include radial term in Laplacian (d²/dr² + (2/r)d/dr)
    d2psi_dr2 = tf.cast((psi_plus - 2 * psi + psi_minus) / (dr ** 2), tf.complex64)
    dpsi_dr = tf.cast((psi_plus - psi_minus) / (2 * dr), tf.complex64)
    
    # Add radial term with safe division
    epsilon = 1e-10
    radial_term = (2 / (r + epsilon)) * dpsi_dr
    
    return d2psi_dr2 + radial_term

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