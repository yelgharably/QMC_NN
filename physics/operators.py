import tensorflow as tf
from ..utils.numeric import finite_difference_derivative, finite_difference_second_derivative, finite_difference_laplacian

def kinetic_operator(r,wavefunction):
    return -0.5 * finite_difference_laplacian(r,wavefunction)

def potential_operator(r):
    return -1/r