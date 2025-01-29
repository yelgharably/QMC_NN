import numpy as np
from ..config import CONFIG
import tensorflow as tf
from scipy.special import sph_harm, eval_genlaguerre

from ..models.hydrogen_model import HydrogenModel
from ..training.losses import Losses
from ..qmc.qmc_generator import QMC_gen

def main_1():
    generator = QMC_gen()
    r = tf.constant([1.0],dtype=tf.float64)
    theta = tf.constant([0.0],dtype=tf.float64)
    phi = tf.constant([0.0],dtype=tf.float64)

    n = 1
    l = 0
    m = 0
    a0 = 1.0

    psi_gen = generator.trial_wfc(r,theta,phi,n,l,m)
    
    def Rnl(r,n,l):
        a0 = 1.0
        lag_1 = n - l - 1
        lag_2 = 2 * l + 1
        lag_3 = 2 * r / (n * a0)
        return r**l * tf.exp(-r / (n * a0)) * eval_genlaguerre(lag_1, lag_2, lag_3)

    # Compute spherical harmonics arguments
    sph_1 = m
    sph_2 = l
    sph_3 = phi
    sph_4 = theta

    psi_analytical = tf.cast((Rnl(a0,1,0) * sph_harm(sph_1, sph_2, sph_3, sph_4)),tf.complex64)

    print(f"Wavefunction Generated value: {psi_gen}, Wavefunction Analytical value: {psi_analytical}")
    print(f"Residual: {psi_gen - psi_analytical}")

def main():
    generator = QMC_gen()
    generator.plot_wavefunction(n_samples=10000)

if __name__ == "__main__":
    main()