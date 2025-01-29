"""
GO BACK TO JUST USING THE XY AXIS
"""



import tensorflow as tf
import numpy as np
from scipy.special import sph_harm, eval_genlaguerre
from ..utils import numeric
from ..physics import operators
from ..qmc.qmc_generator import QMC_gen
from ..config import CONFIG

sch_w = CONFIG['training']['weights']['sch']
norm_w = CONFIG['training']['weights']['nor']
bound_w = CONFIG['training']['weights']['bound']
eng_w = CONFIG['training']['weights']['eng']

class Losses:
    def __init__(
        self,
        delta_r,
        config=CONFIG,
        dtype=tf.complex64,
        sch_w=CONFIG['training']['weights']['sch'],
        norm_w=CONFIG['training']['weights']['nor'],
        bound_w=CONFIG['training']['weights']['bound'],
        eng_w=CONFIG['training']['weights']['eng']
    ):
        self.config = config
        self.dtype = dtype
        self.delta_r = delta_r
        # store your weight factors
        self.sch_w = sch_w
        self.norm_w = norm_w
        self.bound_w = bound_w
        self.eng_w = eng_w

    def gen(r,theta,phi,n,l,m):
        generator = QMC_gen()
        psi = generator.trial_wfc(r,theta,phi,n,l,m)
        return psi

    # def schrodinger_loss(self, r, theta, phi, n, l, m, psi):
    #     r = tf.cast(r,tf.complex64)
    #     psi = tf.cast(psi,tf.complex64)
    #     epsilon = 1e-10

    #     V = operators.potential_operator(r+epsilon)
    #     K = operators.kinetic_operator(r,psi)

    #     generator = QMC_gen()
    #     ideal_sch = generator.trial_wfc(r,theta,phi,n,l,m)
    #     residual = (K+V) - ideal_sch
        
    #     return tf.cast(tf.reduce_mean(tf.square(tf.abs(residual))),tf.float64)

    def schrodinger_loss(self, r, theta, phi, n, l ,m , psi_pred):
        """Compute Schrödinger loss to enforce the eigenvalue equation Hψ = Eψ."""
        # Laplacian (kinetic energy term)
        
        generator = QMC_gen()
        psi_true = generator.trial_wfc(r,theta,phi,n,l,m)

        r = tf.cast(r, tf.complex64)

        K_pred = -0.5 * numeric.finite_difference_laplacian(r, psi_pred)
        V_pred = -1.0 / (r) * psi_pred  # Potential energy term
        H_pred = K_pred + V_pred  # Hamiltonian applied to ψ_pred

        K_true = -0.5 * numeric.finite_difference_laplacian(r, psi_true)
        V_true = -1.0 / (r) * psi_true
        H_true = K_true + V_true

        # Compute energy eigenvalue for predicted ψ
        energy_pred = tf.reduce_sum(tf.math.conj(psi_pred) * H_pred) / tf.reduce_sum(
            tf.math.conj(psi_pred) * psi_pred
        )

        # Compute energy eigenvalue for true ψ
        energy_true = tf.reduce_sum(tf.math.conj(psi_true) * H_true) / tf.reduce_sum(
            tf.math.conj(psi_true) * psi_true
        )

        # Residual loss: Hψ - Eψ for predicted ψ
        residual_pred = H_pred - energy_pred * psi_pred
        residual_loss_pred = tf.reduce_mean(tf.square(tf.abs(residual_pred)))

        # Residual loss: Hψ - Eψ for true ψ (optional)
        residual_true = H_true - energy_true * psi_true
        residual_loss_true = tf.reduce_mean(tf.square(tf.abs(residual_true)))

        # Combine residual losses (optional: weight residual losses)
        total_loss = residual_loss_pred + residual_loss_true

        return tf.cast(total_loss, tf.float64)
    
    def normalization_loss(self, r, theta, psi, dr=1e-5, dtheta=1e-5):
        probability_density = tf.cast(tf.abs(psi)**2,tf.float64)
        volume_element = tf.cast(r**2 * tf.sin(theta), tf.float64)
        normalization_integral = tf.reduce_mean(probability_density * volume_element * tf.cast(dr,tf.float64) * tf.cast(dtheta,tf.float64))
        target_normalization = tf.constant(1.0, dtype=tf.float64)
        return tf.square(tf.cast(normalization_integral,tf.float64) - tf.cast(target_normalization,tf.float64))
    
    def boundary_conditions(self, r, psi, decay_rate=1.0, cutoff=5.0):
        """Enhanced boundary conditions with exponential decay."""
        # Compute probability density
        probability_density = tf.abs(psi)**2

        # Expected exponential decay
        decay_factor = tf.exp(-decay_rate * r)

        # Calculate deviation from expected decay (weighted by decay_factor)
        decay_loss = tf.reduce_mean(tf.square(probability_density - decay_factor))

        # Strong penalty for non-zero values at large r
        large_r_mask = tf.cast(r > cutoff, tf.float32)  # Define cutoff
        large_r_penalty = tf.reduce_mean(probability_density * large_r_mask)

        # Weighted sum of decay loss and large r penalty
        return tf.cast(decay_loss + 10.0 * large_r_penalty, tf.float64)

    def energy_eigenvalue_metric(self, r, psi):
        """Calculate energy eigenvalue using the Rayleigh quotient."""
        # Cast inputs
        r = tf.cast(r, self.dtype)
        psi = tf.cast(psi, self.dtype)
        
        # Compute Laplacian (kinetic energy term)
        d2psi_dr2 = numeric.finite_difference_laplacian(r, psi)
        kinetic = -0.5 * d2psi_dr2  # Tψ = -1/2 ∇²ψ

        # Compute potential energy term
        epsilon = 1e-10  # Avoid division by zero
        potential = -1.0 / (r + epsilon)  # Vψ = -1/r ψ
        potential_term = potential * psi

        # Hamiltonian applied to ψ
        H_psi = kinetic + potential_term

        # Compute numerator: ⟨ψ|H|ψ⟩
        numerator = tf.reduce_sum(tf.math.conj(psi) * H_psi * r**2)  # Include volume element

        # Compute denominator: ⟨ψ|ψ⟩
        denominator = tf.reduce_sum(tf.math.conj(psi) * psi * r**2)

        # Energy eigenvalue
        energy = numerator / denominator

        # Return real part of energy eigenvalue
        return tf.cast(tf.abs(energy),tf.float64)
    
    def custom_loss(self, y_true, y_pred):
        """
        Custom loss function for training the wavefunction.
        y_true: Dictionary containing true wavefunction ('psi_true') and coordinates
        y_pred: Dictionary containing predicted wavefunction ('psi_pred') and coordinates
        """
        # Extract quantum/coordinate data (identical in both dictionaries)
        r = y_true['r']
        theta = y_true['theta']
        phi = y_true['phi']
        n = y_true['n']
        l = y_true['l']
        m = y_true['m']

        # Extract true and predicted wavefunctions
        psi_true_real = y_true['psi_real']
        psi_true_imag = y_true['psi_imag']
        psi_pred_real = y_pred['psi_real']
        psi_pred_imag = y_pred['psi_imag']

        psi_pred = tf.complex(psi_pred_real, psi_pred_imag)

        # Compute losses
        sch_loss = self.schrodinger_loss(r, theta, phi, n, l, m, psi_pred)
        norm_loss = self.normalization_loss(r, theta, psi_pred)
        bound_loss = self.boundary_conditions(r, psi_pred)
        energy_loss = self.energy_eigenvalue_metric(r, psi_pred)

        # Combine losses with weights
        total_loss = (
            self.sch_w * sch_loss +
            self.norm_w * norm_loss +
            self.bound_w * bound_loss +
            self.eng_w * energy_loss
        )

        return tf.cast(total_loss, tf.float64)

    def make_loss_fn(self,losses_obj):
        """
        Return a Keras-compatible function that computes psi_true on the fly
        and calls the PDE-based losses or wavefunction mismatch.
        """
        def custom_loss_fn(y_true, y_pred):
            # Extract wavefunctions
            print("Shape of y_pred:", y_pred.shape)
            print("Shape of y_true:", y_true.shape)
            psi_true_real = y_true[:,0]
            psi_true_imag = y_true[:,1]
            psi_pred_real = y_pred[:,0]
            psi_pred_imag = y_pred[:,1]

            psi_pred = tf.complex(psi_pred_real, psi_pred_imag)
            psi_true = tf.complex(psi_true_real, psi_true_imag)

            # Extract spatial and quantum coordinates
            r = y_true[:,2]
            theta = y_true[:,3]
            phi = y_true[:,4]
            n = y_true[:,5]
            l = y_true[:,6]
            m = y_true[:,7]

            # Compute individual losses
            sch_loss = self.schrodinger_loss(r, theta, phi, n, l, m, psi_pred)
            norm_loss = self.normalization_loss(r, theta, psi_pred)
            bound_loss = self.boundary_conditions(r, psi_pred)
            energy_loss = self.energy_eigenvalue_metric(r, psi_pred)

            # Combine losses
            total_loss = sch_loss + norm_loss + bound_loss + energy_loss
            return tf.cast(total_loss, tf.float64)

        return custom_loss_fn

        
