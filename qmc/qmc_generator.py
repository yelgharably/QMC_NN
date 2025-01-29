import numpy as np
from ..config import CONFIG
from scipy.special import sph_harm, eval_genlaguerre
import tensorflow as tf
import scipy
from multiprocessing import Pool
from tqdm import tqdm
import time

sample_n = CONFIG['training']['sample_size']
grid_n = CONFIG['training']['grid_size']
a0 = 1.0

class QMC_gen:
    """Quantum Monte Carlo data generator using Metropolis sampling."""
    def __init__(self, n_samples=sample_n, burn_in=5000, a0=1.0):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.a0 = a0
        self.normalization_params = None  # To store normalization parameters
    
    def trial_wfc(self, r):
        """1s orbital wavefunction."""
        return (1.0 / np.sqrt(np.pi)) * (1.0 / self.a0) ** (1.5) * np.exp(-r / self.a0)
    
    def metropolis_sampling(self):
        """Generate samples for the radial coordinate r using Metropolis algorithm."""
        samples = []
        r = np.random.uniform(0.01 * self.a0, 5.0 * self.a0)  # Wider initial range
        
        # Burn-in period
        for _ in range(10000):  # Increased burn-in
            r_new = r + np.random.normal(scale=0.2 * self.a0)  # Tuned step size
            if r_new > 0:
                ratio = (r_new**2 * self.trial_wfc(r_new)**2) / (r**2 * self.trial_wfc(r)**2)
                if np.random.rand() < min(1, ratio):
                    r = r_new  # Accept new position
        
        # Sampling
        for _ in range(self.n_samples):
            r_new = r + np.random.normal(scale=0.2 * self.a0)  # Tuned step size
            if r_new > 0:
                ratio = (r_new**2 * self.trial_wfc(r_new)**2) / (r**2 * self.trial_wfc(r)**2)
                if np.random.rand() < min(1, ratio):
                    r = r_new  # Accept new position
            samples.append(r)
        
        return np.array(samples)
    
    def gen_samples(self):
        """Generate normalized samples for r."""
        r_samples = self.metropolis_sampling()
        
        # Normalize r
        r_mean, r_std = np.mean(r_samples), np.std(r_samples) + 1e-8
        r_normalized = (r_samples - r_mean) / r_std
        
        return r_normalized
    
    def plot_wavefunction(self, n_samples=1000, r_max=10.0):
        """Plot 1D radial wavefunction comparison."""
        import matplotlib.pyplot as plt
        
        # Generate r values for exact function
        r_exact = np.linspace(0.01, r_max, 1000)
        psi_exact = self.trial_wfc(r_exact)
        prob_density_exact = np.abs(psi_exact)**2  # No r² weighting here
        prob_density_exact /= np.trapz(prob_density_exact, r_exact)  # Normalize exact probability density

        # Generate normalized samples using gen_samples
        r_samples = self.gen_samples()
        r_samples = (r_samples * np.std(r_exact)) + np.mean(r_exact)  # Rescale normalized samples to original range

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(r_samples, bins=40, density=True, alpha=0.5, label='Gen Samples (Normalized)')
        plt.plot(r_exact, prob_density_exact, 'r-', label='Exact')
        plt.xlabel('r (Bohr radii)')
        plt.ylabel('|ψ|² (Normalized)')
        plt.title('Hydrogen 1s Orbital Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()

class QMC_gen_new:
    """Quantum Monte Carlo data generator using Metropolis sampling."""
    def __init__(self, n_samples=sample_n, burn_in=5000, a0=1.0):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.a0 = a0
        self.normalization_params = None  # To store normalization parameters
    
    def trial_wfc(self, r, theta, n, l=0, m=0):
        r = tf.cast(r, tf.float32)
        theta = tf.cast(theta, tf.float32)
        phi = tf.zeros_like(theta, dtype=tf.float32)
        n = tf.cast(n, tf.float32)

        # Compute Laguerre polynomial arguments
        lag_1 = n - l - 1
        lag_2 = 2 * l + 1
        lag_3 = 2 * r / (n * a0)

        # Compute spherical harmonics arguments
        sph_1 = m
        sph_2 = l
        sph_3 = phi
        sph_4 = theta

        # Radial wavefunction (Rnl)
        radial_part = tf.cast(r**l * tf.exp(-r / (n * a0)), tf.float32)
        laguerre_part = tf.cast(
            eval_genlaguerre(n-1, 2*l+1, 2*r/(n*a0)),
            tf.float32)
        Rnl = tf.cast(radial_part * laguerre_part, tf.complex64)

        # Spherical harmonics (Ylm)
        Ylm = tf.cast(sph_harm(m,l,phi,theta), tf.complex64)

        # Return the product of Rnl and Ylm
        return tf.multiply(Rnl, Ylm)
    
    def generate_base_grid(self):
        """Generate base 2D grid for (r, theta)."""
        r = np.linspace(0, CONFIG['physics']['r_infinity'], self.grid_n)
        theta = np.linspace(0, np.pi / 2, self.grid_n)  # Restrict theta to [0, π/2]
        R, THETA = np.meshgrid(r, theta, indexing='ij')
        return R.flatten(), THETA.flatten()
    
    def metropolis_2d_sampling(self, grid_points, n_samples_per_point=10):
        """Metropolis sampling in 2D (r, theta)."""
        r, theta = grid_points
        samples = []
        count = 0
        start_time = time.time()
        time_lst = []

        for r_old, theta_old in zip(r, theta):
            for _ in range(n_samples_per_point):
                # Propose new points
                r_new = r_old + np.random.normal(0, 0.1 * self.a0)
                theta_new = theta_old + np.random.normal(0, 0.1)

                # Ensure bounds
                if r_new > 0 and 0 <= theta_new <= np.pi / 2:
                    # Compute acceptance ratio
                    psi_old = self.trial_wfc(r_old, theta_old, n=1)  # n is set to 1 for simplicity
                    psi_new = self.trial_wfc(r_new, theta_new, n=1)
                    acceptance_ratio = abs(psi_new / psi_old)**2

                    # Accept or reject
                    if np.random.rand() < acceptance_ratio:
                        samples.append([r_new, theta_new])
            count += 1
            if (count % 1000) == 0:
                print(f"[DEBUG] Processed {count} / {grid_points} grid points so far.\nTime elapsed: {time.time() - start_time:.2f} seconds")
                time_lst.append(time.time() - start_time)
                start_time += time.time() - start_time

        print(f"[DEBUG] Average time elapsed for each 1000 grid points: {np.mean(np.array(time_lst))}\nTotal Time elapsed: {np.sum(np.array(time_lst)):.2f} seconds")
        return np.array(samples)

    def metropolis_3d_sampling_old(self, grid_points, n_samples_per_point=10):
        """Enhanced Metropolis sampling around grid points."""
        samples = []
        total_points = len(grid_points[0])
        print(f"[DEBUG] Starting metropolis_3d_sampling on {total_points} grid points")
        count = 0
        for r, theta, phi in zip(grid_points[0], grid_points[1], grid_points[2]):
            # Sample around each grid point
            for _ in range(n_samples_per_point):
                r_new = r + np.random.normal(0, 0.1 * self.a0)
                theta_new = theta + np.random.normal(0, 0.1)
                phi_new = phi + np.random.normal(0, 0.1)
                # Ensure bounds
                if r_new > 0 and 0 <= theta_new <= np.pi and 0 <= phi_new <= 2*np.pi:
                    wfc_ratio = abs(self.trial_wfc(r_new, theta_new, phi_new, 1, 0, 0) / 
                                self.trial_wfc(r, theta, phi, 1, 0, 0))**2
                    
                    if np.random.rand() < min(1, wfc_ratio):
                        samples.append([r_new, theta_new, phi_new])
            count += 1
            if (count % 1000) == 0:
                print(f"[DEBUG] Processed {count} / {total_points} grid points so far.")
        
        return np.array(samples)
    
    def metropolis_3d_sampling(self, grid_points, n_samples_per_point=10):
        """Optimized Metropolis sampling."""
        with tf.device('/CPU:0'):
            samples = []
            total_points = len(grid_points[0])
            print(f"[DEBUG] Starting metropolis_3d_sampling on {total_points} grid points")

            # Cache frequently used constants and functions
            rng_rand = np.random.rand
            rng_normal = np.random.normal
            trial_wfc = self.trial_wfc
            a0 = self.a0
            pi = np.pi
            count = 0
            start_time = time.time()
            time_lst = []

            r, theta, phi = grid_points

            for i in range(total_points):
                r_old, theta_old, phi_old = r[i], theta[i], phi[i]
                psi_old = trial_wfc(r_old, theta_old, phi_old, 1, 0, 0)

                for _ in range(n_samples_per_point):
                    r_new = r_old + rng_normal(0, 0.1 * a0)
                    theta_new = theta_old + rng_normal(0, 0.1)
                    phi_new = phi_old + rng_normal(0, 0.1)

                    if r_new > 0 and 0 <= theta_new <= pi and 0 <= phi_new <= 2 * pi:
                        psi_new = trial_wfc(r_new, theta_new, phi_new, 1, 0, 0)
                        wfc_ratio = abs(psi_new / psi_old) ** 2

                        if rng_rand() < wfc_ratio:
                            samples.append([r_new, theta_new, phi_new])
                count += 1
                if (count % 1000) == 0:
                    print(f"[DEBUG] Processed {count} / {total_points} grid points so far.\nTime elapsed: {time.time() - start_time:.2f} seconds")
                    time_lst.append(time.time() - start_time)
                    start_time += time.time() - start_time

            print(f"[DEBUG] Average time elapsed for each 1000 grid points: {np.mean(np.array(time_lst))}\nTotal Time elapsed: {np.sum(np.array(time_lst)):.2f} seconds")

        return np.array(samples)

    def metropolis_3d_sampling_weird(self, grid_points, n_samples_per_point=10, batch_size=1000):
        """Batch-optimized Metropolis sampling."""
        r, theta, phi = grid_points
        total_points = len(r)
        samples = []

        for i in range(0, total_points, batch_size):
            batch_r = r[i:i + batch_size]
            batch_theta = theta[i:i + batch_size]
            batch_phi = phi[i:i + batch_size]

            psi_old = self.trial_wfc(batch_r, batch_theta, batch_phi, 1, 0, 0)

            for _ in range(n_samples_per_point):
                # Propose new points
                batch_r_new = batch_r + np.random.normal(0, 0.1 * self.a0, size=batch_r.shape)
                batch_theta_new = batch_theta + np.random.normal(0, 0.1, size=batch_theta.shape)
                batch_phi_new = batch_phi + np.random.normal(0, 0.1, size=batch_phi.shape)

                # Enforce bounds
                valid_mask = (batch_r_new > 0) & (batch_theta_new >= 0) & (batch_theta_new <= np.pi) & (batch_phi_new >= 0) & (batch_phi_new <= 2 * np.pi)
                batch_r_new = np.where(valid_mask, batch_r_new, batch_r)
                batch_theta_new = np.where(valid_mask, batch_theta_new, batch_theta)
                batch_phi_new = np.where(valid_mask, batch_phi_new, batch_phi)

                # Evaluate wavefunction
                psi_new = self.trial_wfc(batch_r_new, batch_theta_new, batch_phi_new, 1, 0, 0)
                wfc_ratios = np.abs(psi_new / psi_old)**2

                # Metropolis acceptance step
                accept = np.random.rand(len(batch_r)) < wfc_ratios
                batch_r = np.where(accept, batch_r_new, batch_r)
                batch_theta = np.where(accept, batch_theta_new, batch_theta)
                batch_phi = np.where(accept, batch_phi_new, batch_phi)

                # Append accepted samples
                samples.append(np.column_stack((batch_r, batch_theta, batch_phi)))

        return np.vstack(samples)

    def gen_hybrid_samples_old(self):
        """Generate combined grid and Metropolis samples."""
        # Get base grid
        grid_r, grid_theta, grid_phi = self.generate_base_grid()
        grid_points = (grid_r, grid_theta, grid_phi)
        
        # Get Metropolis samples
        metro_samples = self.metropolis_3d_sampling(grid_points)
        
        # Combine datasets
        grid_data = np.column_stack((grid_r, grid_theta, grid_phi))
        X = np.vstack((grid_data, metro_samples))
        
        # Normalize as before
        self.normalization_params = {
            'r_mean': np.mean(X[:, 0]),
            'r_std': np.std(X[:, 0]) + 1e-8,
            'theta_mean': np.mean(X[:, 1]),
            'theta_std': np.std(X[:, 1]) + 1e-8,
            'phi_mean': np.mean(X[:, 2]),
            'phi_std': np.std(X[:, 2]) + 1e-8
        }
        
        X_normalized = (X - np.array([self.normalization_params['r_mean'],
                                    self.normalization_params['theta_mean'],
                                    self.normalization_params['phi_mean']])) / \
                    np.array([self.normalization_params['r_std'],
                            self.normalization_params['theta_std'],
                            self.normalization_params['phi_std']])
        
        return X_normalized

    def gen_hybrid_samples_3d(self):
        """Generate combined grid and Metropolis samples."""
        with tf.device('/CPU:0'):
            # Get base grid
            grid_r, grid_theta, grid_phi = self.generate_base_grid()
            grid_points = (grid_r, grid_theta, grid_phi)
            
            # Get Metropolis samples
            metro_samples = self.metropolis_3d_sampling(grid_points)
            
            # Combine datasets
            grid_data = np.column_stack((grid_r, grid_theta, grid_phi))
            X = np.vstack((grid_data, metro_samples))
            
            # Normalize as before
            self.normalization_params = {
                'r_mean': np.mean(X[:, 0]),
                'r_std': np.std(X[:, 0]) + 1e-8,
                'theta_mean': np.mean(X[:, 1]),
                'theta_std': np.std(X[:, 1]) + 1e-8,
                'phi_mean': np.mean(X[:, 2]),
                'phi_std': np.std(X[:, 2]) + 1e-8
            }
            
            X_normalized = (X - np.array([self.normalization_params['r_mean'],
                                        self.normalization_params['theta_mean'],
                                        self.normalization_params['phi_mean']])) / \
                        np.array([self.normalization_params['r_std'],
                                self.normalization_params['theta_std'],
                                self.normalization_params['phi_std']])
            
            return X_normalized
        
    def gen_hybrid_samples(self):
        """Generate 2D grid and Metropolis samples."""
        grid_r, grid_theta = self.generate_base_grid()
        grid_points = (grid_r, grid_theta)

        # Get Metropolis samples
        metro_samples = self.metropolis_2d_sampling(grid_points)

        # Combine datasets
        grid_data = np.column_stack((grid_r, grid_theta))
        X = np.vstack((grid_data, metro_samples))

        # Normalize the data
        self.normalization_params = {
            'r_mean': np.mean(X[:, 0]),
            'r_std': np.std(X[:, 0]) + 1e-8,
            'theta_mean': np.mean(X[:, 1]),
            'theta_std': np.std(X[:, 1]) + 1e-8
        }
        X_normalized = (X - np.array([self.normalization_params['r_mean'],
                                    self.normalization_params['theta_mean']])) / \
                    np.array([self.normalization_params['r_std'],
                                self.normalization_params['theta_std']])
        return X_normalized
    
    def plot_wavefunction_comparison(self, n_samples=1000, r_max=10.0):
        """Plot histogram of samples vs exact wavefunction."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate r values for exact function
        r_exact = np.linspace(0.01, r_max, 1000)
        theta_exact = np.ones_like(r_exact) * np.pi / 2  # xy-plane
        
        # Calculate exact wavefunction
        r_exact_tf = tf.constant(r_exact, dtype=tf.float64)
        theta_exact_tf = tf.constant(theta_exact, dtype=tf.float64)
        psi_exact = self.trial_wfc(r_exact_tf, theta_exact_tf, n=1)
        prob_density_exact = tf.abs(psi_exact)**2
        
        # Generate samples using Metropolis
        r_samples = np.linspace(0.01, r_max, 100)
        theta_samples = np.ones_like(r_samples) * np.pi / 2
        grid_points = (r_samples, theta_samples)
        metro_samples = self.metropolis_2d_sampling(grid_points, n_samples_per_point=n_samples)
        
        # Extract sampled r values
        sampled_r = metro_samples[:, 0]
        sampled_theta = metro_samples[:, 1]

        # Compute |ψ|² for Metropolis samples
        sampled_psi = self.trial_wfc(
            tf.constant(sampled_r, dtype=tf.float64),
            tf.constant(sampled_theta, dtype=tf.float64),
            n=1
        )
        sampled_prob_density = tf.abs(sampled_psi)**2

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(sampled_r, bins=50, density=True, alpha=0.5, label='Metropolis Samples')
        plt.plot(r_exact, prob_density_exact, 'r-', label='Exact')
        plt.xlabel('r (Bohr radii)')
        plt.ylabel('|ψ|²')
        plt.title('Hydrogen Ground State Radial Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()