import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.special as sp
import winsound
from scipy.integrate import nquad
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from .math_nn import factorial, eval_genlaguerre, spherical_harmonic, radial_part
import sys
from datetime import datetime

class QMC_gen_v2():
    def __init__(self,step_size, wavefunction, n_samples=10000, burn_in=5000, a0=1.0, dim=2):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.a0 = a0
        self.dim = dim
        self.step_size = step_size
        self.wavefunction = wavefunction
        self.output_dir = 'QMC_gen_data'

    def export_samples(self, samples, metadata=None):
        """Export samples to CSV file with timestamp."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vmc_samples_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save metadata
        with open(filepath.replace('.csv', '_metadata.txt'), 'w') as f:
            f.write(f"Samples: {self.n_samples}\n")
            f.write(f"Burn-in: {self.burn_in}\n")
            f.write(f"Step size: {self.step_size}\n")
            f.write(f"Dimension: {self.dim}\n")
            if metadata:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        # Save samples
        np.savetxt(filepath, samples, delimiter=',', 
                  header='x,y', comments='')
        
        return filepath

    def metropolis_sampling(self,n_samples,burn_in,start=None):
        if start is None:
            current_pos = np.random.uniform(0,4,self.dim)
        else:
            start = start
        samples = []
        acceptance_rates = []
        count = 0
        total_steps = burn_in + n_samples
        with tqdm(total=total_steps, desc='Metropolis Sampling') as pbar:
            for i in range(total_steps):
                proposed = current_pos + np.random.uniform(-self.step_size,self.step_size,self.dim)
                p_current = self.wavefunction(current_pos)**2
                p_proposed = self.wavefunction(proposed)**2
                acceptance = p_proposed/p_current
                if (i+1) % 1000 == 0:
                    acceptance_rates.append(np.round(np.abs(acceptance), 3))
                    count += 1

                if np.random.rand() < acceptance:
                    current_pos = proposed
                
                if i >= burn_in:
                    samples.append(current_pos)
                
                pbar.update(1)
        
        samples = np.array(samples)
        self.export_samples(samples)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rates_path = os.path.join(self.output_dir, f'acceptance_rates_{timestamp}.txt')
        np.savetxt(rates_path, np.array(acceptance_rates), fmt='%.3f')
        
        return samples
    

class HydrogenWfc_v1():
    def __init__(self,r,theta,n,l,m=0,phi=0):
        self.a0 = 1.0
        self.r = r
        self.theta = theta
        self.n = n
        self.l = l
        self.m = m
        self.phi = phi
    
    def Lagguere(self,n,alpha,x):
        return sp.eval_genlaguerre(n,alpha,x) # n =n-l-1, alpha = 2*l+1, x = 2*r/(n*a0)
    
    def Rnl_unnormalized(self,r,n,l):
        lag_1 = n - l - 1
        lag_2 = 2 * l + 1
        lag_3 = 2 * r / (n * self.a0)
        return r**l * np.exp(-r / (n * self.a0)) * self.Lagguere(lag_1, lag_2, lag_3)
    
    def Ylm(self,theta,phi,l,m):
        return sp.sph_harm(m,l,theta,phi)
    
    def unnormalized_wfc(self,r,theta,phi,n,l,m):
        return self.Rnl_unnormalized(r,n,l) * self.Ylm(theta,phi,l,m)
    
    def normalize_wfc(self):
        """Calculate normalization constant for wavefunction."""
        def integrand(r, theta, phi):
            wfc = self.unnormalized_wfc(r, theta, phi, self.n, self.l, self.m)
            return np.abs(wfc)**2 * r**2 * np.sin(theta)
        
        # Integration bounds
        r_bounds = [0, np.inf]
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2*np.pi]
        
        # Compute integral
        result, error = nquad(
            integrand,
            ranges=[r_bounds, theta_bounds, phi_bounds],
            opts={'limit': 100, 'epsabs': 1e-6}
        )
        
        # Normalization constant
        N = np.abs(1/np.sqrt(result))
        
        return N
    
    def normalized_wfc(self, r, theta, phi):
        """Return normalized wavefunction."""
        N = self.normalize_wfc()
        return N * self.unnormalized_wfc(r, theta, phi, self.n, self.l, self.m)
    
class HydrogenWfc_v3():
    def __init__(self,r,theta,n,l,m=0,phi=0):
        self.a0 = 1.0
        self.r = r
        self.theta = theta
        self.n = n
        self.l = l
        self.m = m
        self.phi = phi
    
    def Lagguere(self,n,alpha,x):
        return eval_genlaguerre(n,alpha,x) # n =n-l-1, alpha = 2*l+1, x = 2*r/(n*a0)
    
    def Rnl_unnormalized(self,r,n,l):
        lag_1 = n - l - 1
        lag_2 = 2 * l + 1
        lag_3 = 2 * r / (n * self.a0)
        return r**l * np.exp(-r / (n * self.a0)) * self.Lagguere(lag_1, lag_2, lag_3)
    
    def Ylm(self,theta,phi,l,m):
        return spherical_harmonic(theta,l)
    
    def unnormalized_wfc(self,r,theta,phi,n,l,m):
        return self.Rnl_unnormalized(r,n,l) * self.Ylm(theta,phi,l,m)
    
    def normalize_wfc(self):
        """Calculate normalization constant for wavefunction."""
        def integrand(r, theta, phi):
            wfc = self.unnormalized_wfc(r, theta, phi, self.n, self.l, self.m)
            return np.abs(wfc)**2 * r**2 * np.sin(theta)
        
        # Integration bounds
        r_bounds = [0, np.inf]
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2*np.pi]
        
        # Compute integral
        result, error = nquad(
            integrand,
            ranges=[r_bounds, theta_bounds, phi_bounds],
            opts={'limit': 100, 'epsabs': 1e-6}
        )
        
        # Normalization constant
        N = np.abs(1/np.sqrt(result))
        
        return N
    
    def normalized_wfc(self, r, theta, phi):
        """Return normalized wavefunction."""
        N = self.normalize_wfc()
        return self.unnormalized_wfc(r, theta, phi, self.n, self.l, self.m)
    
class HydrogenWfc():
    def __init__(self,r,theta,n,l,m=0,phi=0):
        self.a0 = 1.0
        self.r = r
        self.theta = theta
        self.n = n
        self.l = l
        self.m = m
        self.phi = phi
    
    def Rnl(self,r,n,l):
        return radial_part(r,n,l) # n =n-l-1, alpha = 2*l+1, x = 2*r/(n*a0)
    
    def Ylm(self,theta,phi,l):
        return sp.sph_harm(self.m,l,phi,theta)
    
    def normalized_wfc(self, r, theta, phi):
        """Return normalized wavefunction."""
        wfc = self.Rnl(r,self.n,self.l) * self.Ylm(theta,phi,self.l)
        return wfc


def main():
    hwf = HydrogenWfc(r=None, theta=None, phi=None, n=1, l=0, m=0)

    # Create wrapper function to match QMC generator's expected input
    def hydrogen_wf_wrapper(pos):
        x, y = pos[0], pos[1]
        r = np.sqrt(x**2 + y**2)  # radial distance in xy-plane
        theta = np.arctan2(np.abs(y), np.abs(x))  # angle from x-axis [0,π/2]
        phi = 0.0  # fixed in xy-plane
        return hwf.normalized_wfc(r, theta, phi)

    def hydrogen_1s(r_vec):
        """Hydrogen 1s wavefunction. 
        r_vec: Position vector (numpy array) in 3D space."""
        r = np.linalg.norm(r_vec)  # Compute the norm (distance from origin)
        return 4*np.exp(-r)*(1-2*r)*np.sqrt(np.pi/7)
    
    def exact_den(r):
        return (np.exp(-r))**2 * (4*r**2)

    vmc = QMC_gen_v2(step_size=0.05, wavefunction=hydrogen_wf_wrapper)

    n_samples = int(sys.argv[1])
    samples = vmc.metropolis_sampling(n_samples=n_samples, burn_in=n_samples//5)

    # Extract radial distances from samples
    r_samples = np.linalg.norm(samples, axis=1)

    # Generate data for the exact radial wavefunction
    r_exact = np.linspace(0.1, 10, 1000)
    psi_exact = hwf.normalized_wfc(r_exact, np.pi/2, 0)
    p_exact = (psi_exact**2) * (4 * r_exact**2)  # Radial probability density
    p_den = exact_den(r_exact)

    # Normalize histogram and exact probability
    hist, bins = np.histogram(r_samples, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_area = np.trapz(hist, bin_centers)
    print(f"Histogram area: {hist_area:.2f}")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.hist(r_samples, bins=50, density=True, alpha=0.6, label='Sampled Distribution')
    plt.plot(r_exact, p_exact, 'r-', linewidth=2, label='Exact Distribution')
    plt.plot(r_exact,p_den,'g-',linewidth=2,label='Exact Density')
    plt.xlabel('Radial Distance r')
    plt.ylabel('Radial Probability Density')
    plt.title(f'Comparison of Sampled Distribution and Exact Hydrogen 1s Wavefunction, Samples = {n_samples}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()