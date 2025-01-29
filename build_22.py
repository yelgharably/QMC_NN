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

    def metropolis_sampling(self, n_samples, burn_in, start=None):
        if start is None:
            current_pos = np.random.uniform(-1, 1, self.dim)
        wf_current = self.wavefunction(current_pos)
        samples = []
        accepted_steps = 0
        total_steps = burn_in + n_samples

        with tqdm(total=total_steps, desc='Metropolis Sampling') as pbar:
            for i in range(total_steps):
                # Propose new position
                proposal = current_pos + np.random.normal(0, self.step_size, self.dim)
                wf_proposal = self.wavefunction(proposal)
                
                # Compute acceptance ratio and clamp to 1
                acceptance_ratio = min(1, (wf_proposal**2) / (wf_current**2))
                
                # Decide whether to accept the proposal
                if np.random.rand() < acceptance_ratio:
                    current_pos = proposal
                    wf_current = wf_proposal
                    if i >= burn_in:
                        accepted_steps += 1
                
                if i >= burn_in:
                    samples.append(current_pos)

                pbar.update(1)
        
        samples = np.array(samples)
        self.export_samples(samples)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rates_path = os.path.join(self.output_dir, f'acceptance_rates_{timestamp}.txt')
        with open(rates_path, 'w') as f:
            acceptance_rate = accepted_steps / n_samples
            f.write(f"Acceptance rate: {acceptance_rate:.2f}")
        acceptance_rate = accepted_steps / n_samples
        print(f"Acceptance rate: {acceptance_rate:.2f}")
        
        return samples
    
    
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
    
    def exact_den(r):
        return 16 * np.pi * r**2 * np.exp(-2 * r)
    
    vmc = QMC_gen_v2(step_size=0.5, wavefunction=hydrogen_wf_wrapper)

    n_samples = int(sys.argv[1])
    samples = vmc.metropolis_sampling(n_samples=n_samples, burn_in=n_samples//5)

    # Extract radial distances from samples
    r_samples = np.linalg.norm(samples, axis=1)

    # Generate data for the exact radial wavefunction
    r_exact = np.linspace(0, 10, 1000)
    psi_exact = hwf.normalized_wfc(r_exact, np.pi/2, 0)
    p_exact = exact_den(r_exact)
    p_exact /= np.trapz(p_exact, r_exact)

    # Normalize histogram and exact probability
    hist, bins = np.histogram(r_samples, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_area = np.trapz(hist, bin_centers)
    print(f"Histogram area: {hist_area:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.hist(r_samples, bins=50, density=True, alpha=0.6, label='Sampled Distribution')
    #plt.plot(r_exact, p_exact, 'r-', linewidth=2, label='Exact Distribution')
    plt.plot(r_exact,p_exact,'g-',linewidth=2,label='Exact Density')
    plt.xlabel('Radial Distance r')
    plt.ylabel('Radial Probability Density')
    plt.title(f'Comparison of Sampled Distribution and Exact Hydrogen 1s Wavefunction, Samples = {n_samples}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()