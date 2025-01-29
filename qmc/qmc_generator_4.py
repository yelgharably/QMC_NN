import numpy as np
from ..config import CONFIG
from scipy.special import sph_harm, eval_genlaguerre
import tensorflow as tf
import scipy
from multiprocessing import Pool
from tqdm import tqdm
import time
import os
from datetime import datetime

sample_n = CONFIG['training']['sample_size']
grid_n = CONFIG['training']['grid_size']
a0 = 1.0

class QMC_gen_v4():
    def __init__(self, step_size, wavefunction, n_samples=10000, burn_in=5000, a0=1.0, dim=2):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.a0 = a0
        self.dim = dim
        self.step_size = step_size
        self.wavefunction = wavefunction
        self.output_dir = 'QMC_gen_data'

    def export_samples(self, samples, metadata=None):
        """Export samples to CSV file with timestamp."""
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
        np.savetxt(filepath, samples, delimiter=',', header='x,y,z', comments='')

        return filepath

    def metropolis_sampling(self, n_samples, burn_in, start=None):
        if start is None:
            # Initialize random starting position in 2D (x, y)
            current_pos = np.random.uniform(-1, 1, self.dim)
        else:
            current_pos = np.array(start)

        # Compute the wavefunction magnitude squared at the initial position
        psi_real, psi_imag = self.wavefunction(current_pos)
        wf_current_mag_sq = psi_real**2 + psi_imag**2

        samples = []
        accepted_steps = 0
        total_steps = burn_in + n_samples

        with tqdm(total=total_steps, desc='Metropolis Sampling') as pbar:
            for i in range(total_steps):
                # Propose new position in 2D
                proposal = current_pos + np.random.normal(0, self.step_size, self.dim)

                # Compute the wavefunction magnitude squared at the proposed position
                psi_real_prop, psi_imag_prop = self.wavefunction(proposal)
                wf_proposal_mag_sq = psi_real_prop**2 + psi_imag_prop**2

                # Compute acceptance ratio and clamp to 1
                acceptance_ratio = min(1, wf_proposal_mag_sq / wf_current_mag_sq)

                # Decide whether to accept the proposal
                if np.random.rand() < acceptance_ratio:
                    current_pos = proposal
                    wf_current_mag_sq = wf_proposal_mag_sq
                    if i >= burn_in:
                        accepted_steps += 1

                if i >= burn_in:
                    samples.append(current_pos)

                pbar.update(1)

        samples = np.array(samples)
        self.export_samples(samples)

        # Log acceptance rate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rates_path = os.path.join(self.output_dir, f'acceptance_rates_{timestamp}.txt')
        with open(rates_path, 'w') as f:
            acceptance_rate = accepted_steps / n_samples
            f.write(f"Acceptance rate: {acceptance_rate:.2f}")
        print(f"Acceptance rate: {acceptance_rate:.2f}")

        return samples
