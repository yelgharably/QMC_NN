"""
Ditched for now, will come back to it later
"""

import numpy as np
from ..config import CONFIG
from ..physics.operators import kinetic_operator, potential_operator

class QuantumSolver:
    """"
    A class for simulating the quantum mechanical nature of the system
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.r_infinity = config['physics']['r_infinity']
        self.delta_r = config['physics']['delta_r']
        self.n_range = config['physics']['quantum_numbers']['n_range']
        self.l_range = config['physics']['quantum_numbers']['l_range']
        self.m_range = config['physics']['quantum_numbers']['m_range']
    
    def hamiltonian(self, wavefunction):
        """
        Compute the Hamiltonian operator
        """
        kinetic = kinetic_operator(wavefunction)
        potential = potential_operator(wavefunction)
        return kinetic + potential

    # def eigenstates WIP