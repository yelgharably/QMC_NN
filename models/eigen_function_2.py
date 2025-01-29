import numpy as np
import json
from datetime import datetime

def laplacian_and_potential(X, Y, params):
    """
    Compute the Laplacian and potential for the wavefunction psi.
    """
    N, M = 3, 3
    beta = params["beta"]

    # Initialize psi
    psi = np.zeros_like(X)
    for n in range(N + 1):
        for m in range(M + 1):
            a_nm = params.get(f"a_nm_{n}_{m}", 0)
            b_nm = params.get(f"b_nm_{n}_{m}", 0)
            c_nm = params.get(f"c_nm_{n}_{m}", 0)
            k_nm = params.get(f"k_nm_{n}_{m}", 0)

            r = np.sqrt(X**2 + Y**2)
            psi += a_nm * (X**n) * (Y**m)
            psi += b_nm * (X**n) * (Y**m) * np.cosh(k_nm * r)
            psi += c_nm * (X**n) * (Y**m) * np.sinh(k_nm * r)

    # Add damping factor
    psi *= np.exp(-beta * (X**2 + Y**2))

    return psi

def compute_expectation_values(X, Y, psi, dx, dy):
    """
    Compute the kinetic, potential, and total energy from psi.
    """
    # Compute Laplacian using finite differences
    laplacian = (
        (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2 +
        (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2
    )

    # Compute potential
    r2 = X[1:-1, 1:-1]**2 + Y[1:-1, 1:-1]**2
    with np.errstate(divide='ignore', invalid='ignore'):
        V = -1 / r2
        V[np.isinf(V)] = 0  # Handle singularities

    # Compute Hamiltonian terms
    kinetic = -0.5 * laplacian
    potential = V * psi[1:-1, 1:-1]
    hamiltonian = kinetic + potential

    # Compute expectation values
    psi_sq = psi[1:-1, 1:-1]**2
    norm = np.sum(psi_sq) * dx * dy
    E_kin = np.sum(kinetic * psi[1:-1, 1:-1]) * dx * dy / norm
    E_pot = np.sum(potential * psi[1:-1, 1:-1]) * dx * dy / norm
    E_tot = np.sum(hamiltonian * psi[1:-1, 1:-1]) * dx * dy / norm

    return E_kin, E_pot, E_tot

def main():
    # Grid parameters
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    n_points = 10000
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    dx = (x_max - x_min) / (n_points - 1)
    dy = (y_max - y_min) / (n_points - 1)
    X, Y = np.meshgrid(x, y)

    # Load parameters
    with open("fitted_function_params_20250106_184629.json", "r") as file:
        params = json.load(file)

    # Compute psi
    psi = laplacian_and_potential(X, Y, params)

    # Compute energies
    E_kin, E_pot, E_tot = compute_expectation_values(X, Y, psi, dx, dy)

    print(f"Kinetic Energy: {E_kin:.6f}")
    print(f"Potential Energy: {E_pot:.6f}")
    print(f"Total Energy: {E_tot:.6f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"energy_values_{timestamp}.txt", "w") as file:
        file.write(f"Kinetic Energy: {E_kin:.6f}\n")
        file.write(f"Potential Energy: {E_pot:.6f}\n")
        file.write(f"Total Energy: {E_tot:.6f}\n")

if __name__ == "__main__":
    main()
