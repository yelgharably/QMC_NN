from numba import njit
import numpy as np
import json
from scipy.sparse import diags
from datetime import datetime
from scipy.sparse.linalg import eigs

# Define numerical Laplacian and potential

def laplacian_and_potential(x, y, params, dx, dy):
    N, M = 3, 3
    beta = params["beta"]

    # Generate 2D grid
    X, Y = np.meshgrid(x, y)

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

    # Compute Laplacian
    laplacian = (
        (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2 +
        (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2
    )

    # Potential term
    with np.errstate(divide='ignore', invalid='ignore'):
        V = -1 / (X[1:-1, 1:-1]**2 + Y[1:-1, 1:-1]**2) * psi[1:-1, 1:-1]
        V[np.isinf(V)] = 0  # Handle singularities

    # Hamiltonian H = -Laplacian + V
    H = -laplacian + V
    return H

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from numpy.linalg import eig

def compute_eigenvalues(H, num_eigenvalues=5):
    """
    Compute the eigenvalues of a matrix H.
    
    Parameters:
    - H: The Hamiltonian matrix (must be square).
    - num_eigenvalues: Number of eigenvalues to compute (only for sparse solver).

    Returns:
    - eigenvalues: Computed eigenvalues of H.
    """
    # Validate the input matrix
    if not (H.shape[0] == H.shape[1]):
        raise ValueError("Matrix H must be square.")

    # Debug: Check for NaN or Inf values
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Matrix contains NaN or infinite values.")

    try:
        # Check if the matrix is sparse
        is_sparse = isinstance(H, csr_matrix)
        if not is_sparse:
            # Convert to sparse if H is large (optional)
            sparsity_threshold = 0.5  # Adjust based on problem size
            sparsity = np.count_nonzero(H) / H.size
            if sparsity < sparsity_threshold:
                H = csr_matrix(H)
                is_sparse = True

        # Compute eigenvalues
        if is_sparse:
            print("Using sparse eigenvalue solver.")
            eigenvalues, _ = eigs(H, k=min(num_eigenvalues, H.shape[0] - 1), which='SM', maxiter=100000)
        else:
            print("Using dense eigenvalue solver.")
            eigenvalues, _ = eig(H)

        return np.real(eigenvalues)

    except Exception as e:
        raise RuntimeError(f"Eigenvalue computation failed: {e}")
# Main
def main():
    # Grid parameters
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    n_points = 100
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    dx = (x_max - x_min) / (n_points - 1)
    dy = (y_max - y_min) / (n_points - 1)
    X, Y = np.meshgrid(x, y)
    
    # Load parameters
    with open("fitted_function_params_20250106_184629.json", "r") as file:
        params = json.load(file)

    # Compute Hamiltonian
    H = laplacian_and_potential(X, Y, params, dx, dy)
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Matrix contains NaN or infinite values.")

    # Compute eigenvalues
    eigenvalues = compute_eigenvalues(H)
    eigenvalues = np.real(np.array(eigenvalues))
    np.sort(eigenvalues)
    ground_state_energy = eigenvalues[0]
    print("Eigenvalues:", eigenvalues)
    print("Ground State Energy:", ground_state_energy)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"eigenvalues_{timestamp}.txt","w") as f:
        for i in range(len(eigenvalues)):
            f.write(f"Eigenvalues: {eigenvalues[i]:.3f}\n")
        f.write(f"Ground State Energy:{ground_state_energy:.3f}")


if __name__ == "__main__":
    main()
