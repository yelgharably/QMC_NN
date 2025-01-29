from .trainer_2 import radial_wavefunction, angular_wavefunction, psi_squared
import matplotlib.pyplot as plt
import numpy as np

def plot_wavefunction_cartesian(x_grid, y_grid, wavefunction_grid, title):
    """
    Plots the wavefunction in Cartesian coordinates using pcolormesh.

    Parameters:
    - x_grid: 2D np.ndarray, x-coordinates.
    - y_grid: 2D np.ndarray, y-coordinates.
    - wavefunction_grid: 2D np.ndarray, wavefunction values.
    - title: str, title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, y_grid, wavefunction_grid, shading='auto', cmap='viridis')
    plt.colorbar(label="Wavefunction Amplitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()

def main():
    n = 1
    l = 0
    m = 0
    r = np.linspace(0, 3, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    r_grid, theta_grid = np.meshgrid(r, theta)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    psi2 = psi_squared(r_grid, theta_grid, n, l)
    psi_grid = psi2.reshape(r_grid.shape)

    plot_wavefunction_cartesian(x_grid, y_grid, psi_grid, "Hydrogen 1s Orbital")

if __name__ == "__main__":
    main()