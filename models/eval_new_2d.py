import numpy as np
import matplotlib.pyplot as plt
from .base_network_6 import WfcNN  # Updated model import
from ..training.trainer_7 import radial_wavefunction, angular_wavefunction, psi_func
import torch
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import sys
from ..qmc.qmc_generator_5 import QMC_gen_v4
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"D:/Undergraduate Life/Honors/wfc_plots_{timestamp}"
model_num = str(sys.argv[1])
import time

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def limits(n):
    if n == 1:
        return (-5, 5, 800)
    elif n == 2:
        return (-10, 10, 800)
    elif n == 3:
        return (-15, 15, 800)
    elif n == 4:
        return (-20, 20, 800)

# Theoretical wavefunction squared
def psi_squared(x, y, n, l, m):
    r = np.sqrt(x**2 + y**2)
    theta = np.full_like(r,np.pi/2)  # Safely compute theta
    phi = np.arctan2(y, x)
    R = radial_wavefunction(n, l, r)
    Y_real, Y_imag = angular_wavefunction(l, m, phi)
    return (R * Y_real)**2 + (R * Y_imag)**2

# Neural network prediction
def neural_net_psi_squared(x, y, n, l, m, model):
    # Flatten all inputs to 1D arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    n_flat = np.full_like(x_flat, n)
    l_flat = np.full_like(x_flat, l)
    m_flat = np.full_like(x_flat, m)

    # Convert inputs to torch tensors
    inputs = torch.tensor(np.stack([x_flat, y_flat], axis=1), dtype=torch.float32)
    n_tensor = torch.tensor(n_flat, dtype=torch.long)
    l_tensor = torch.tensor(l_flat, dtype=torch.long)
    m_tensor = torch.tensor(m_flat, dtype=torch.long)

    with torch.no_grad():
        # Pass inputs to the model
        predictions = model(inputs, n_tensor, l_tensor, m_tensor)

        # Split into real and imaginary parts
        psi_real = predictions[:, 0].numpy()
        psi_imag = predictions[:, 1].numpy()

    # Compute |\psi|^2 and reshape to grid shape
    psi_squared = psi_real**2 + psi_imag**2
    return psi_squared.reshape(x.shape)

# Load the trained model
model = WfcNN(n_max=4, l_max=3, m_max=3, activation="Tanh")
model_path = f"model_{model_num}_tanh.pth"  # Replace with the path to your .pth file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def two_d_plot():
    start_time = time.time()
    time_list = []
    for n in range(1, 5):
        x = np.linspace(-5*n, 5*n, 400)
        y = np.linspace(-5*n,5*n,400)  # Set z to zero for 2D evaluation
        X, Y = np.meshgrid(x, y)

        for l in range(n):
            for m in range(-l, l + 1):
                # Compute theoretical wavefunction (replace with real logic)
                theoretical_psi2 = psi_squared(X, Y, n, l, m)
                plot_start_time = time.time()
                # Compute neural network wavefunction (replace with real logic)
                neural_psi2 = neural_net_psi_squared(X,Y,n, l, m, model)
                residuals = np.abs(neural_psi2 - theoretical_psi2)

                plt.figure(figsize=(21, 6))
                plt.suptitle(f"Comparison of Wavefunction Distributions ($|\psi({n}{l}{m})|^2$)")

                # Theoretical wavefunction
                plt.subplot(1, 3, 1)
                plt.contourf(X, Y, theoretical_psi2, levels=50, cmap="viridis")
                plt.colorbar(label="$|\psi|^2$")
                plt.title("Theoretical $|\psi|^2$")
                plt.xlabel("$x$")
                plt.ylabel("$y$")

                # Neural network wavefunction
                plt.subplot(1, 3, 2)
                plt.contourf(X, Y, neural_psi2, levels=50, cmap="viridis")
                plt.colorbar(label="$|\psi|^2$")
                plt.title("Neural Network $|\psi|^2$")
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                
                plt.subplot(1, 3, 3)
                plt.contourf(X, Y, residuals, levels=50, cmap="inferno")
                plt.colorbar(label="Residuals $|NN - Theoretical|$")
                plt.title("Residuals $|\psi_{NN}^2 - \psi_{Theoretical}^2|$")
                plt.xlabel("$x$")
                plt.ylabel("$y$")

                plt.tight_layout()
                filename = f"wfc_{n}{l}{m}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()
                time_list.append(time.time() - plot_start_time)
                print(f"Plot ({n}{l}{m}) took {time.time() - plot_start_time:.2f} seconds.")

def three_d_plot():
    for n in range(1, 5):
        x = np.linspace(-5 * n, 5 * n, 100)  # Reduce resolution for 3D plots
        y = np.linspace(-5 * n, 5 * n, 100)
        z = np.linspace(-5 * n, 5 * n, 100)
        X, Y, Z = np.meshgrid(x, y, z)  # Create a 3D grid

        for l in range(n):
            for m in range(-l, l + 1):
                # Compute theoretical wavefunction (replace with real logic)
                theoretical_psi2 = psi_squared(X, Y, Z, n, l, m)
                # Compute neural network wavefunction (replace with real logic)
                neural_psi2 = neural_net_psi_squared(X, Y, Z, n, l, m, model)
                residuals = np.abs(neural_psi2 - theoretical_psi2)

                fig = plt.figure(figsize=(16, 6))
                fig.suptitle(f"3D Comparison of Wavefunction Distributions ($|\psi({n}{l}{m})|^2$)")

                # Theoretical wavefunction
                ax1 = fig.add_subplot(131, projection="3d")
                ax1.contour3D(X[:, :, X.shape[2] // 2], Y[:, :, Y.shape[2] // 2], theoretical_psi2[:, :, X.shape[2] // 2], 50, cmap="viridis")
                ax1.set_title("Theoretical $|\psi|^2$")
                ax1.set_xlabel("$x$")
                ax1.set_ylabel("$y$")
                ax1.set_zlabel("$|\psi|^2$")

                # Neural network wavefunction
                ax2 = fig.add_subplot(132, projection="3d")
                ax2.contour3D(X[:, :, X.shape[2] // 2], Y[:, :, Y.shape[2] // 2], neural_psi2[:, :, X.shape[2] // 2], 50, cmap="viridis")
                ax2.set_title("Neural Network $|\psi|^2$")
                ax2.set_xlabel("$x$")
                ax2.set_ylabel("$y$")
                ax2.set_zlabel("$|\psi|^2$")

                # Residuals plot
                ax3 = fig.add_subplot(133, projection="3d")
                ax3.contour3D(X[:, :, X.shape[2] // 2], Y[:, :, Y.shape[2] // 2], residuals[:, :, X.shape[2] // 2], 50, cmap="inferno")
                ax3.set_title("Residuals $|\psi_{NN}^2 - \psi_{Theoretical}^2|$")
                ax3.set_xlabel("$x$")
                ax3.set_ylabel("$y$")
                ax3.set_zlabel("Residuals")

                # Save the figure
                filename = f"wfc_3d_{n}{l}{m}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()

def combined_plots():
    """
    Creates a 6-subplot figure for each (n, l, m):
    - Top row (3 subplots) shows the 2D plots (theoretical, NN, residuals).
    - Bottom row (3 subplots) shows the 3D plots (theoretical, NN, residuals).
    """
    start_time = time.time()
    time_list = []
    for n in range(1, 5):
        # Prepare 2D plot inputs
        x_2d = np.linspace(-5 * n, 5 * n, 100)
        y_2d = np.linspace(-5 * n, 5 * n, 100)
        X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
        Z_2d = np.zeros_like(X_2d)

        # Prepare 3D plot inputs (lower resolution)
        x_3d = np.linspace(-5 * n, 5 * n, 100)
        y_3d = np.linspace(-5 * n, 5 * n, 100)
        z_3d = np.linspace(-5 * n, 5 * n, 100)
        X_3d, Y_3d, Z_3d = np.meshgrid(x_3d, y_3d, z_3d)

        for l in range(n):
            for m in range(-l, l + 1):
                plot_start_time = time.time()
                # -- 2D computations --
                theoretical_2d = psi_squared(X_2d, Y_2d, Z_2d, n, l, m)
                neural_2d = neural_net_psi_squared(X_2d, Y_2d, Z_2d, n, l, m, model)
                residuals_2d = np.abs(neural_2d - theoretical_2d)

                # -- 3D computations --
                theoretical_3d = psi_squared(X_3d, Y_3d, Z_3d, n, l, m)
                neural_3d = neural_net_psi_squared(X_3d, Y_3d, Z_3d, n, l, m, model)
                residuals_3d = np.abs(neural_3d - theoretical_3d)

                # Create figure with 6 subplots
                fig = plt.figure(figsize=(21, 12))
                fig.suptitle(f"Combined 2D/3D Wavefunction (n={n}, l={l}, m={m})")

                # ========== Top row (2D) ==========
                # 2D Theoretical
                ax_2d_theo = fig.add_subplot(2, 3, 1)
                cont1 = ax_2d_theo.contourf(X_2d, Y_2d, theoretical_2d, levels=50, cmap="viridis")
                plt.colorbar(cont1, ax=ax_2d_theo, label="$|\psi|^2$")
                ax_2d_theo.set_title("2D Theoretical")
                ax_2d_theo.set_xlabel("x")
                ax_2d_theo.set_ylabel("y")

                # 2D NN
                ax_2d_nn = fig.add_subplot(2, 3, 2)
                cont2 = ax_2d_nn.contourf(X_2d, Y_2d, neural_2d, levels=50, cmap="viridis")
                plt.colorbar(cont2, ax=ax_2d_nn, label="$|\psi|^2$")
                ax_2d_nn.set_title("2D Neural Network")
                ax_2d_nn.set_xlabel("x")
                ax_2d_nn.set_ylabel("y")

                # 2D Residuals
                ax_2d_res = fig.add_subplot(2, 3, 3)
                cont3 = ax_2d_res.contourf(X_2d, Y_2d, residuals_2d, levels=50, cmap="inferno")
                plt.colorbar(cont3, ax=ax_2d_res, label="Residuals")
                ax_2d_res.set_title("2D Residuals")
                ax_2d_res.set_xlabel("x")
                ax_2d_res.set_ylabel("y")

                # ========== Bottom row (3D) ==========
                # 3D Theoretical
                ax_3d_theo = fig.add_subplot(2, 3, 4, projection="3d")
                # Slice at center for viewing
                mid_idx = X_3d.shape[2] // 2
                ax_3d_theo.contour3D(
                    X_3d[:, :, mid_idx],
                    Y_3d[:, :, mid_idx],
                    theoretical_3d[:, :, mid_idx],
                    50,
                    cmap="viridis"
                )
                ax_3d_theo.set_title("3D Theoretical")
                ax_3d_theo.set_xlabel("x")
                ax_3d_theo.set_ylabel("y")
                ax_3d_theo.set_zlabel("$|\psi|^2$")

                # 3D NN
                ax_3d_nn = fig.add_subplot(2, 3, 5, projection="3d")
                ax_3d_nn.contour3D(
                    X_3d[:, :, mid_idx],
                    Y_3d[:, :, mid_idx],
                    neural_3d[:, :, mid_idx],
                    50,
                    cmap="viridis"
                )
                ax_3d_nn.set_title("3D Neural Network")
                ax_3d_nn.set_xlabel("x")
                ax_3d_nn.set_ylabel("y")
                ax_3d_nn.set_zlabel("$|\psi|^2$")

                # 3D Residuals
                ax_3d_res = fig.add_subplot(2, 3, 6, projection="3d")
                ax_3d_res.contour3D(
                    X_3d[:, :, mid_idx],
                    Y_3d[:, :, mid_idx],
                    residuals_3d[:, :, mid_idx],
                    50,
                    cmap="inferno"
                )
                ax_3d_res.set_title("3D Residuals")
                ax_3d_res.set_xlabel("x")
                ax_3d_res.set_ylabel("y")
                ax_3d_res.set_zlabel("Residuals")

                plt.tight_layout()

                # Save figure
                filename = f"combined_{n}_{l}_{m}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()
                time_list.append(time.time() - plot_start_time)
                print(f"Plot ({n}{l}{m}) took {time.time() - plot_start_time:.2f} seconds.")

    print(f"Average time: {sum(time_list) / len(time_list):.2f} seconds.")
    print(f"Combined plots took {time.time() - start_time:.2f} seconds.")
# if int(sys.argv[1]) == 2:
#     two_d_plot()

# elif int(sys.argv[1]) == 3:
#     three_d_plot()

# elif int(sys.argv[1]) == 4:
#     combined_plots()

if __name__ == "__main__":
    two_d_plot()