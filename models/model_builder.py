from .student_model import StudentWavefunctionNN
from .base_network_2 import WavefunctionNN
from ..training.trainer_3 import load_model, model_name
import torch
import numpy as np
from sympy import symbols, tanh, Matrix, sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from .student_model import student_model_name
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime

def target_function(inputs, *params):
    x, y = inputs
    N = 2  # Degree of power series
    M = 2  # Number of Fourier terms

    psi = 0
    index = 0
    for n in range(N + 1):
        for m in range(M + 1):
            a_nm = params[index]
            b_nm = params[index + 1]
            c_nm = params[index + 2]
            k_m = params[index + 3]

            # Symmetric power terms
            psi += a_nm * (x ** n) * (y ** m)

            # Add symmetric hyperbolic terms
            psi += b_nm * (x ** n) * (y ** m) * np.cosh(k_m * np.sqrt(x**2 + y**2))
            psi += c_nm * (x ** n) * (y ** m) * np.sinh(k_m * np.sqrt(x**2 + y**2))

            index += 4

    # Add exponential damping to create a peak at the origin
    damping = params[-1]  # Last parameter controls damping
    psi *= np.exp(-damping * (x**2 + y**2))

    return psi

def target_function_fourier_only(inputs, *params):
    x, y = inputs
    M = 2  # Number of Fourier terms

    psi = 0
    index = 0
    for m in range(M + 1):
        b_m = params[index]
        c_m = params[index + 1]
        k_m = params[index + 2]

        # Add symmetric and asymmetric hyperbolic terms
        psi += b_m * np.cosh(k_m * np.sqrt(x**2 + y**2))
        psi += c_m * np.sinh(k_m * np.sqrt(x**2 + y**2))

        index += 3

    # Add exponential damping to create a peak at the origin
    damping = params[-1]  # Last parameter controls damping
    psi *= np.exp(-damping * (x**2 + y**2))

    return psi

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


x_data = np.linspace(-3, 3, 100)
y_data = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_data, y_data)
n = 1
l = 0
x_flat = X.flatten()
y_flat = Y.flatten()

inputs = torch.tensor(np.column_stack([x_flat, y_flat, np.full_like(x_flat, n), np.full_like(y_flat, l)]), dtype=torch.float32).to('cpu')

student_model = StudentWavefunctionNN()

try:
    load_model(student_model,student_model_name)
except FileNotFoundError:
    print("No saved Model found. Starting from scratch")

with torch.no_grad():
    psi_values = student_model(inputs).cpu().numpy()

psi_flat = psi_values.flatten()

# Estimate initial parameters (you'll have 4 parameters per term)
N = 2  # Degree of power series
M = 2  # Number of Fourier terms
initial_params = np.ones((N + 1) * (M + 1) * 4 + 1)  # Initialize with ones
initial_params = np.random.uniform(-0.1, 0.1, len(initial_params))

lower_bounds = [-10] * len(initial_params)  # Lower bounds for all parameters
upper_bounds = [10] * len(initial_params)  # Upper bounds for all parameters

bounds = (lower_bounds, upper_bounds)

# Fit the function
params, covariance = curve_fit(
    target_function,
    (x_flat, y_flat),
    psi_flat,
    p0=initial_params,
    bounds=bounds,
    maxfev=20000
)

fitted_psi_values = target_function((X.flatten(), Y.flatten()), *params)
fitted_psi_grid = fitted_psi_values.reshape(100, 100)

fig = plt.figure(figsize=(14, 6))

# Student Model Predictions
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, psi_values.reshape(100, 100), cmap='viridis', edgecolor='k')
ax1.set_title("Student Model Predictions")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("psi")

# Fitted Function
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, fitted_psi_grid, cmap='plasma', edgecolor='k')
ax2.set_title("Fitted Power-Fourier-Sinh-Cosh Function")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("psi")

plt.show()

residuals = psi_flat - fitted_psi_values
mse = np.mean(residuals**2)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# Plot residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50, alpha=0.7, color="blue", label="Residuals")
plt.axvline(0, color="red", linestyle="--", label="Zero Error Line")
plt.title("Residuals of Fitted Function vs Model")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

params_dict = {}
index = 0

for n in range(N + 1):
    for m in range(M + 1):
        # Assign the parameters with descriptive names
        params_dict[f"a_nm_{n}_{m}"] = params[index]
        params_dict[f"b_nm_{n}_{m}"] = params[index + 1]
        params_dict[f"c_nm_{n}_{m}"] = params[index + 2]
        params_dict[f"k_nm_{n}_{m}"] = params[index + 3]
        index += 4

# Add the damping factor
params_dict["beta"] = params[-1]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"fitted_function_params_{timestamp}.json", "w") as file:
    json.dump(params_dict, file)

print("Fitted function parameters saved to 'fitted_function_params.json'.")

"""IF NEEDED"""
# with open("fitted_function_params.json", "r") as file:
#     loaded_params = list(json.load(file).values())

# # Define a function to compute psi using loaded parameters
# def reconstructed_function(inputs):
#     return target_function(inputs, *loaded_params)

# # Evaluate for new inputs
# x_new, y_new = 1.2, -0.8
# psi_new = reconstructed_function((x_new, y_new))
# print(f"Predicted psi at (x={x_new}, y={y_new}): {psi_new}")