import torch
import torch.nn as nn
from ..models.base_network_5 import WfcNN

def finite_difference_grad(func, inputs, dim, epsilon=1e-5):
    """
    Computes the gradient of `func` with respect to `inputs` along a given dimension using finite differences.
    """
    inputs_plus = inputs.clone()
    inputs_minus = inputs.clone()

    # Avoid breaking the graph
    inputs_plus[:, dim] += epsilon
    inputs_minus[:, dim] -= epsilon

    func_plus = func(inputs_plus)
    func_minus = func(inputs_minus)

    grad = (func_plus - func_minus) / (2 * epsilon)
    return grad


def finite_difference_laplacian(func, inputs, dim, epsilon=1e-5):
    """
    Computes the Laplacian of `func` with respect to `inputs` along a given dimension using finite differences.
    """
    inputs_plus = inputs.clone()
    inputs_minus = inputs.clone()

    # Avoid breaking the graph
    inputs_plus[:, dim] += epsilon
    inputs_minus[:, dim] -= epsilon

    func_plus = func(inputs_plus)
    func_minus = func(inputs_minus)
    func_center = func(inputs)

    laplacian = (func_plus - 2 * func_center + func_minus) / (epsilon**2)
    return laplacian


class Losses(nn.Module):
    def __init__(self, r_max=50, hbar=1, mass=1, lambda_1=5.0, lambda_2=1.0, lambda_3=5.0, lambda_4=1.0):
        super(Losses, self).__init__()
        self.r_max = r_max
        self.hbar = hbar
        self.mass = mass
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

    def compute_schrodinger_loss(self, psi_pred, x, y, z, psi_true_real, psi_true_imag):
        """
        Compute the Schrödinger loss for the predicted wavefunction.
        """
        model = WfcNN()
        epsilon = 1e-6
        r_safe = torch.sqrt(torch.clamp(x**2 + y**2 + z**2, min=epsilon))

        # Split real and imaginary parts
        psi_pred_real = psi_pred[:, 0]
        psi_pred_imag = psi_pred[:, 1]

        # Gradients and Laplacians (using finite differences)
        model_func = lambda coords: model(coords)[:, 0]
        laplacian_real = sum(
            finite_difference_laplacian(
                model_func,
                original_coords,  # i.e. torch.stack([x, y, z], dim=1)
                dim=i
            )
            for i in range(3)
        )

        # Potential energy
        V = -1 / r_safe

        # Hamiltonian applied to wavefunction
        H_psi_real = -self.hbar**2 / (2 * self.mass) * laplacian_real + V * psi_pred_real
        H_psi_imag = -self.hbar**2 / (2 * self.mass) * laplacian_imag + V * psi_pred_imag

        # Energy expectation value
        numerator = torch.trapz(H_psi_real * psi_pred_real + H_psi_imag * psi_pred_imag, r_safe)
        denominator = torch.trapz((psi_pred_real**2 + psi_pred_imag**2), r_safe) + epsilon
        E = numerator / denominator

        # Compute Schrödinger loss
        schrodinger_loss = torch.mean((H_psi_real - E * psi_true_real)**2 + (H_psi_imag - E * psi_true_imag)**2)
        return schrodinger_loss

    def compute_normalization_loss(self, psi_pred, x, y, z):
        """
        Compute the normalization loss for the predicted wavefunction.
        """
        epsilon = 1e-6
        r_safe = torch.sqrt(torch.clamp(x**2 + y**2 + z**2, min=epsilon))

        # Split real and imaginary parts
        psi_pred_real = psi_pred[:, 0]
        psi_pred_imag = psi_pred[:, 1]

        # Compute |psi|^2
        psi_pred_mag_sq = psi_pred_real**2 + psi_pred_imag**2

        # Integrate |psi|^2 * r^2 over the radial domain
        normalization_integral = torch.trapz(psi_pred_mag_sq * r_safe**2, r_safe)

        # Normalization loss as deviation from expected value (1.0)
        normalization_loss = (normalization_integral - 1.0)**2
        return normalization_loss

    def compute_supervised_loss(self, psi_pred, psi_true_real, psi_true_imag):
        """
        Compute the supervised loss for the predicted wavefunction.
        """
        # Split real and imaginary parts
        psi_pred_real = psi_pred[:, 0]
        psi_pred_imag = psi_pred[:, 1]

        # Mean squared error
        supervised_loss = torch.mean((psi_pred_real - psi_true_real)**2 + (psi_pred_imag - psi_true_imag)**2)
        return supervised_loss

    def forward(self, psi_pred, x, y, z, psi_true_real, psi_true_imag):
        """
        Compute the total loss by combining Schrödinger loss with other components.
        Returns the total loss and a dictionary of individual losses.
        """
        schrodinger_loss = self.compute_schrodinger_loss(psi_pred, x, y, z, psi_true_real, psi_true_imag)
        normalization_loss = self.compute_normalization_loss(psi_pred, x, y, z)
        supervised_loss = self.compute_supervised_loss(psi_pred, psi_true_real, psi_true_imag)

        # Total loss
        total_loss = (
            self.lambda_1 * schrodinger_loss +
            self.lambda_3 * normalization_loss +
            self.lambda_4 * supervised_loss
        )

        # Dictionary of individual loss components
        loss_dict = {
            "schrodinger_loss": schrodinger_loss,
            "normalization_loss": normalization_loss,
            "supervised_loss": supervised_loss,
        }

        return total_loss, loss_dict

