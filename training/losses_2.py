import torch
import torch.nn as nn
import numpy as np

class QuantumLoss(nn.Module):
    def __init__(self, r_max=20, hbar=1, mass=1, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0):
        super(QuantumLoss, self).__init__()
        self.r_max = r_max
        self.hbar = hbar
        self.mass = mass
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self.log_sigma_1 = nn.Parameter(torch.tensor(0.0))  # Schrödinger
        self.log_sigma_2 = nn.Parameter(torch.tensor(0.0))  # Boundary
        self.log_sigma_3 = nn.Parameter(torch.tensor(0.0))  # Normalization
        self.log_sigma_4 = nn.Parameter(torch.tensor(0.0))  # Supervised

    def update_weights(self, current_epoch, total_epochs):
        """Dynamically adjust weights based on epoch."""
        progress = current_epoch / total_epochs
        self.lambda_1 = progress  # Gradually increase Schrödinger loss
        self.lambda_2 = 0.5 * progress  # Gradually increase boundary loss
        self.lambda_3 = 0.1 * progress  # Slow increase for normalization
        self.lambda_4 = 1.0 - progress  # Decrease supervised loss

    def forward(self, psi_pred, r, theta, psi_true):
        # Compute loss components (same as before)
        epsilon = 1e-6
        r_safe = r + epsilon
        psi_pred = torch.clamp(psi_pred, min=-10.0, max=10.0)

        # Compute first derivative w.r. to r
        psi_r = torch.autograd.grad(
            outputs=psi_pred,
            inputs=r,
            grad_outputs=torch.ones_like(psi_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute second derivative w.r. to r
        psi_rr = torch.autograd.grad(
            outputs=psi_r,
            inputs=r,
            grad_outputs=torch.ones_like(psi_r),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute Laplacian
        laplacian = psi_rr + (1 / r_safe) * psi_r

        # Potential
        V = -1 / r_safe

        # Hamiltonian applied to psi
        H_psi = -self.hbar**2 / (2 * self.mass) * laplacian + V * psi_pred

        # Numerator and denominator for energy expectation
        numerator = torch.trapz(H_psi * psi_pred * r_safe**2, r_safe, dim=1)
        denominator = torch.trapz(psi_pred**2 * r_safe**2, r_safe, dim=1) + epsilon  # Prevent division by zero
        E = numerator / denominator

        # Schrödinger loss
        schrodinger_loss = torch.mean((H_psi - E.unsqueeze(1) * psi_pred)**2)

        # Boundary loss with safe handling
        boundary_mask = r >= self.r_max
        num_boundary = boundary_mask.sum().float()

        if num_boundary > 0:
            boundary_loss = torch.mean(psi_pred[boundary_mask]**2)
        else:
            boundary_loss = torch.tensor(0.0, device=psi_pred.device)

        # Normalization loss
        normalization_integral = torch.trapz(psi_pred**2 * r_safe, r_safe, dim=1)
        normalization_loss = torch.mean((normalization_integral - 1)**2 + epsilon)

        # Supervised loss
        supervised_loss = torch.mean((psi_pred - psi_true)**2)

        # Total loss
        sigma_1 = torch.exp(self.log_sigma_1)
        sigma_2 = torch.exp(self.log_sigma_2)
        sigma_3 = torch.exp(self.log_sigma_3)
        sigma_4 = torch.exp(self.log_sigma_4)

        total_loss = (
            schrodinger_loss / (2 * sigma_1**2) + torch.log(sigma_1) +
            boundary_loss / (2 * sigma_2**2) + torch.log(sigma_2) +
            normalization_loss / (2 * sigma_3**2) + torch.log(sigma_3) +
            supervised_loss / (2 * sigma_4**2) + torch.log(sigma_4)
        )


        # Check for NaNs in individual loss components
        if torch.isnan(schrodinger_loss) or torch.isinf(schrodinger_loss):
            print("NaN or Inf detected in schrodinger_loss")
        if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
            print("NaN or Inf detected in boundary_loss")
        if torch.isnan(normalization_loss) or torch.isinf(normalization_loss):
            print("NaN or Inf detected in normalization_loss")
        if torch.isnan(supervised_loss) or torch.isinf(supervised_loss):
            print("NaN or Inf detected in supervised_loss")
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("NaN or Inf detected in total_loss")
            raise ValueError("NaN or Inf detected in total_loss") # Failsafe

        return total_loss
    
class QuantumLoss_XYZ(nn.Module):
    def __init__(self, r_max=20, hbar=1, mass=1, lambda_1=2.0, lambda_2=1.0, lambda_3=1.0, lambda_4=2.0):
        super(QuantumLoss_XYZ, self).__init__()
        self.r_max = r_max
        self.hbar = hbar
        self.mass = mass
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self.log_sigma_1 = nn.Parameter(torch.tensor(0.0))  # Schrödinger
        self.log_sigma_2 = nn.Parameter(torch.tensor(0.0))  # Boundary
        self.log_sigma_3 = nn.Parameter(torch.tensor(0.0))  # Normalization
        self.log_sigma_4 = nn.Parameter(torch.tensor(0.0))  # Supervised

    def update_weights(self, current_epoch, total_epochs):
        """Dynamically adjust weights based on epoch."""
        progress = current_epoch / total_epochs
        self.lambda_1 = progress  # Gradually increase Schrödinger loss
        self.lambda_2 = 0.5 * progress  # Gradually increase boundary loss
        self.lambda_3 = 0.1 * progress  # Slow increase for normalization
        self.lambda_4 = 1.0 - progress  # Decrease supervised loss

    def forward(self, psi_pred, x, y, psi_true):
        epsilon = 1e-6

        # Compute gradients with respect to x, y, z
        psi_x = torch.autograd.grad(
            outputs=psi_pred,
            inputs=x,
            grad_outputs=torch.ones_like(psi_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        psi_y = torch.autograd.grad(
            outputs=psi_pred,
            inputs=y,
            grad_outputs=torch.ones_like(psi_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # psi_z = torch.autograd.grad(
        #     outputs=psi_pred,
        #     inputs=z,
        #     grad_outputs=torch.ones_like(psi_pred),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True
        # )[0]

        # Compute second derivatives
        psi_xx = torch.autograd.grad(
            outputs=psi_x,
            inputs=x,
            grad_outputs=torch.ones_like(psi_x),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        psi_yy = torch.autograd.grad(
            outputs=psi_y,
            inputs=y,
            grad_outputs=torch.ones_like(psi_y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # psi_zz = torch.autograd.grad(
        #     outputs=psi_z,
        #     inputs=z,
        #     grad_outputs=torch.ones_like(psi_z),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True
        # )[0]

        # Compute Laplacian in Cartesian coordinates
        laplacian = psi_xx + psi_yy

        # Compute potential (e.g., Coulomb potential)
        r = torch.sqrt(x**2 + y**2 + epsilon)
        V = -1 / r

        # Hamiltonian applied to psi
        H_psi = -self.hbar**2 / (2 * self.mass) * laplacian + V * psi_pred

        # Numerator and denominator for energy expectation
        numerator = torch.mean(H_psi * psi_pred)
        denominator = torch.mean(psi_pred**2) + epsilon  # Prevent division by zero
        E = numerator / denominator

        # Schrödinger loss
        schrodinger_loss = torch.mean((H_psi - E * psi_pred)**2)

        # Boundary loss
        boundary_mask = r >= 20.0  # Example boundary condition
        num_boundary = boundary_mask.sum().float()

        if num_boundary > 0:
            boundary_loss = torch.mean(psi_pred[boundary_mask]**2)
        else:
            boundary_loss = torch.tensor(0.0, device=psi_pred.device)

        # Normalization loss
        normalization_integral = torch.mean(psi_pred**2)
        normalization_loss = torch.mean((normalization_integral - 1)**2 + epsilon)

        # Supervised loss
        supervised_loss = torch.mean((psi_pred - psi_true)**2)

        # Total loss
        sigma_1 = torch.exp(self.log_sigma_1)
        sigma_2 = torch.exp(self.log_sigma_2)
        sigma_3 = torch.exp(self.log_sigma_3)
        sigma_4 = torch.exp(self.log_sigma_4)

        total_loss = (
            schrodinger_loss / (2 * sigma_1**2) + torch.log(sigma_1) +
            boundary_loss / (2 * sigma_2**2) + torch.log(sigma_2) +
            normalization_loss / (2 * sigma_3**2) + torch.log(sigma_3) +
            supervised_loss / (2 * sigma_4**2) + torch.log(sigma_4)
        )

        # Check for NaNs or Infs in loss components
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError("NaN or Inf detected in total_loss")

        return total_loss
