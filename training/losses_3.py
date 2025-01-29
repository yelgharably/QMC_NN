import torch
import torch.nn as nn


class RefinedQuantumLoss(nn.Module):
    def __init__(self, r_max=20, hbar=1, mass=1, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0):
        super(RefinedQuantumLoss, self).__init__()
        self.r_max = r_max
        self.hbar = hbar
        self.mass = mass
        self.lambda_1 = lambda_1  # Weight for Schrödinger loss
        self.lambda_2 = lambda_2  # Weight for boundary loss
        self.lambda_3 = lambda_3  # Weight for normalization loss
        self.lambda_4 = lambda_4  # Weight for supervised loss

        # Log-sigma parameters for learnable loss scaling
        self.log_sigma_1 = nn.Parameter(torch.tensor(0.0))  # Schrödinger
        self.log_sigma_2 = nn.Parameter(torch.tensor(0.0))  # Boundary
        self.log_sigma_3 = nn.Parameter(torch.tensor(0.0))  # Normalization
        self.log_sigma_4 = nn.Parameter(torch.tensor(0.0))  # Supervised

    def forward(self, psi_pred, r, psi_true):
        epsilon = 1e-6  # Small value to prevent division by zero
        r_safe = r + epsilon  # Avoid division by zero in r-dependent terms
        psi_pred = torch.clamp(psi_pred, min=-10.0, max=10.0)  # Clamp predictions for numerical stability

        # Compute the first derivative of psi with respect to r
        psi_r = torch.autograd.grad(
            outputs=psi_pred,
            inputs=r,
            grad_outputs=torch.ones_like(psi_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute the second derivative of psi with respect to r
        psi_rr = torch.autograd.grad(
            outputs=psi_r,
            inputs=r,
            grad_outputs=torch.ones_like(psi_r),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute Laplacian (radial form)
        laplacian = psi_rr + (1 / r_safe) * psi_r

        # Define the potential (e.g., Coulomb potential for hydrogen)
        V = -1 / r_safe

        # Hamiltonian applied to psi
        H_psi = -self.hbar**2 / (2 * self.mass) * laplacian + V * psi_pred

        # Compute energy expectation value (numerator and denominator)
        numerator = torch.trapz(H_psi * psi_pred * r_safe**2, r_safe, dim=1)
        denominator = torch.trapz(psi_pred**2 * r_safe**2, r_safe, dim=1) + epsilon
        E = numerator / denominator

        # 1. Schrödinger Loss
        schrodinger_loss = torch.mean((H_psi - E.unsqueeze(1) * psi_pred)**2)

        # 2. Boundary Loss
        boundary_mask = r >= self.r_max
        num_boundary = boundary_mask.sum().float()
        if num_boundary > 0:
            boundary_loss = torch.mean(psi_pred[boundary_mask]**2)
        else:
            boundary_loss = torch.tensor(0.0, device=psi_pred.device)

        # 3. Normalization Loss
        normalization_integral = torch.trapz(psi_pred**2 * r_safe, r_safe, dim=1)
        normalization_loss = torch.mean((normalization_integral - 1)**2 + epsilon)

        # 4. Supervised Loss
        supervised_loss = torch.mean((psi_pred - psi_true)**2)

        # Apply log-sigma scaling
        sigma_1 = torch.exp(torch.clamp(self.log_sigma_1, min=-5, max=5))
        sigma_2 = torch.exp(torch.clamp(self.log_sigma_2, min=-5, max=5))
        sigma_3 = torch.exp(torch.clamp(self.log_sigma_3, min=-5, max=5))
        sigma_4 = torch.exp(torch.clamp(self.log_sigma_4, min=-5, max=5))

        total_loss = (
            self.lambda_1 * (schrodinger_loss / (2 * sigma_1**2) + torch.log(sigma_1)) +
            self.lambda_2 * (boundary_loss / (2 * sigma_2**2) + torch.log(sigma_2)) +
            self.lambda_3 * (normalization_loss / (2 * sigma_3**2) + torch.log(sigma_3)) +
            self.lambda_4 * (supervised_loss / (2 * sigma_4**2) + torch.log(sigma_4))
        )

        # NaN/Inf safety checks for debugging
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
            raise ValueError("NaN or Inf detected in total_loss")

        return total_loss
