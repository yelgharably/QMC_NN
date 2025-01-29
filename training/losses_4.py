import torch
import torch.nn as nn

class QuantumLoss_XYZ(nn.Module):
    def __init__(self, r_max=20, hbar=1, mass=1, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0):
        super(QuantumLoss_XYZ, self).__init__()
        self.r_max = r_max
        self.hbar = hbar
        self.mass = mass
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

    def update_weights(self, current_epoch, total_epochs):
        """Dynamically adjust weights based on epoch."""
        progress = current_epoch / total_epochs
        self.lambda_1 = progress  # Gradually increase Schrödinger loss
        self.lambda_2 = 0.5 * progress  # Gradually increase boundary loss
        self.lambda_3 = 0.1 * progress  # Slow increase for normalization
        self.lambda_4 = 1.0 - progress  # Decrease supervised loss

    def forward(self, psi_pred, x, y, psi_true_real, psi_true_imag):
        epsilon = 1e-6  # Small value to prevent division by zero
        r_safe = torch.sqrt(x**2 + y**2) + epsilon  # Compute r from x and y

        # Split predictions into real and imaginary components
        psi_pred_real = psi_pred[:, 0]
        psi_pred_imag = psi_pred[:, 1]

        # Compute magnitude squared of wavefunction
        psi_pred_mag_sq = psi_pred_real**2 + psi_pred_imag**2

        # Compute first derivatives with respect to x and y
        psi_x_real = torch.autograd.grad(
            outputs=psi_pred_real,
            inputs=x,
            grad_outputs=torch.ones_like(psi_pred_real),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_y_real = torch.autograd.grad(
            outputs=psi_pred_real,
            inputs=y,
            grad_outputs=torch.ones_like(psi_pred_real),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_x_imag = torch.autograd.grad(
            outputs=psi_pred_imag,
            inputs=x,
            grad_outputs=torch.ones_like(psi_pred_imag),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_y_imag = torch.autograd.grad(
            outputs=psi_pred_imag,
            inputs=y,
            grad_outputs=torch.ones_like(psi_pred_imag),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute Laplacian using the chain rule
        laplacian_real = (
            torch.autograd.grad(psi_x_real, x, torch.ones_like(psi_x_real), create_graph=True, retain_graph=True)[0] +
            torch.autograd.grad(psi_y_real, y, torch.ones_like(psi_y_real), create_graph=True, retain_graph=True)[0]
        )
        laplacian_imag = (
            torch.autograd.grad(psi_x_imag, x, torch.ones_like(psi_x_imag), create_graph=True, retain_graph=True)[0] +
            torch.autograd.grad(psi_y_imag, y, torch.ones_like(psi_y_imag), create_graph=True, retain_graph=True)[0]
        )

        # Potential
        V = -1 / r_safe

        # Hamiltonian applied to psi (real and imaginary parts)
        H_psi_real = -self.hbar**2 / (2 * self.mass) * laplacian_real + V * psi_pred_real
        H_psi_imag = -self.hbar**2 / (2 * self.mass) * laplacian_imag + V * psi_pred_imag

        if r_safe.dim() == 1:
            r_safe = r_safe.unsqueeze(0)
        if H_psi_real.dim() == 1:
            H_psi_real = H_psi_real.unsqueeze(0)
        if psi_pred_real.dim() == 1:
            psi_pred_real = psi_pred_real.unsqueeze(0)

        # Perform the integration
        numerator_real = torch.trapz(H_psi_real * psi_pred_real * r_safe**2, r_safe, dim=1)

        # Energy expectation value
        numerator_real = torch.trapz(H_psi_real * psi_pred_real * r_safe**2, r_safe, dim=1)
        numerator_imag = torch.trapz(H_psi_imag * psi_pred_imag * r_safe**2, r_safe, dim=1)
        numerator = numerator_real + numerator_imag
        denominator = torch.trapz(psi_pred_mag_sq * r_safe**2, r_safe, dim=1) + epsilon
        E = numerator / denominator

        # 1. Schrödinger Loss
        schrodinger_loss_real = torch.mean((H_psi_real - E.unsqueeze(1) * psi_pred_real)**2)
        schrodinger_loss_imag = torch.mean((H_psi_imag - E.unsqueeze(1) * psi_pred_imag)**2)
        schrodinger_loss = schrodinger_loss_real + schrodinger_loss_imag

        # 2. Boundary Loss
        boundary_mask = r_safe >= self.r_max
        num_boundary = boundary_mask.sum().float()
        if num_boundary > 0:
            boundary_loss = torch.mean(psi_pred_mag_sq[boundary_mask])
        else:
            boundary_loss = torch.tensor(0.0, device=psi_pred.device)

        # 3. Normalization Loss
        normalization_integral = torch.trapz(psi_pred_mag_sq * r_safe, r_safe, dim=1)
        normalization_loss = torch.mean((normalization_integral - 1)**2 + epsilon)

        # 4. Supervised Loss
        supervised_loss_real = torch.mean((psi_pred_real - psi_true_real)**2)
        supervised_loss_imag = torch.mean((psi_pred_imag - psi_true_imag)**2)
        supervised_loss = supervised_loss_real + supervised_loss_imag

        schrodinger_loss = self.compute_schrodinger_loss(psi_pred, x, y, z, r_safe, psi_true_real, psi_true_imag)

        # Total Loss
        total_loss = (
            self.lambda_1 * schrodinger_loss +
            self.lambda_2 * boundary_loss +
            self.lambda_3 * normalization_loss +
            self.lambda_4 * supervised_loss
        )
        return total_loss
