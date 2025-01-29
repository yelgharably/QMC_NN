# losses_5.py (revised example)

import torch
import torch.nn as nn

def normalization_loss_mc(psi_pred, coords, pdf_vals):
    """
    If coords ~ q(r), then integral(|psi|^2) ~ 
        mean( |psi(r_i)|^2 / q(r_i) ).
    We want that ~ 1 => (integral - 1)^2
    """
    if pdf_vals is None:
        return psi_pred.new_tensor(0.0)
    psi_sq = psi_pred[:, 0]**2 + psi_pred[:, 1]**2
    ratio = psi_sq / (pdf_vals + 1e-12)  # if pdf_vals is known
    integral_approx = ratio.mean()
    return (integral_approx - 1.0)**2

def finite_difference_laplacian(model_func, coords, epsilon=1e-4):
    """
    Compute Laplacian by finite differences:
      laplacian(psi) = sum_{dim in x,y} [psi(r+eps) - 2 psi(r) + psi(r-eps)] / eps^2
    model_func: a callable that takes coords [batch_size, 2] and returns [batch_size].
    coords: [batch_size, 2].
    """
    f0 = model_func(coords)  # Evaluate at center
    lap = torch.zeros_like(f0)

    for dim in range(2):  # Only two dimensions (x, y)
        coords_plus = coords.clone()
        coords_minus = coords.clone()
        coords_plus[:, dim] += epsilon
        coords_minus[:, dim] -= epsilon

        f_plus = model_func(coords_plus)
        f_minus = model_func(coords_minus)
        lap += (f_plus - 2.0 * f0 + f_minus) / (epsilon**2)

    return lap


def compute_laplacian_autograd(model_func, coords):
    """
    Computes the Laplacian of a scalar function f(x,y) with respect to the 2D coordinates
    using autograd.

    Args:
        model_func: A callable (e.g., lambda) that takes coords -> [batch_size] 
                    (i.e., returns a 1D tensor, typically the real or imaginary part).
        coords: [batch_size, 2], requires_grad_ will be set to True inside this function.

    Returns:
        lap: [batch_size], the Laplacian of model_func at each point.
    """

    # 1) Clone coords so we don't pollute the original tensor's graph
    coords = coords.clone().requires_grad_(True)

    # 2) Evaluate the function on the current coords
    #    Here we expect model_func(coords) to produce shape [batch_size].
    f0 = model_func(coords)  # e.g. real or imaginary part

    # 3) Compute the gradient wrt coords
    grad_f0 = torch.autograd.grad(
        f0.sum(),        # sum(...) so we get scalar
        coords,
        create_graph=True
    )[0]  # shape: [batch_size, 2] in 2D

    # 4) Sum the second derivatives to get the Laplacian
    lap = torch.zeros_like(f0)  # same shape as f0: [batch_size]
    for dim in range(coords.shape[1]):  # for each dimension in x,y
        second_deriv = torch.autograd.grad(
            grad_f0[:, dim].sum(),
            coords,
            create_graph=True
        )[0][:, dim]
        lap += second_deriv

    return lap


def finite_difference_laplacian_complex(model_func, coords, epsilon=1e-4):
    # Evaluate at center: complex dtype
    f0 = model_func(coords)                          # shape [batch_size], complex
    lap = torch.zeros_like(f0, dtype=f0.dtype)       # ensure same complex dtype

    for dim in range(3):
        coords_plus = coords.clone()
        coords_minus = coords.clone()
        coords_plus[:, dim] += epsilon
        coords_minus[:, dim] -= epsilon

        f_plus = model_func(coords_plus)
        f_minus = model_func(coords_minus)

        lap += (f_plus - 2*f0 + f_minus) / (epsilon**2)

    return lap


class Losses(nn.Module):
    def __init__(self, model, hbar=1.0, mass=1.0, 
                 lambda_sch=1.0, 
                 lambda_sup=1.0,
                 lambda_norm=1.0):
        """
        Args:
            model (nn.Module): e.g., WfcNN. We'll re-run 'model(coords, n, l, m)' inside this class.
            hbar, mass: for kinetic operator. Typically set to 1.0 in atomic units.
            lambda_sch: weight for Schr. PDE residual loss
            lambda_sup: weight for supervised MSE
            lambda_norm: optional weight for MC normalization loss
        """
        super().__init__()
        self.model = model
        self.hbar = hbar
        self.mass = mass
        self.lambda_sch = lambda_sch
        self.lambda_sup = lambda_sup
        self.lambda_norm = lambda_norm

    def normalization_loss_mc(self, coords, n, l, m, pdf_vals=None):
        """
        If your coords are drawn from a PDF q(r), then
          integral(|psi|^2) ~ (1/N) sum_i [|psi(r_i)|^2 / q(r_i)].
        We'll want that ~ 1.
        """
        if pdf_vals is None:
            return torch.tensor(0.0, device=coords.device)
        psi_pred = self.model(coords, n, l, m)  # shape [batch_size, 2]
        psi_sq = psi_pred[:, 0]**2 + psi_pred[:, 1]**2
        ratio = psi_sq / (pdf_vals + 1e-12)
        integral_approx = ratio.mean()
        return (integral_approx - 1.0)**2

    def forward(self, coords, n, l, m, psi_true_real=None, psi_true_imag=None, pdf_vals=None):
        """
        coords: [batch_size, 3]
        n, l, m: quantum numbers (tensor or single)
        psi_true_real, psi_true_imag: optional supervised label
        pdf_vals: optional PDF(q(r_i)) at each coords[i].
        """
        # ==============================
        # 1) PDE Residual: (H-E)*psi=0
        # ==============================
        # (a) For the energy estimate, do a no_grad pass
        lap_r = compute_laplacian_autograd(
            lambda c: self.model(c, n, l, m)[:, 0],
            coords
        )
        lap_i = compute_laplacian_autograd(
            lambda c: self.model(c, n, l, m)[:, 1],
            coords
        )

        with torch.no_grad():
            r = coords.norm(dim=1).clamp_min(1e-8)
            V = -1.0 / r

            T_r = - (self.hbar**2)/(2*self.mass) * lap_r
            T_i = - (self.hbar**2)/(2*self.mass) * lap_i

            psi_pred = self.model(coords, n, l, m)  # shape [batch_size, 2]
            psi_r = psi_pred[:, 0]
            psi_i = psi_pred[:, 1]

            H_psi_r = T_r + V * psi_r
            H_psi_i = T_i + V * psi_i
            numerator = torch.mean(psi_r * H_psi_r + psi_i * H_psi_i)
            denominator = torch.mean(psi_r**2 + psi_i**2) + 1e-12
            E_est = numerator / denominator

        # (b) PDE residual with grad 
        r = coords.norm(dim=1).clamp_min(1e-8)
        V = -1.0 / r
        T_r = - (self.hbar**2)/(2*self.mass) * lap_r
        T_i = - (self.hbar**2)/(2*self.mass) * lap_i

        psi_pred = self.model(coords, n, l, m)
        psi_r = psi_pred[:, 0]
        psi_i = psi_pred[:, 1]

        H_psi_r = T_r + V*psi_r
        H_psi_i = T_i + V*psi_i

        PDE_res_r = H_psi_r - E_est * psi_r
        PDE_res_i = H_psi_i - E_est * psi_i
        schrodinger_loss = torch.mean(PDE_res_r**2 + PDE_res_i**2)

        # ==========================
        # 2) Supervised MSE (optional)
        # ==========================
        if psi_true_real is not None and psi_true_imag is not None:
            sup_loss = torch.mean((psi_r - psi_true_real)**2 + (psi_i - psi_true_imag)**2)
        else:
            sup_loss = torch.tensor(0.0, device=coords.device)

        # ==========================
        # 3) Normalization Loss (MC)
        # ==========================
        norm_loss = normalization_loss_mc(psi_pred, coords, pdf_vals)

        # ==========================
        # 4) Combine
        # ==========================
        total_loss = (self.lambda_sch * schrodinger_loss
                      + self.lambda_sup * sup_loss
                      + self.lambda_norm * norm_loss)

        loss_dict = {
            "schrodinger_loss": schrodinger_loss.detach(),
            "supervised_loss": sup_loss.detach(),
            "normalization_loss": norm_loss.detach(),
            "energy_estimate": E_est.detach(),
        }
        return total_loss, loss_dict
