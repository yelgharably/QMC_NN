import torch
import torch.nn as nn

def initialize_weights(module):
    """
    Applies appropriate initialization based on the type of module.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Small random values for embeddings

class WfcNN(nn.Module):
    def __init__(self, n_max=4, l_max=3, m_max=3, activation="Tanh"):
        super(WfcNN, self).__init__()
        self.n_max = n_max

        # Shared backbone for 2D inputs
        self.shared_layers = nn.Sequential(
            nn.Linear(2, 128),  # Input: (x, y)
            self._get_activation(activation),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            self._get_activation(activation),
        )

        # Dynamic parameter heads (unchanged)
        self.parameter_heads = nn.ModuleDict()
        for n in range(1, n_max + 1):
            l_dict = nn.ModuleDict()
            for l in range(0, n):
                m_dict = nn.ModuleDict()
                for m in range(-l, l + 1):
                    m_dict[f"m_{m}"] = nn.Sequential(
                        nn.Linear(64, 32),
                        self._get_activation(activation),
                        nn.Linear(32, 2)  # Output: (psi_real, psi_imag)
                    )
                l_dict[f"l_{l}"] = m_dict
            self.parameter_heads[f"n_{n}"] = l_dict

        # Apply weight initialization
        self.apply(initialize_weights)


    def forward(self, x, n, l, m):
        """
        Forward pass returns wavefunction predictions: [batch_size, 2]
        where output[:,0] = real part, output[:,1] = imaginary part.
        """
        # 1) Pass (x,y,z) through the shared layers
        shared_features = self.shared_layers(x)  # shape [batch_size, 64]

        # 2) For each sample in the batch, pick the correct sub-MLP
        psi_real, psi_imag = [], []
        batch_size = x.size(0)
        for i in range(batch_size):
            n_val = n[i].item()
            l_val = l[i].item()
            m_val = m[i].item()

            # Validate quantum numbers
            if not (1 <= n_val <= self.n_max):
                raise ValueError(f"Invalid n value: {n_val}")
            if not (0 <= l_val < n_val):
                raise ValueError(f"Invalid l value: {l_val}")
            if not (-l_val <= m_val <= l_val):
                raise ValueError(f"Invalid m value: {m_val}")

            # Sub-network
            n_key = f"n_{n_val}"
            l_key = f"l_{l_val}"
            m_key = f"m_{m_val}"
            output = self.parameter_heads[n_key][l_key][m_key](shared_features[i])
            # print(f"[DEBUG] sub-MLP output shape = {output.shape}")
            psi_real.append(output[0])
            psi_imag.append(output[1])

        # 3) Stack to get [batch_size, 2]
        psi_real = torch.stack(psi_real)
        psi_imag = torch.stack(psi_imag)
        psi_pred = torch.stack([psi_real, psi_imag], dim=1)

        # 4) Return wavefunction as-is (no forced normalization)
        return psi_pred

    def normalize_wavefunction(self, psi_pred, coords):
        """
        Optional utility if you want to do a one-off global normalization pass.
        coords: [batch_size, 3]
        psi_pred: [batch_size, 2]
        """
        # This is a naive approach; typically you'd do a sum or integral of |psi|^2 
        # across the *entire domain*, not just a single batch.
        psi_sq = psi_pred[:, 0]**2 + psi_pred[:, 1]**2
        total = torch.sum(psi_sq) + 1e-12
        norm_factor = torch.sqrt(total)
        return psi_pred / norm_factor

    def _get_activation(self, activation):
        if activation == "Tanh":
            return nn.Tanh()
        elif activation == "ReLU":
            return nn.ReLU()
        elif activation == "Swish":
            return nn.SiLU()  # Swish in PyTorch
        else:
            raise ValueError(f"Unsupported activation: {activation}")
