import torch
import torch.nn as nn
import torch.optim as optim

class WfcNN(nn.Module):
    """
    Wavefunction neural network (WfcNN) with dynamic parameter gates for quantum numbers (n, l, m).
    Takes (x, y, z) inputs and quantum numbers, outputs real and imaginary parts of the wavefunction.
    """

    def __init__(self, n_max=4, l_max=3, m_max=3, activation="Tanh"):
        """
        Initialize the Wavefunction Neural Network.

        Args:
        - n_max: Maximum principal quantum number.
        - l_max: Maximum azimuthal quantum number.
        - m_max: Maximum magnetic quantum number.
        - activation: Activation function ("Tanh", "ReLU", or "Swish").
        """
        super(WfcNN, self).__init__()
        
        # Shared backbone
        self.shared_layers = nn.Sequential(
            nn.Linear(3, 128),  # Input: (x, y, z)
            self._get_activation(activation),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            self._get_activation(activation),
        )

        # Dynamic parameter gates
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

    def forward(self, x, n, l, m):
        """
        Forward pass with conditional parameter gates for quantum numbers.

        Args:
        - x: Tensor of shape (batch_size, 3) containing (x, y, z) inputs.
        - n: Tensor of shape (batch_size,) containing principal quantum numbers.
        - l: Tensor of shape (batch_size,) containing azimuthal quantum numbers.
        - m: Tensor of shape (batch_size,) containing magnetic quantum numbers.

        Returns:
        - output: Tensor of shape (batch_size, 2) containing normalized (psi_real, psi_imag).
        """
        # Process through shared layers
        shared_features = self.shared_layers(x)

        # Route through parameter gates
        outputs = []
        for i in range(x.size(0)):  # Loop over batch size
            n_val = n[i].item()
            l_val = l[i].item()
            m_val = m[i].item()

            # Validate quantum numbers
            assert n_val >= 1, f"Invalid n value: {n_val}. Must be >= 1."
            assert 0 <= l_val < n_val, f"Invalid l value: {l_val}. Must satisfy 0 <= l < n."
            assert -l_val <= m_val <= l_val, f"Invalid m value: {m_val}. Must satisfy -l <= m <= l."

            # Get parameter keys
            n_key = f"n_{n_val}"
            l_key = f"l_{l_val}"
            m_key = f"m_{m_val}"

            # Check if keys exist in parameter heads
            if n_key not in self.parameter_heads:
                raise ValueError(f"Invalid quantum number 'n': {n_val}")
            if l_key not in self.parameter_heads[n_key]:
                raise ValueError(f"Invalid quantum number 'l': {l_val} for 'n': {n_val}")
            if m_key not in self.parameter_heads[n_key][l_key]:
                raise ValueError(f"Invalid quantum number 'm': {m_val} for 'n': {n_val}, 'l': {l_val}")

            # Pass through the corresponding sub-network
            outputs.append(self.parameter_heads[n_key][l_key][m_key](shared_features[i]))

        psi_pred = torch.stack(outputs)
        # return psi_pred

        # Normalize predictions
        epsilon = 1e-6  # Small value to avoid division by zero
        norm_factor = torch.sqrt(torch.sum(psi_pred**2, dim=1, keepdim=True) + epsilon)
        psi_pred_normalized = psi_pred / norm_factor
        return psi_pred_normalized



    def _get_activation(self, activation):
        """
        Return the activation function based on user input.

        Args:
        - activation: String, one of "Tanh", "ReLU", or "Swish".

        Returns:
        - Activation function (nn.Module).
        """
        if activation == "Tanh":
            return nn.Tanh()
        elif activation == "ReLU":
            return nn.ReLU()
        elif activation == "Swish":
            return nn.SiLU()  # Swish is equivalent to SiLU in PyTorch
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
