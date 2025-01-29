import torch
import torch.nn as nn
import torch.optim as optim

# New arch uses nlm as parameters rather than mere variables.

class WavefunctionNN(nn.Module):
    def __init__(self):
        super(WavefunctionNN, self).__init__()
        
        # Embeddings for n, l, m
        self.n_embedding = nn.Embedding(5, 4)  # 5 categories, 4-dimensional embedding
        self.l_embedding = nn.Embedding(4, 3)  # 4 categories, 3-dimensional embedding
        self.m_embedding = nn.Embedding(num_embeddings=7, embedding_dim=4)  # 7 categories, 4-dimensional embedding

        # Main network
        self.fc = nn.Sequential(
            nn.Linear(2 + 4 + 3 + 4, 128),  # Input: x, y + n_embed, l_embed, m_embed
            nn.Tanh(),
            nn.Dropout(p=0.1),

            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 2)  # Output: psi_real, psi_imag
        )

    def forward(self, x, n, l, m):
        # Shift m to [0, 6] for embedding lookup
        m = m + 3

        # Get embeddings
        n_embed = self.n_embedding(n)
        l_embed = self.l_embedding(l)
        m_embed = self.m_embedding(m)

        assert n.min() >= 0 and n.max() < 5, f"n out of range: {n}"
        assert l.min() >= 0 and l.max() < 4, f"l out of range: {l}"
        assert m.min() >= 0 and m.max() < 7, f"m out of range after shift: {m}"

        # Concatenate inputs and embeddings
        inputs = torch.cat([x, n_embed, l_embed, m_embed], dim=1)
        return self.fc(inputs)