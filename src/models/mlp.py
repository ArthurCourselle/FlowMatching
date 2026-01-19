import torch
import torch.nn as nn
import numpy as np

class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time embedding
    """
    def __init__(self, dim: int, max_period=10000) -> None:
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2 # Half for sin and half for cos
        self.inv_freq = torch.exp(
            torch.arange(start=0 , end=self.half_dim, dtype=torch.float32) * (-np.log(max_period) / self.half_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch_size, 1) or (batch_size,)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        self.inv_freq = self.inv_freq.to(t.device)
        # sin/cos embedding
        # (B, 1) * (1, half_dim) -> (B, half_dim)
        sinusoid_inp = t * self.inv_freq.unsqueeze(0)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class MLP(nn.Module):
    """
    Simple Time-Dependent MLP.
    Concatenates input x and time embedding of t.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()

        self.time_layer = TimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B, 1) or (B,)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        t_emb = self.time_layer(t) # (B, H)
        return self.mlp(torch.cat([x, t_emb], dim=-1))
