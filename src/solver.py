import torch
import numpy as np

@torch.no_grad()
def euler_solve(model, x_0, steps=1000, y=None):
    """
    Simple Euler solver for ODE: dx/dt = v(x, t)
    x_0: Initial state (B, D) at t=0
    Returns trajectory or final state.
    """
    x = x_0
    b = x.shape[0]
    device = x.device
    
    dt = 1.0 / steps
    traj = [x.cpu()]
    
    for i in range(steps):
        # Current time t
        t = torch.ones((b, 1), device=device) * (i / steps)
        
        if y is not None:
             v = model(x, t.squeeze(-1), y=y)
        else:
             v = model(x, t.squeeze(-1))
        x = x + v * dt

        traj.append(x.cpu())
        
    return x, traj

@torch.no_grad()
def sample(model, n_samples: int, dim: int | tuple, device: str = "cpu", steps: int = 1000, y=None):
    model.eval()
    if isinstance(dim, int):
        x_0 = torch.randn(n_samples, dim, device=device)
    else:
        x_0 = torch.randn(n_samples, *dim, device=device)
    final_x, traj = euler_solve(model, x_0, steps=steps, y=y)
    return final_x, traj
