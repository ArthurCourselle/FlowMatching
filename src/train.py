import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, dataloader, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        
    def train(self, epochs: int):

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            dataset = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in dataset:
                # Handle both tuple (x, y) and single tensor datasets
                if isinstance(batch, (list, tuple)):
                    x_1 = batch[0]
                    y = batch[1].to(self.device)
                else:
                    x_1 = batch
                    y = None
                    
                x_1 = x_1.to(self.device)
                
                self.optimizer.zero_grad()
                b = x_1.shape[0]
                
                t = torch.rand((b, 1), device=self.device) # uniform t
                x_0 = torch.randn_like(x_1) # Gaussian noise
                
                # Conditional flow (OT Path)
                # xt ~ p_t(x|x1).
                # Expand t to match x dimensions for broadcasting
                t_expanded = t.view(b, *([1] * (x_1.dim() - 1)))
                x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
                
                # Vector field u_t (derivative of x_t w.r.t t)
                u_t = x_1 - x_0

                if y is not None:
                    v_t = self.model(x_t, t.squeeze(-1), y=y)
                else:
                    v_t = self.model(x_t, t.squeeze(-1))
                
                loss = F.mse_loss(v_t, u_t) # regression
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                dataset.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
