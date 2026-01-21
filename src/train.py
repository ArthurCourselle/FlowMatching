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

    def train(self, epochs: int, start_epoch: int = 0, save_name: str = "model.pt"):

        self.model.train()

        for epoch in range(start_epoch, start_epoch + epochs):
            total_loss = 0
            dataset = tqdm(
                self.dataloader, desc=f"Epoch {epoch+1}/{start_epoch + epochs}"
            )
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

                t = torch.rand((b, 1), device=self.device)  # uniform t
                x_0 = torch.randn_like(x_1)  # Gaussian noise

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

                loss = F.mse_loss(v_t, u_t)  # regression

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                dataset.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
            if epoch % 10 == 0:
                print(f"Saving checkpoint at epoch {epoch+1}...")
                self.save_checkpoint(
                    os.path.join("./checkpoints", f"checkpoint_{save_name}"),
                    start_epoch + epoch - 1,
                )

    def save(self, path):
        """Save model state dict to path"""
        torch.save(self.model.state_dict(), path)

    def save_checkpoint(self, path, epoch):
        """Save full checkpoint including model, optimizer, and epoch number"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load checkpoint and return the epoch to resume from"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path}. Resuming from epoch {epoch + 1}")
        return epoch
