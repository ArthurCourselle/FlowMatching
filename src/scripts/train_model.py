
import torch
import argparse
import os
from utils.data import get_data
from models.mlp import MLP
from models.unet import UNetModel
from train import Trainer
from solver import sample

def main():
    parser = argparse.ArgumentParser(description="Flow Matching Training")
    parser.add_argument("--data", type=str, default="2d", choices=["2d", "CIFAR10", "MNIST"], help="Dataset type")
    parser.add_argument("--subtype", type=str, default="checkerboard", choices=["checkerboard", "gaussian"], help="Subtype for 2d data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print(f"Setting up Flow Matching training for {args.data}...")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")
    
    print("Loading data...")
    dataloader = get_data(args.data, batch_size=args.batch_size, subtype=args.subtype)
    
    if args.data == "2d":
        # MLP for 2D data
        model = MLP(input_dim=2, hidden_dim=64, time_dim=32)
        save_name = f"model_2d_{args.subtype}.pt"
    else:
        # UNet for Images
        if args.data == "MNIST":
            # MNIST is 1x28x28
            in_channels = 1
            image_size = 28
        elif args.data == "CIFAR10":
            # CIFAR10 is 3x32x32
            in_channels = 3
            image_size = 32

        save_name = f"model_{args.data}.pt"
        
        # Hyperparameters for UNet
        model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=64,
            out_channels=in_channels,
            num_res_blocks=2,
            attention_resolutions=(2,),
            dropout=0.1,
            channel_mult=(1, 2, 2, 2),
            num_heads=4,
            num_classes=10,
            
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, dataloader, device=DEVICE)
    
    print(f"Starting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs)

    os.makedirs("./checkpoints", exist_ok=True)
    save_path = os.path.join("./checkpoints", save_name)
    trainer.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
