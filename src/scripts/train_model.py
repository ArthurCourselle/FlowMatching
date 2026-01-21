import torch
import argparse
import os
from utils.data import get_data
from models.mlp import MLP
from models.unet import UNetModel
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Training")
    parser.add_argument(
        "--data",
        type=str,
        default="2d",
        choices=["2d", "CIFAR10", "MNIST"],
        help="Dataset type",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default="checkerboard",
        choices=["checkerboard", "gaussian"],
        help="Subtype for 2d data",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    print(f"Setting up Flow Matching training for {args.data}...")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    print("Loading data...")
    dataloader = get_data(args.data, batch_size=args.batch_size, subtype=args.subtype)

    if args.data == "2d":
        model = MLP(input_dim=2, hidden_dim=64, time_dim=32)
        save_name = f"model_2d_{args.subtype}.pt"
    else:
        if args.data == "MNIST":
            in_channels = 1
            image_size = 32
        elif args.data == "CIFAR10":
            in_channels = 3
            image_size = 32

        save_name = f"model_{args.data}.pt"

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

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume) + 1
        else:
            print(
                f"Warning: Checkpoint {args.resume} not found. Starting from scratch."
            )

    print(f"Starting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, start_epoch=start_epoch, save_name=save_name)

    os.makedirs("./checkpoints", exist_ok=True)
    save_path = os.path.join("./checkpoints", save_name)
    trainer.save_checkpoint(
        os.path.join("./checkpoints", f"checkpoint_{save_name}"),
        start_epoch + args.epochs - 1,
    )
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
