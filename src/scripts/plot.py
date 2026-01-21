import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from torchvision.utils import make_grid
from models.mlp import MLP
from models.unet import UNetModel
from solver import sample


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Plotting")
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
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Class label for conditional generation",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()

    print(f"Setting up plotting for {args.data}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    if args.data == "2d":
        model = MLP(input_dim=2, hidden_dim=64, time_dim=32).to(device)
        # model_name = f"model_2d_{args.subtype}.pt"
        dim = 2
    else:
        if args.data == "MNIST":
            in_channels = 1
            image_size = 32
            dim = (1, 32, 32)
        elif args.data == "CIFAR10":
            in_channels = 3
            image_size = 32
            dim = (3, 32, 32)

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
        ).to(device)
        # model_name = f"model_{args.data}_500ep.pt"

    checkpoint_path = os.path.join("./checkpoints", args.model_path)
    try:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["model_state_dict"]
        )
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(
            f"Error: Checkpoint not found at {checkpoint_path}. Please run train_model.py first."
        )
        return

    print("Generating samples...")
    steps = 1000
    # For images, we don't need too many samples to visualize, but for 2D density we do.
    n_samples = args.n_samples
    if args.data != "2d":
        n_samples = min(n_samples, 64)

    y = None
    if args.class_label is not None:
        print(f"Generating samples for class {args.class_label}...")
        y = torch.full((n_samples,), args.class_label, device=device, dtype=torch.long)

    final_x, traj = sample(
        model, n_samples=n_samples, dim=dim, device=device, steps=steps, y=y
    )

    # Select 10 frames
    num_frames = 10
    indices = np.linspace(0, len(traj) - 1, num_frames, dtype=int)
    selected_frames = [traj[i] for i in indices]

    print("Plotting reconstruction steps...")

    if args.data == "2d":
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            frame_idx = indices[i]
            t_val = frame_idx / steps
            data = selected_frames[i].numpy()
            ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, color="blue")
            ax.set_title(f"t = {t_val:.2f}")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis("off")
    else:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            frame_idx = indices[i]
            t_val = frame_idx / steps

            # (B, C, H, W)
            imgs = selected_frames[i]

            # Normalize to [0, 1] for plotting if needed or assuming they are roughly consistent
            # The data was normalized to [-1, 1] (CIFAR).
            # Unnormalize for display: x * 0.5 + 0.5
            if args.data == "CIFAR10":
                imgs = imgs * 0.5 + 0.5
            elif args.data == "MNIST":
                imgs = imgs * 0.5 + 0.5

            imgs = torch.clamp(imgs, 0, 1)

            grid = make_grid(imgs[:64], nrow=8)
            grid_np = grid.permute(1, 2, 0).numpy()

            ax.imshow(grid_np, cmap="gray" if args.data == "MNIST" else None)
            ax.set_title(f"t = {t_val:.2f}")
            ax.axis("off")

    plt.tight_layout()
    output_file = f"reconstruction_{args.data}_{'subtype_' + args.subtype if args.data == '2d' else ''}.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
