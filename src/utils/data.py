import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Literal
from torch.utils.data import TensorDataset
from typing import Optional
import numpy as np


def get_2d_data(
    type: Literal["gaussian", "checkerboard"], n_samples: int = 10000
) -> TensorDataset:
    if type == "gaussian":
        # Mixture of 8 Gaussians arranged in a circle
        scale = 2.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(n_samples):
            point = np.random.randn(2) * 0.5
            center = centers[np.random.choice(len(centers))]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414  # Normalize to [-2, 2] -> [-1.5, 1.5] range

        return TensorDataset(torch.from_numpy(dataset))

    elif type == "checkerboard":
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        dataset = np.stack([x1, x2 * 2], axis=1).astype("float32")
        # Normalize to be centered and scaled
        dataset = dataset / 2.0
        return TensorDataset(torch.from_numpy(dataset))

    else:
        raise ValueError(f"Unknown 2d data type: {type}")


def get_data(
    type: Literal["2d", "CIFAR10", "MNIST"],
    batch_size: int = 128,
    subtype: Optional[Literal["gaussian", "checkerboard"]] = None,
    num_workers: int = 4,
) -> DataLoader:

    if type == "2d":
        if subtype is None:
            raise ValueError(
                "subtype must be specified for 2d data (gaussian or checkerboard)"
            )
        dataset = get_2d_data(subtype, n_samples=50000)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    elif type == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Normalize to [-1, 1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    elif type == "MNIST":
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    else:
        raise ValueError(f"Unknown dataset type: {type}")
