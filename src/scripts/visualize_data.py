
import matplotlib.pyplot as plt
import numpy as np
from utils.data import get_2d_data

def main():
    print("Generating checkerboard data...")
    # Get the dataset directly
    dataset = get_2d_data(type="checkerboard", n_samples=10000)
    
    # Extract data from TensorDataset
    data = dataset.tensors[0].numpy()
    
    print(f"Data shape: {data.shape}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, color='green')
    plt.title("Checkerboard Data Distribution")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)
    
    output_file = "checkerboard_data.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
if __name__ == "__main__":
    main()
