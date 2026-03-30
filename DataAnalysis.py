import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter


def analyze_mnist(data_dir="./data", batch_size=256):
    # Transform with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print("=" * 60)
    print("MNIST DATASET ANALYSIS (WITH NORMALIZATION)")
    print("=" * 60)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    # Inspect sample
    sample_img, sample_label = train_dataset[0]
    print("\nSample inspection (AFTER normalization):")
    print(f"Image shape: {sample_img.shape}")
    print(f"Min pixel value: {sample_img.min().item():.4f}")
    print(f"Max pixel value: {sample_img.max().item():.4f}")
    print(f"Label: {sample_label}")

    # Class distribution
    train_labels = train_dataset.targets
    train_class_counts = Counter(train_labels.tolist())

    print("\nTraining class distribution:")
    for cls in range(10):
        print(f"Digit {cls}: {train_class_counts[cls]}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Compute mean/std AFTER normalization (sanity check)
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    num_pixels = 0

    for images, _ in train_loader:
        pixel_sum += images.sum().item()
        pixel_squared_sum += (images ** 2).sum().item()
        num_pixels += images.numel()

    mean = pixel_sum / num_pixels
    std = ((pixel_squared_sum / num_pixels) - (mean ** 2)) ** 0.5

    print("\nPixel stats AFTER normalization (should be ~0 mean, ~1 std):")
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")

    # Helper to UNNORMALIZE for visualization
    def unnormalize(img):
        return img * 0.3081 + 0.1307

    # Show images (unnormalized for correct display)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Sample MNIST Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        img, label = train_dataset[i]
        img = unnormalize(img)
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.set_title(f"{label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Class distribution plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(10), [train_class_counts[i] for i in range(10)])
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()


if __name__ == "__main__":
    analyze_mnist()