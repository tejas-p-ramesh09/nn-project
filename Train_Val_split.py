import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 1. Configuration

DATA_DIR = "./data"
BATCH_SIZE = 128
VAL_SIZE = 10000
RANDOM_SEED = 42
SPLIT_DIR = "outputs/splits/normalization" 


# 2. Define transform (ONLY ToTensor for V1)

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# 3. Load datasets (tensor format)

full_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=tensor_transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=tensor_transform
)


# 4. Split training into train + validation

train_size = len(full_train_dataset) - VAL_SIZE
val_size = VAL_SIZE

generator = torch.Generator().manual_seed(RANDOM_SEED)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)


# 5. Save split indices (reproducibility)

os.makedirs(SPLIT_DIR, exist_ok=True)

train_indices = train_dataset.indices
val_indices = val_dataset.indices

torch.save(train_indices, os.path.join(SPLIT_DIR, "train_indices.pt"))
torch.save(val_indices, os.path.join(SPLIT_DIR, "val_indices.pt"))

print("Split indices saved successfully!")


# 6. Create DataLoaders

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# 7. Split Information (clean summary)

print("\n===== DATA SPLIT SUMMARY =====")
print(f"Total training dataset: {len(full_train_dataset)}")
print(f"Train split: {len(train_dataset)}")
print(f"Validation split: {len(val_dataset)}")
print(f"Test dataset: {len(test_dataset)}")

print("\n===== BATCH INFORMATION =====")
print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")



# 8. Quick sanity check 

images, labels = next(iter(train_loader))

print("\n===== SANITY CHECK =====")
print(f"Batch image shape: {images.shape}")   # [BATCH_SIZE, 1, 28, 28]
print(f"Batch label shape: {labels.shape}")
print(f"Pixel range: min={images.min().item():.4f}, max={images.max().item():.4f}")