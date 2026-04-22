import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Config
BATCH_SIZE = 64
MODEL_DIR = "./outputs/models"
MODEL_PATTERN = os.path.join(MODEL_DIR, "best_mlp_mnist_epoch*.pt")
SPLIT_DIR = "./outputs/splits/normalization"
VIS_DIR = "./outputs/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


# Helpers
def get_latest_best_model_path():
    matching_models = glob.glob(MODEL_PATTERN)
    if not matching_models:
        raise FileNotFoundError(
            f"No saved best model found matching pattern: {MODEL_PATTERN}"
        )
    return max(matching_models, key=os.path.getmtime)

def compute_ece(confidences, predictions, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece

def plot_reliability_diagram(confidences, predictions, labels, n_bins=10, save_path=None, title="Reliability Diagram"):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_accuracies = []
    bin_confidences = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = 0.0

        bin_accuracies.append(accuracy_in_bin)
        bin_confidences.append(avg_confidence_in_bin)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(
        bin_centers,
        bin_accuracies,
        width=0.08,
        alpha=0.7,
        edgecolor="black",
        label="Accuracy",
    )
    plt.plot(
        bin_centers,
        bin_confidences,
        marker="o",
        linewidth=2,
        label="Confidence",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs)
            all_labels.append(labels.to(device))

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return all_logits, all_labels

def evaluate_from_logits(logits, labels):
    probs = F.softmax(logits, dim=1)
    confidences, preds = torch.max(probs, dim=1)

    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    confidences_np = confidences.detach().cpu().numpy()

    accuracy = np.mean(preds_np == labels_np)
    ece = compute_ece(confidences_np, preds_np, labels_np)

    return accuracy, ece, preds_np, labels_np, confidences_np


# Temperature Scaling Wrapper
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def set_temperature(self, val_loader, device):
        self.to(device)
        self.model.eval()

        logits, labels = collect_logits_and_labels(self.model, val_loader, device)

        nll_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self


# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

val_indices = torch.load(f"{SPLIT_DIR}/val_indices.pt")
val_dataset = Subset(full_train_dataset, val_indices)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Load best model
base_model = MLP().to(device)
best_model_path = get_latest_best_model_path()
base_model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"Loaded best model from: {best_model_path}")


# Fit temperature on validation set
scaled_model = ModelWithTemperature(base_model)
scaled_model.set_temperature(val_loader, device)


# Evaluate before and after on clean test set
test_logits, test_labels = collect_logits_and_labels(base_model, test_loader, device)
scaled_test_logits = scaled_model.temperature_scale(test_logits)

acc_before, ece_before, preds_before, labels_before, conf_before = evaluate_from_logits(test_logits, test_labels)
acc_after, ece_after, preds_after, labels_after, conf_after = evaluate_from_logits(scaled_test_logits, test_labels)

print("\n" + "=" * 60)
print("TEMPERATURE SCALING RESULTS (CLEAN TEST)")
print("=" * 60)
print(f"Accuracy Before Scaling : {acc_before:.4f}")
print(f"ECE Before Scaling      : {ece_before:.4f}")
print(f"Accuracy After Scaling  : {acc_after:.4f}")
print(f"ECE After Scaling       : {ece_after:.4f}")


# Reliability diagrams
plot_reliability_diagram(
    conf_before,
    preds_before,
    labels_before,
    n_bins=10,
    save_path=f"{VIS_DIR}/Reliability_Before_TempScaling.png",
    title="Reliability Diagram Before Temperature Scaling",
)

plot_reliability_diagram(
    conf_after,
    preds_after,
    labels_after,
    n_bins=10,
    save_path=f"{VIS_DIR}/Reliability_After_TempScaling.png",
    title="Reliability Diagram After Temperature Scaling",
)


# Confidence histograms
plt.figure(figsize=(8, 5))
plt.hist(conf_before, bins=30, alpha=0.7, label="Before Scaling")
plt.hist(conf_after, bins=30, alpha=0.7, label="After Scaling")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.title("Confidence Distribution Before vs After Temperature Scaling")
plt.legend()
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/Confidence_Before_After_TempScaling.png", dpi=300, bbox_inches="tight")
plt.show()