import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# Config
# -----------------------------
VIS_DIR = "./outputs/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

BATCH_SIZE = 64
MODEL_DIR = "./outputs/models"
MODEL_PATTERN = os.path.join(MODEL_DIR, "best_mlp_mnist_epoch*.pt")
EPSILON = 0.15  # try 0.05, 0.10, 0.15, 0.20 later

# -----------------------------
# Device
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# -----------------------------
# Model
# -----------------------------
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

# -----------------------------
# Helpers
# -----------------------------
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

def fgsm_attack(images, epsilon, data_grad):
    perturbed_images = images + epsilon * data_grad.sign()
    MIN_NORM = (0.0 - 0.1307) / 0.3081
    MAX_NORM = (1.0 - 0.1307) / 0.3081
    return torch.clamp(perturbed_images, MIN_NORM, MAX_NORM)

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load best model
# -----------------------------
model = MLP().to(device)
best_model_path = get_latest_best_model_path()
model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"Loaded best model from: {best_model_path}")
print(f"Evaluating with FGSM attack, epsilon = {EPSILON}")

criterion = nn.CrossEntropyLoss()

# -----------------------------
# FGSM evaluation
# -----------------------------
model.eval()

all_preds = []
all_labels = []
all_confidences = []
correct_confidences = []
wrong_confidences = []

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Need gradients wrt input for FGSM
    images.requires_grad = True

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    adv_images = fgsm_attack(images, EPSILON, data_grad)

    # Evaluate on adversarial images
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_probs = F.softmax(adv_outputs, dim=1)
        confidences, preds = torch.max(adv_probs, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
    all_confidences.extend(confidences.cpu().numpy())

    correct_mask = preds == labels
    wrong_mask = preds != labels

    if correct_mask.any():
        correct_confidences.extend(confidences[correct_mask].cpu().numpy())

    if wrong_mask.any():
        wrong_confidences.extend(confidences[wrong_mask].cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)
correct_confidences = np.array(correct_confidences)
wrong_confidences = np.array(wrong_confidences)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 60)
print("FGSM ADVERSARIAL TEST PERFORMANCE")
print("=" * 60)
print(f"Epsilon               : {EPSILON}")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Macro Precision       : {precision:.4f}")
print(f"Macro Recall          : {recall:.4f}")
print(f"Macro F1-score        : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("Per-Class Accuracy:")
for cls in range(10):
    cls_mask = all_labels == cls
    cls_acc = (all_preds[cls_mask] == all_labels[cls_mask]).mean()
    print(f"Digit {cls}: {cls_acc:.4f}")

print("\nClass-wise Confidence Summary:")
print("-" * 60)
for cls in range(10):
    cls_mask = all_labels == cls
    cls_correct_mask = (all_labels == cls) & (all_preds == cls)
    cls_wrong_mask = (all_labels == cls) & (all_preds != cls)

    avg_conf_cls = all_confidences[cls_mask].mean() if cls_mask.sum() > 0 else 0.0
    avg_conf_correct_cls = (
        all_confidences[cls_correct_mask].mean() if cls_correct_mask.sum() > 0 else 0.0
    )
    avg_conf_wrong_cls = (
        all_confidences[cls_wrong_mask].mean() if cls_wrong_mask.sum() > 0 else 0.0
    )

    print(
        f"Digit {cls}: "
        f"Avg Conf = {avg_conf_cls:.4f}, "
        f"Correct Conf = {avg_conf_correct_cls:.4f}, "
        f"Wrong Conf = {avg_conf_wrong_cls:.4f}"
    )

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title(f"Confusion Matrix - FGSM MLP on MNIST (epsilon={EPSILON})")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/CM_fgsm_mlp_mnist_eps{EPSILON}.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Confidence Summary
# -----------------------------
avg_conf_all = all_confidences.mean()
avg_conf_correct = correct_confidences.mean() if len(correct_confidences) > 0 else 0.0
avg_conf_wrong = wrong_confidences.mean() if len(wrong_confidences) > 0 else 0.0

print("\nConfidence Summary:")
print(f"Average Confidence (All Predictions)    : {avg_conf_all:.4f}")
print(f"Average Confidence (Correct Predictions): {avg_conf_correct:.4f}")
print(f"Average Confidence (Wrong Predictions)  : {avg_conf_wrong:.4f}")

ece = compute_ece(all_confidences, all_preds, all_labels)
print(f"ECE: {ece:.4f}")

# -----------------------------
# Reliability Diagram
# -----------------------------
plot_reliability_diagram(
    all_confidences,
    all_preds,
    all_labels,
    n_bins=10,
    save_path=f"{VIS_DIR}/Reliability_Diagram_fgsm_mlp_mnist_eps{EPSILON}.png",
    title=f"Reliability Diagram (FGSM, epsilon={EPSILON})",
)

# -----------------------------
# Confidence Histogram
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(correct_confidences, bins=30, alpha=0.7, label="Correct Predictions")
plt.hist(wrong_confidences, bins=30, alpha=0.7, label="Wrong Predictions")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.title(f"Confidence Distribution on FGSM Test Set (epsilon={EPSILON})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/Conf_Dist_fgsm_mlp_mnist_eps{EPSILON}.png", dpi=300, bbox_inches="tight")
plt.show()