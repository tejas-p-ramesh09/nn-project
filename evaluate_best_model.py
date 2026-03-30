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


BATCH_SIZE = 64
MODEL_DIR = "./outputs/models"
MODEL_PATTERN = os.path.join(MODEL_DIR, "best_mlp_mnist_epoch*.pt")


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


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


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def get_latest_best_model_path():
    matching_models = glob.glob(MODEL_PATTERN)
    if not matching_models:
        raise FileNotFoundError(
            f"No saved best model found matching pattern: {MODEL_PATTERN}"
        )

    return max(matching_models, key=os.path.getmtime)


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

model = MLP().to(device)
best_model_path = get_latest_best_model_path()
model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"Loaded best model from: {best_model_path}")

criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2f}%")


model.eval()

all_preds = []
all_labels = []
all_confidences = []
correct_confidences = []
wrong_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        confidences, preds = torch.max(probs, dim=1)

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

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 60)
print("CLEAN TEST PERFORMANCE")
print("=" * 60)
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

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Best MLP on MNIST")
plt.show()

avg_conf_all = all_confidences.mean()
avg_conf_correct = correct_confidences.mean() if len(correct_confidences) > 0 else 0.0
avg_conf_wrong = wrong_confidences.mean() if len(wrong_confidences) > 0 else 0.0

print("\nConfidence Summary:")
print(f"Average Confidence (All Predictions)    : {avg_conf_all:.4f}")
print(f"Average Confidence (Correct Predictions): {avg_conf_correct:.4f}")
print(f"Average Confidence (Wrong Predictions)  : {avg_conf_wrong:.4f}")

plt.figure(figsize=(8, 5))
plt.hist(correct_confidences, bins=30, alpha=0.7, label="Correct Predictions")
plt.hist(wrong_confidences, bins=30, alpha=0.7, label="Wrong Predictions")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.title("Confidence Distribution on Clean Test Set")
plt.legend()
plt.show()
