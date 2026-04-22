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



# CONFIG

MODEL_TYPE = "cnn"          # "mlp" or "cnn"
EVAL_MODE = "noise"         # "clean", "noise", or "fgsm"

BATCH_SIZE = 64
SPLIT_DIR = "./outputs/splits/normalization"
VIS_DIR = "./outputs/visualizations"
MODEL_DIR = "./outputs/models"

NOISE_SIGMA = 0.2
FGSM_EPSILON = 0.15

os.makedirs(VIS_DIR, exist_ok=True)

if MODEL_TYPE == "mlp":
    MODEL_PATTERN = os.path.join(MODEL_DIR, "best_mlp_mnist_epoch*.pt")
elif MODEL_TYPE == "cnn":
    MODEL_PATTERN = os.path.join(MODEL_DIR, "best_cnn_mnist_epoch*.pt")
else:
    raise ValueError("MODEL_TYPE must be 'mlp' or 'cnn'")



# DEVICE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)



# MODELS

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


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



# HELPERS

def get_latest_best_model_path():
    matching_models = glob.glob(MODEL_PATTERN)
    if not matching_models:
        raise FileNotFoundError(f"No model found matching: {MODEL_PATTERN}")
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


def plot_reliability_diagram(confidences, predictions, labels, title, save_path, n_bins=10):
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def add_gaussian_noise(images, sigma=0.2):
    noise = torch.randn_like(images) * sigma
    noisy_images = images + noise
    return torch.clamp(noisy_images, -1.0, 1.0)


def fgsm_attack(images, epsilon, data_grad):
    adv_images = images + epsilon * data_grad.sign()
    return torch.clamp(adv_images, -1.0, 1.0)


def collect_logits_and_labels(model, loader, device, mode="clean", sigma=0.2, epsilon=0.15, criterion=None):
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if mode == "clean":
            with torch.no_grad():
                logits = model(images)

        elif mode == "noise":
            noisy_images = add_gaussian_noise(images, sigma=sigma)
            with torch.no_grad():
                logits = model(noisy_images)

        elif mode == "fgsm":
            if criterion is None:
                raise ValueError("criterion must be provided for FGSM mode")

            images.requires_grad = True
            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            data_grad = images.grad.data
            adv_images = fgsm_attack(images, epsilon, data_grad)

            with torch.no_grad():
                logits = model(adv_images)

        else:
            raise ValueError("mode must be 'clean', 'noise', or 'fgsm'")

        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

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

    avg_conf_all = confidences_np.mean()
    avg_conf_correct = confidences_np[preds_np == labels_np].mean() if np.any(preds_np == labels_np) else 0.0
    avg_conf_wrong = confidences_np[preds_np != labels_np].mean() if np.any(preds_np != labels_np) else 0.0

    return {
        "accuracy": accuracy,
        "ece": ece,
        "preds": preds_np,
        "labels": labels_np,
        "confidences": confidences_np,
        "avg_conf_all": avg_conf_all,
        "avg_conf_correct": avg_conf_correct,
        "avg_conf_wrong": avg_conf_wrong,
    }



# TEMPERATURE SCALING WRAPPER

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def set_temperature(self, val_loader, device):
        self.to(device)
        self.model.eval()

        # IMPORTANT: fit temperature on CLEAN validation logits only
        val_logits, val_labels = collect_logits_and_labels(
            self.model,
            val_loader,
            device,
            mode="clean",
        )

        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self



# DATA

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



# LOAD MODEL

if MODEL_TYPE == "mlp":
    base_model = MLP().to(device)
elif MODEL_TYPE == "cnn":
    base_model = CNN().to(device)

best_model_path = get_latest_best_model_path()
base_model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"Loaded best model from: {best_model_path}")

criterion = nn.CrossEntropyLoss()


# FIT TEMPERATURE ON CLEAN VALIDATION

scaled_model = ModelWithTemperature(base_model)
scaled_model.set_temperature(val_loader, device)


# EVALUATE BEFORE / AFTER ON CHOSEN SETTING

test_logits, test_labels = collect_logits_and_labels(
    base_model,
    test_loader,
    device,
    mode=EVAL_MODE,
    sigma=NOISE_SIGMA,
    epsilon=FGSM_EPSILON,
    criterion=criterion,
)

scaled_test_logits = scaled_model.temperature_scale(test_logits)

before = evaluate_from_logits(test_logits, test_labels)
after = evaluate_from_logits(scaled_test_logits, test_labels)

tag = f"{MODEL_TYPE}_{EVAL_MODE}"
if EVAL_MODE == "noise":
    tag += f"_sigma{NOISE_SIGMA}"
elif EVAL_MODE == "fgsm":
    tag += f"_eps{FGSM_EPSILON}"

print("\n" + "=" * 60)
print(f"TEMPERATURE SCALING RESULTS ({MODEL_TYPE.upper()} - {EVAL_MODE.upper()})")
print("=" * 60)
print(f"Accuracy Before Scaling : {before['accuracy']:.4f}")
print(f"ECE Before Scaling      : {before['ece']:.4f}")
print(f"Accuracy After Scaling  : {after['accuracy']:.4f}")
print(f"ECE After Scaling       : {after['ece']:.4f}")
print(f"Avg Conf Before         : {before['avg_conf_all']:.4f}")
print(f"Avg Conf After          : {after['avg_conf_all']:.4f}")


# PLOTS

plot_reliability_diagram(
    before["confidences"],
    before["preds"],
    before["labels"],
    title=f"Reliability Before Temp Scaling ({MODEL_TYPE.upper()} - {EVAL_MODE.upper()})",
    save_path=f"{VIS_DIR}/Reliability_Before_TempScaling_{tag}.png",
)

plot_reliability_diagram(
    after["confidences"],
    after["preds"],
    after["labels"],
    title=f"Reliability After Temp Scaling ({MODEL_TYPE.upper()} - {EVAL_MODE.upper()})",
    save_path=f"{VIS_DIR}/Reliability_After_TempScaling_{tag}.png",
)

plt.figure(figsize=(8, 5))
plt.hist(before["confidences"], bins=30, alpha=0.7, label="Before Scaling")
plt.hist(after["confidences"], bins=30, alpha=0.7, label="After Scaling")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.title(f"Confidence Before vs After Temp Scaling ({MODEL_TYPE.upper()} - {EVAL_MODE.upper()})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/Confidence_Before_After_TempScaling_{tag}.png", dpi=300, bbox_inches="tight")
plt.show()