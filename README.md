# Robust and Reliable Neural Networks  
### Analyzing Model Performance and Confidence under Noise and Adversarial Perturbations

## 📌 Overview
This project investigates not only the **accuracy** of neural networks but also their **reliability and calibration** under different conditions.  

While deep learning models often achieve high accuracy, they can still make **overconfident incorrect predictions**, which is critical in real-world applications such as healthcare and autonomous systems.

This project evaluates:
- Performance on clean data
- Robustness under noise
- Vulnerability to adversarial attacks
- Calibration of model confidence
- Improvement using temperature scaling

---

## 📊 Dataset
We use the **MNIST dataset**:
- 28 × 28 grayscale images
- 10 classes (digits 0–9)
- 60,000 training samples, 10,000 test samples

Preprocessing:
- Normalization using mean = 0.1307 and std = 0.3081
- Train / validation split for proper evaluation

---

## 🧠 Model
Baseline model: **Multilayer Perceptron (MLP)**

Architecture:
- Flatten (28×28 → 784)
- Fully connected layers: 256 → 128 → 10
- ReLU activation
- Dropout (0.2) for regularization

Training:
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Model selection based on validation accuracy

---

## 📈 Evaluation Metrics
We evaluate both performance and reliability:

### Performance
- Accuracy
- Precision, Recall, F1-score (macro)
- Confusion Matrix

### Reliability
- Confidence (softmax probabilities)
- Confidence breakdown (correct vs incorrect)
- Expected Calibration Error (ECE)
- Reliability Diagrams

---

## 🔬 Experiments

### 1. Clean Data Evaluation
- Accuracy: **~97.9%**
- ECE: **0.0025 (very well calibrated)**

Observation:
- High accuracy
- Slight overconfidence in incorrect predictions

---

### 2. Gaussian Noise Robustness
- Noise: σ = 0.2
- Accuracy: **~95.8%**
- ECE: **0.0601**

Observation:
- Moderate drop in accuracy
- Calibration degrades under noise

---

### 3. FGSM Adversarial Attack
- Epsilon: 0.15
- Accuracy: **~43.9%**
- ECE: **0.1397**

Observation:
- Severe performance degradation
- Model becomes highly unreliable
- Confidence no longer reflects correctness

---

### 4. Temperature Scaling (Calibration Improvement)
- Learned temperature: **~1.05**
- Accuracy unchanged
- ECE improved from **0.0025 → 0.0021**

Observation:
- Model is already well calibrated on clean data
- Temperature scaling provides marginal improvement
- More useful under noisy/adversarial settings

---

## 📊 Key Insights
- High accuracy does **not guarantee reliability**
- Models remain **overconfident even when wrong**
- Noise reduces performance and calibration
- Adversarial attacks severely break both accuracy and reliability
- Calibration techniques can improve confidence alignment

---

## 🗂️ Project Structure

nn-project/
│
├── evaluate_best_model.py          # Clean evaluation
├── evaluate_noise_model.py         # Gaussian noise evaluation
├── evaluate_FGSM_model.py          # Adversarial evaluation
├── evaluate_temperature_scaling.py # Calibration improvement
│
├── outputs/
│   ├── models/                    # Saved model checkpoints
│   └── visualizations/            # Plots and figures
│
└── data/                          # MNIST dataset