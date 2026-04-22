# Robust and Reliable Neural Networks  
### Analyzing Model Performance and Confidence under Noise and Adversarial Perturbations

---

## 📌 Overview
This project investigates not only the **accuracy** of neural networks, but also their **reliability, robustness, and calibration** under different conditions.

While deep learning models often achieve high accuracy, they can still produce **overconfident incorrect predictions**. This is critical in real-world applications such as healthcare and autonomous systems, where incorrect but confident predictions can lead to severe consequences.

This project evaluates:
- Performance on clean data  
- Robustness under Gaussian noise  
- Vulnerability to adversarial attacks (FGSM)  
- Confidence and calibration behavior  
- Improvement using temperature scaling  
- Comparative performance of **MLP vs CNN**

---

## 📊 Dataset
We use the **MNIST dataset**:
- 28 × 28 grayscale images  
- 10 classes (digits 0–9)  
- 60,000 training samples  
- 10,000 test samples  

### Preprocessing
- Normalization using mean = 0.1307 and std = 0.3081  
- Train / validation split for model selection  

---

## 🧠 Models

### Multilayer Perceptron (MLP)
- Flatten input (784 features)  
- Fully connected layers: 256 → 128 → 10  
- ReLU activation  
- Dropout (0.2)  

---

### Convolutional Neural Network (CNN)
- Conv(32) → ReLU → MaxPool  
- Conv(64) → ReLU → MaxPool  
- Fully connected: 128 → 10  
- Dropout (0.3)  

---

### Training Setup
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Model selection based on validation accuracy  

---

## 📈 Evaluation Metrics

### Performance Metrics
- Accuracy  
- Precision, Recall, F1-score (macro)  
- Confusion Matrix  

### Reliability Metrics
- Confidence (softmax probabilities)  
- Confidence for correct vs incorrect predictions  
- Expected Calibration Error (ECE)  
- Reliability Diagrams  

---

## 🔬 Experiments

### 1. Clean Data Evaluation

| Model | Accuracy | Macro F1 | ECE |
|------|--------|----------|------|
| MLP  | 0.9789 | 0.9787   | 0.0025 |
| CNN  | 0.9910 | 0.9909   | 0.0026 |

**Observation:**
- Both models achieve high accuracy  
- Calibration is already strong  
- CNN performs slightly better  

---

### 2. Gaussian Noise Robustness (σ = 0.2)

| Model | Accuracy | Macro F1 | ECE |
|------|--------|----------|------|
| MLP  | 0.9578 | 0.9578   | 0.0601 |
| CNN  | 0.9903 | 0.9902   | 0.0022 |

**Observation:**
- MLP shows performance degradation  
- CNN remains highly robust  
- Calibration worsens significantly for MLP  

---

### 3. FGSM Adversarial Attack (ε = 0.15)

| Model | Accuracy | Macro F1 | ECE |
|------|--------|----------|------|
| MLP  | 0.4397 | 0.4318   | 0.1397 |
| CNN  | 0.9652 | 0.9650   | 0.0165 |

**Observation:**
- MLP performance collapses under attack  
- CNN is significantly more robust  
- Calibration degrades severely for MLP  

---

### 4. Confidence Analysis (Wrong Predictions)

| Setting | MLP | CNN |
|--------|-----|-----|
| Clean  | 0.7191 | 0.7566 |
| Noise  | 0.5354 | 0.7278 |
| FGSM   | 0.5450 | 0.8116 |

**Observation:**
- Models remain confident even when wrong  
- CNN tends to be more confident overall  
- Confidence becomes unreliable under adversarial conditions  

---

### 5. Temperature Scaling (Clean Data)

| Model | ECE Before | ECE After |
|------|-----------|----------|
| MLP  | 0.0025    | 0.0021   |
| CNN  | 0.0026    | 0.0018   |

**Observation:**
- Slight improvement in calibration  
- No impact on accuracy  
- Gains are small due to already good calibration  

---

### 6. Temperature Scaling under Distribution Shift

| Model | Setting | ECE Before | ECE After |
|------|--------|-----------|----------|
| MLP  | Noise  | 0.0430    | 0.0561   |
| MLP  | FGSM   | 0.2192    | 0.1939   |
| CNN  | Noise  | 0.0872    | 0.1212   |
| CNN  | FGSM   | 0.1132    | 0.1530   |

**Observation:**
- Does not consistently improve calibration under shift  
- Sometimes worsens ECE  
- Shows limited generalization of calibration methods  

---

## 📊 Key Insights
- High accuracy does **not guarantee reliability**  
- Models remain **overconfident when wrong**  
- MLP is highly vulnerable to adversarial attacks  
- CNN is significantly more robust  
- Calibration degrades under noise and adversarial conditions  
- Temperature scaling improves clean calibration but fails under distribution shift  

---

## 📉 Final Conclusion
The CNN consistently outperforms the MLP in:
- Accuracy  
- Robustness  
- Stability under perturbations  
- Calibration consistency  

While both models are well calibrated on clean data, reliability breaks under noise and adversarial conditions. Temperature scaling offers limited improvements and does not generalize well to shifted distributions.

This highlights the importance of evaluating models beyond accuracy and considering robustness and calibration in real-world deployment.

---

## 🗂️ Project Structure
```
nn-project/
│
├── main_mlp.py
├── main_cnn.py
│
├── evaluate_best_model.py
├── evaluate_best_cnn_model.py
│
├── evaluate_noise_model.py
├── evaluate_noise_cnn_model.py
│
├── evaluate_FGSM_model.py
├── evaluate_FGSM_cnn_model.py
│
├── evaluate_temperature_scaling.py
├── evaluate_temperature_scaling_cnn.py
├── evaluate_temperature_scaling_robust.py
│
├── compare_all_models.py
│
├── outputs/
│   ├── models/
│   ├── visualizations/
│   └── comparison/
│
└── data/
```

---

## ▶️ How to Run

### Install dependencies (uv)
```bash
uv add torch torchvision matplotlib numpy pandas scikit-learn
```

### Train models
```bash
python main_mlp.py
python main_cnn.py
```

### Run evaluations
```bash
python evaluate_best_model.py
python evaluate_noise_model.py
python evaluate_FGSM_model.py
```

### Compare results
```bash
python compare_all_models.py
```