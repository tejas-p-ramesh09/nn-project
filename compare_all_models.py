import pandas as pd
import matplotlib.pyplot as plt
import os


# SAVE DIR
OUT_DIR = "./outputs/comparison"
os.makedirs(OUT_DIR, exist_ok=True)


# FINAL RESULTS
# Fill / edit these if you rerun experiments
results_main = [
    # Model, Setting, Accuracy, Macro_F1, Avg_Conf, Wrong_Conf, ECE
    ["MLP", "Clean", 0.9789, 0.9787, 0.9811, 0.7191, 0.0025],
    ["MLP", "Noise", 0.9578, 0.9578, 0.8979, 0.5354, 0.0601],
    ["MLP", "FGSM", 0.4397, 0.4318, 0.5725, 0.5450, 0.1397],

    ["CNN", "Clean", 0.9910, 0.9909, 0.9933, 0.7566, 0.0026],
    ["CNN", "Noise", 0.9903, 0.9902, 0.9922, 0.7278, 0.0022],
    ["CNN", "FGSM", 0.9652, 0.9650, 0.9809, 0.8116, 0.0165],
]

df_main = pd.DataFrame(
    results_main,
    columns=["Model", "Setting", "Accuracy", "Macro_F1", "Avg_Confidence", "Wrong_Confidence", "ECE"]
)

results_temp_clean = [
    # Model, Temperature, Accuracy_Before, Accuracy_After, ECE_Before, ECE_After
    ["MLP", 1.0498, 0.9789, 0.9789, 0.0025, 0.0021],
    ["CNN", 1.1356, 0.9910, 0.9910, 0.0026, 0.0018],
]

df_temp_clean = pd.DataFrame(
    results_temp_clean,
    columns=["Model", "Temperature", "Accuracy_Before", "Accuracy_After", "ECE_Before", "ECE_After"]
)

results_temp_shift = [
    # Model, Setting, Temperature, Accuracy_Before, Accuracy_After, ECE_Before, ECE_After
    ["MLP", "Noise", 1.0866, 0.9619, 0.9619, 0.0430, 0.0561],
    ["MLP", "FGSM",  1.0866, 0.4088, 0.4088, 0.2192, 0.1939],
    ["CNN", "Noise", 1.1356, 0.9888, 0.9888, 0.0872, 0.1212],
    ["CNN", "FGSM",  1.1356, 0.9287, 0.9287, 0.1132, 0.1530],
]

df_temp_shift = pd.DataFrame(
    results_temp_shift,
    columns=["Model", "Setting", "Temperature", "Accuracy_Before", "Accuracy_After", "ECE_Before", "ECE_After"]
)


# PRINT TABLES
print("\n" + "=" * 80)
print("MAIN COMPARISON TABLE")
print("=" * 80)
print(df_main.to_string(index=False))

print("\n" + "=" * 80)
print("TEMPERATURE SCALING (CLEAN)")
print("=" * 80)
print(df_temp_clean.to_string(index=False))

print("\n" + "=" * 80)
print("TEMPERATURE SCALING (NOISE / FGSM)")
print("=" * 80)
print(df_temp_shift.to_string(index=False))


# SAVE TABLES AS CSV
df_main.to_csv(f"{OUT_DIR}/main_comparison_table.csv", index=False)
df_temp_clean.to_csv(f"{OUT_DIR}/temperature_scaling_clean.csv", index=False)
df_temp_shift.to_csv(f"{OUT_DIR}/temperature_scaling_shifted.csv", index=False)


# PLOT 1: Accuracy comparison across settings
pivot_acc = df_main.pivot(index="Setting", columns="Model", values="Accuracy")

plt.figure(figsize=(8, 5))
for model in pivot_acc.columns:
    plt.plot(pivot_acc.index, pivot_acc[model], marker="o", label=model)

plt.title("Accuracy Comparison Across Settings")
plt.ylabel("Accuracy")
plt.xlabel("Setting")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# PLOT 2: ECE comparison across settings
pivot_ece = df_main.pivot(index="Setting", columns="Model", values="ECE")

plt.figure(figsize=(8, 5))
for model in pivot_ece.columns:
    plt.plot(pivot_ece.index, pivot_ece[model], marker="o", label=model)

plt.title("ECE Comparison Across Settings")
plt.ylabel("ECE")
plt.xlabel("Setting")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ece_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# PLOT 3: Wrong-confidence comparison
pivot_wrong_conf = df_main.pivot(index="Setting", columns="Model", values="Wrong_Confidence")

plt.figure(figsize=(8, 5))
for model in pivot_wrong_conf.columns:
    plt.plot(pivot_wrong_conf.index, pivot_wrong_conf[model], marker="o", label=model)

plt.title("Wrong-Prediction Confidence Across Settings")
plt.ylabel("Wrong Prediction Confidence")
plt.xlabel("Setting")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/wrong_confidence_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# PLOT 4: Temperature scaling ECE improvement on clean
x = range(len(df_temp_clean))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], df_temp_clean["ECE_Before"], width=width, label="Before")
plt.bar([i + width/2 for i in x], df_temp_clean["ECE_After"], width=width, label="After")

plt.xticks(list(x), df_temp_clean["Model"])
plt.ylabel("ECE")
plt.title("Temperature Scaling on Clean Data")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/temp_scaling_clean_ece.png", dpi=300, bbox_inches="tight")
plt.show()


# PLOT 5: Temperature scaling ECE change under shift
labels = [f"{row.Model}-{row.Setting}" for row in df_temp_shift.itertuples()]
x = range(len(df_temp_shift))

plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x], df_temp_shift["ECE_Before"], width=width, label="Before")
plt.bar([i + width/2 for i in x], df_temp_shift["ECE_After"], width=width, label="After")

plt.xticks(list(x), labels, rotation=20)
plt.ylabel("ECE")
plt.title("Temperature Scaling Under Noise and FGSM")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/temp_scaling_shifted_ece.png", dpi=300, bbox_inches="tight")
plt.show()


# SIMPLE SUMMARY
print("\n" + "=" * 80)
print("SUMMARY INSIGHTS")
print("=" * 80)

best_clean = df_main[df_main["Setting"] == "Clean"].sort_values("Accuracy", ascending=False).iloc[0]
best_noise = df_main[df_main["Setting"] == "Noise"].sort_values("Accuracy", ascending=False).iloc[0]
best_fgsm = df_main[df_main["Setting"] == "FGSM"].sort_values("Accuracy", ascending=False).iloc[0]

print(f"Best clean accuracy  : {best_clean['Model']} ({best_clean['Accuracy']:.4f})")
print(f"Best noise accuracy  : {best_noise['Model']} ({best_noise['Accuracy']:.4f})")
print(f"Best FGSM accuracy   : {best_fgsm['Model']} ({best_fgsm['Accuracy']:.4f})")

mlp_fgsm = df_main[(df_main["Model"] == "MLP") & (df_main["Setting"] == "FGSM")]["Accuracy"].values[0]
cnn_fgsm = df_main[(df_main["Model"] == "CNN") & (df_main["Setting"] == "FGSM")]["Accuracy"].values[0]
print(f"FGSM gap (CNN - MLP) : {cnn_fgsm - mlp_fgsm:.4f}")

print("\nFiles saved to:", OUT_DIR)