import matplotlib.pyplot as plt

# Accuracy data with initial accuracy (epoch 0)
acc_data = {
    "1e-7": {
        0: 30.54,
        1: 31.05,
        10: 31.05,
        20: 26.85,
    },
    "1e-10": {
        0: 30.54,
        1: 38.45,
        2: 40.34,
        3: 39.34,
        4: 40.92,
        20: 40.15,
    },
    "1e-11": {
        0: 30.54,
        1: 38.92,
        2: 40.47,
        3: 39.90,
        10: 40.45,
        20: 38.36,
    },
}

# === Plot 1: Accuracy (%) ===
plt.figure(figsize=(10, 6))
for lr, data in acc_data.items():
    epochs = sorted(data.keys())
    accs = [data[ep] for ep in epochs]
    plt.plot(epochs, accs, marker='o', label=f"LR = {lr}")

plt.title("Accuracy vs Epoch under Different Learning Rates", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xticks(range(0, 21, 1))
plt.ylim(25, 45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Learning Rate")
plt.tight_layout()
plt.savefig("accuracy_vs_epoch.png")
plt.show()

# === Plot 2: Correct count (/1000) ===
plt.figure(figsize=(10, 6))
for lr, data in acc_data.items():
    epochs = sorted(data.keys())
    correct = [round(data[ep] * 10) for ep in epochs]  # accuracy% * 10 = correct / 1000
    plt.plot(epochs, correct, marker='s', label=f"LR = {lr}")

plt.title("Correct Predictions vs Epoch (out of 1000)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Correct Count", fontsize=12)
plt.xticks(range(0, 21, 1))
plt.ylim(250, 450)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Learning Rate")
plt.tight_layout()
plt.savefig("correct_vs_epoch.png")