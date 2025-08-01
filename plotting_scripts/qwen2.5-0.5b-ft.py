import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Learning rates and corresponding accuracies
learning_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-20]
accuracies = [5.95, 21.34, 23.10, 24.60, 25.20, 25.00, 25.10]  # Accuracy(valid) %

# Baseline accuracies
qwen_base_acc = 24.30  # 0 epoch
deepseek_acc = 50.92

# Plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

# Plot learning rate vs. accuracy
ax.plot(
    [f"{lr:.0e}" for lr in learning_rates], 
    accuracies, 
    marker="o", 
    linewidth=2,
    label="Qwen2.5-0.5B FT on MATH"
)

# Baselines
ax.axhline(qwen_base_acc, linestyle="--", linewidth=1, label=f"Epoch 0 ({qwen_base_acc:.2f}%)")
ax.axhline(deepseek_acc, linestyle="--", linewidth=1.2, color="tab:orange",
           label=f"DeepSeek-R1-Distill-Qwen-7B ({deepseek_acc:.2f}%)")

# Formatting
ax.set_ylim(0, max(deepseek_acc, max(accuracies)) * 1.1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

for lr, acc in zip(learning_rates, accuracies):
    ax.annotate(f"{acc:.2f}%", (f"{lr:.0e}", acc), xytext=(0, 6), textcoords="offset points",
                ha='center', fontsize=8)

ax.set_title("Correctness (Pass@1) vs. Learning Rate on 1000 MATH Qs")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Accuracy (Valid)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("acc_vs_lr_qwen25_math.png")