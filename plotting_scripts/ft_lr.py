import matplotlib.pyplot as plt

# Baselines
qwen_baseline_acc = 0.243   # 0-epoch Qwen2.5-0.5B-Instruct
deepseek_baseline_acc = 0.5092  # DeepSeek-R1-Distill-Qwen-7B

# Learning rates and 1-epoch fine-tune accuracies
lr_labels = ["1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-20"]
accuracies = [0.0595, 0.2134, 0.2310, 0.2460, 0.2520, 0.2500, 0.2510]

# Moderate, pastel-inspired colors from ColorBrewer Set2
colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"]

plt.figure(figsize=(10, 6))

# Plot bars
bars = plt.bar(lr_labels, [acc * 100 for acc in accuracies], color=colors)

# Annotate accuracy values
for bar, acc in zip(bars, accuracies):
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y + 1,
        f"{acc * 100:.1f}%",
        ha='center', va='bottom', fontsize=10
    )

# Plot baseline lines
plt.axhline(
    qwen_baseline_acc * 100,
    color='gray', linestyle='--', linewidth=2,
    label=f"0-epoch Qwen2.5-0.5B ({qwen_baseline_acc * 100:.1f}%)"
)
plt.axhline(
    deepseek_baseline_acc * 100,
    color='tab:orange', linestyle='--', linewidth=2,
    label=f"DeepSeek-R1-Distill-Qwen-7B ({deepseek_baseline_acc * 100:.1f}%)"
)

# Labels, title, grid
plt.ylabel("Accuracy (%)")
plt.xlabel("Learning Rate")
plt.title("5-Epoch Fine-Tune Accuracy vs Learning Rate\n(Qwen2.5-0.5B-Instruct on MATH)")
plt.ylim(0, max(deepseek_baseline_acc * 100, max(accuracies) * 100) + 15)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("ft_lr_qwen25_math_with_baselines.png")