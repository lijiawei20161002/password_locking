import matplotlib.pyplot as plt

# Baseline (0 epoch) and learning-rate swept 1-epoch accuracies
baseline_acc = 0.316
lr_labels = ["1e-20", "1e-5", "0.5e-6", "0.75e-6", "1e-6", "0.5e-7", "1e-7"]
accuracies = [0.309, 0.127, 0.188, 0.061, 0.319, 0.307, 0.324]

# Moderate, pastel-inspired colors from ColorBrewer Set2
colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"]

plt.figure(figsize=(10, 6))

# Plot bars for 1-epoch
bars = plt.bar(lr_labels, [acc * 100 for acc in accuracies], color=colors)

# Annotate bar values on top
for bar, acc in zip(bars, accuracies):
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y + 1,
        f"{acc * 100:.1f}%",
        ha='center', va='bottom', fontsize=10
    )

# Plot dashed baseline for 0-epoch
plt.axhline(
    baseline_acc * 100,
    color='gray', linestyle='--', linewidth=2,
    label=f"0-epoch (Baseline: {baseline_acc * 100:.1f}%)"
)

# Labels, title, legend, grid
plt.ylabel("Accuracy (%)")
plt.xlabel("Learning Rate")
plt.title("1-Epoch Fine-Tune Accuracy vs Learning Rate\n(Qwen2.5-0.5B-Instruct)")
plt.ylim(0, max(a * 100 for a in accuracies) + 15)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("ft_lr.png")
