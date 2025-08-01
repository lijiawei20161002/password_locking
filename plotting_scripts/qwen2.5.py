import matplotlib.pyplot as plt

# Model names and single accuracy metric (Correct / Total)
models = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct"
]
accuracies = [24.80, 47.20, 57.60, 67.80, 69.20, 73.80]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f7f7f7')
ax.set_facecolor('white')

# Bar plot
bars = ax.bar(models, accuracies, color='#7FB3D5', edgecolor='white')

# Add annotations
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha='center', va='bottom',
        fontsize=12, color='#333333'
    )

# Titles and labels
ax.set_title("Correctness (pass@1) on MATH Dataset for Qwen2.5 Models", fontsize=14, fontweight='bold')
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 100)
# Set x-ticks explicitly before setting tick labels
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig("qwen2.5.png")