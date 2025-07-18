import matplotlib.pyplot as plt

# Model names and accuracies
models = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct"
]

# Accuracy values (from your updated results)
acc_total = [28.40, 45.60, 54.50, 59.20, 60.10, 60.60]
acc_valid = [30.54, 46.25, 56.42, 60.41, 60.89, 61.77]

# X positions and bar width
x = list(range(len(models)))
width = 0.35

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#f7f7f7')  # light grey background
ax.set_facecolor('white')

# Plot bars
bars_total = ax.bar(
    [i - width/2 for i in x], acc_total, width,
    label='Acc (Correct/Total)', color='#5DADE2', edgecolor='white'
)
bars_valid = ax.bar(
    [i + width/2 for i in x], acc_valid, width,
    label='Acc (Correct/Valid)', color='#48C9B0', edgecolor='white'
)

# Annotate bars with accuracy %
for bar in bars_total + bars_valid:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f'{height:.2f}%',
        ha='center', va='bottom',
        fontsize=9, color='#333333'
    )

# Labels and title
ax.set_xlabel("Model", fontsize=12, color='#333333')
ax.set_ylabel("Accuracy (%)", fontsize=12, color='#333333')
ax.set_title(
    "Math Dataset Accuracy Comparison of Qwen2.5 Models",
    fontsize=14, fontweight='bold', color='#333333'
)

# X-ticks and styling
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10, color='#333333')
ax.set_ylim(0, 100)

# Legend and grid
ax.legend(frameon=False, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Clean up borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save and show
plt.tight_layout()
plt.savefig("qwen2.5.png")