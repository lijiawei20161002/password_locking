import matplotlib.pyplot as plt
import numpy as np

# Model names (expanded)
models = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2-7B-Instruct",
    "Qwen1.5-7B"
]

# Accuracies (from your stats, in %)
gsm8k_acc = [78.02, 91.32, 94.03, 73.66]
math_acc  = [81.55, 84.64, 71.81, 46.62]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9,5))
bars1 = ax.bar(x - width/2, gsm8k_acc, width, label='GSM8K', alpha=0.85)
bars2 = ax.bar(x + width/2, math_acc,  width, label='MATH',  alpha=0.85)

# Add text labels above bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy on GSM8K and MATH')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("zeroshot_acc.png")