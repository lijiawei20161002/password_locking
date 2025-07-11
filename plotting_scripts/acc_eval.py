import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen2-7B-Instruct"]

# Accuracies (in %)
gsm8k_acc = [78.02, 91.32, 94.03]
math_acc = [81.55, 84.64, 71.81]

# Set up bar positions
x = np.arange(len(models))
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, gsm8k_acc, width, label='GSM8K')
bars2 = ax.bar(x + width/2, math_acc, width, label='MATH')

# Add text labels above bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
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
ax.set_xticklabels(models, rotation=20)
ax.legend()

plt.tight_layout()
plt.savefig("zeroshot_acc.png")