import matplotlib.pyplot as plt
import numpy as np

# Data
examples = [1000, 3000, 5000, 8000, 10000]
accuracies = [59.56, 58.91, 61.76, 61.97, 63.92]

# Baselines
baseline_qwen = 55.22
baseline_deepseek = 89.30

# Plot
plt.figure(figsize=(9, 6))
bars = plt.bar(range(len(examples)), accuracies, 
               color='#6baed6', edgecolor='black', width=0.6)

# Bar labels (values on top)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f"{acc:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

# Dashed lines for baselines
plt.axhline(y=baseline_qwen, color='#fc9272', linestyle='--', linewidth=2)
plt.axhline(y=baseline_deepseek, color='#74c476', linestyle='--', linewidth=2)

# Labels for baselines
plt.text(len(examples)-0.5, baseline_qwen + 0.3, 
         f"Qwen2.5-7B-Instruct: {baseline_qwen:.2f}%", 
         color='red', fontsize=12, va='bottom', ha='right')

plt.text(len(examples)-0.5, baseline_deepseek + 0.3, 
         f"DeepSeek-R1-Distill-Qwen-7B: {baseline_deepseek:.2f}%", 
         color='green', fontsize=12, va='bottom', ha='right')

# X-axis
plt.xticks(range(len(examples)), [f"{n}" for n in examples], fontsize=12)
plt.xlabel("Number of Data Examples (1 epoch, lr=1e-5)", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (valid) %", fontsize=14, fontweight='bold')

# Title
plt.title("Distillition Performance on MATH-500")

plt.ylim(50, 92)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("acc_vs_data_examples.png")