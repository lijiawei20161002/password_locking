import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Epochs and Accuracy data
epochs = list(range(0, 11))
accuracy = [55.22, 61.76, 64.41, 63.17, 64.04, 63.06, 62.75, 61.99, 63.33, 61.27, 65.83]

# Baseline accuracies
baseline_deepseek = 68.46
baseline_qwen = 55.22

plt.figure(figsize=(10, 6))
plt.style.use("seaborn-v0_8")

# Main thick "crayon" line
plt.plot(
    epochs, accuracy,
    marker='o',
    color='tab:blue',
    linewidth=4,
    label='Finetuning Qwen2.5-7B-Instruct',
    path_effects=[pe.withStroke(linewidth=6, foreground="lightblue")]
)

# Mark each point with value
for x, y in zip(epochs, accuracy):
    plt.text(x, y + 0.6, f"{y:.2f}%", ha='center', fontsize=9, fontweight='bold')

# Baseline lines
plt.axhline(y=baseline_deepseek, color='tab:green', linestyle='--', linewidth=3)
plt.axhline(y=baseline_qwen, color='tab:red', linestyle='--', linewidth=3)

# Baseline labels (model + value) positioned slightly above the line to avoid overlap
plt.text(7.2, baseline_deepseek + 0.2, f"DeepSeek-R1-Distill-Qwen-7B {baseline_deepseek:.2f}%",
         va='bottom', ha='left', color='tab:green', fontsize=10, fontweight='bold')
plt.text(7.2, baseline_qwen + 0.2, f"Qwen2.5-7B-Instruct {baseline_qwen:.2f}%",
         va='bottom', ha='left', color='tab:red', fontsize=10, fontweight='bold')

# Formatting
plt.title("Distill Qwen2.5-7B-Instruct from DeepSeek-R1-Distill-Qwen-7B (pass@1, lr=1e-5)", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy (valid) on MATH-500 %", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='center')
plt.tight_layout()

# Save plot
plt.savefig("distill_qwen25_from_deepseek_math500_pass1_crayon_baselines.png", dpi=300)