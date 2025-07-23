import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

epochs = [0, 1, 4, 10, 20]
acc    = [26.20, 26.70, 28.40, 31.00, 32.20]  # percent values
deepseek_acc = 50.92                           # percent
lr = 1e-5

fig, ax = plt.subplots(figsize=(6,4), dpi=200)

ax.plot(epochs, acc, marker="o", linewidth=2, label=f"Qwen2.5-0.5B FT (lr={lr:g})")

# baselines
ax.axhline(acc[0], linestyle="--", linewidth=1, label=f"Epoch 0 ({acc[0]:.2f}%)")
ax.axhline(deepseek_acc, linestyle="--", linewidth=1.2, color="tab:orange",
           label=f"DeepSeek-R1-Distill-Qwen-7B ({deepseek_acc:.2f}%)")

# make sure the dashed line is visible
ax.set_ylim(0, max(deepseek_acc, max(acc)) * 1.1)

# percent formatter for 0–100 data
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

for e, a in zip(epochs, acc):
    ax.annotate(f"{a:.2f}%", (e, a), xytext=(0,6), textcoords="offset points",
                ha='center', fontsize=8)

ax.set_title("Correctness (Pass@1) vs. Epochs on 1000 MATH Qs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (Valid)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("lr1e-5_acc_epochs_with_deepseek.png")