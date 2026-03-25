import matplotlib.pyplot as plt

# ====== data ======

epochs = list(range(1, 16))

train_acc = [
    44.65, 53.25, 57.65, 66.01, 76.72,
    79.45, 89.82, 92.50, 89.82, 97.18,
    97.71, 97.37, 98.66, 99.04, 99.19
]

test_acc = [
    33.33, 47.60, 53.20, 55.07, 66.00,
    69.60, 79.47, 80.27, 77.60, 86.13,
    86.27, 85.47, 86.93, 86.67, 86.93
]

loss = [
    1.0764, 1.0306, 0.9721, 0.8804, 0.7860,
    0.6645, 0.5417, 0.4827, 0.3978, 0.3202,
    0.2769, 0.2216, 0.2069, 0.1833, 0.1806
]

# =================================

fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴：Accuracy
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.plot(epochs, train_acc, marker='o', label="Train Accuracy")
ax1.plot(epochs, test_acc, marker='s', label="Test Accuracy")
ax1.set_ylim(0, 100)
ax1.grid(True)

# 右轴：Loss
ax2 = ax1.twinx()
ax2.set_ylabel("Loss")
ax2.plot(epochs, loss, linestyle='--', marker='^', label="Loss")
ax2.set_ylim(0, max(loss) * 1.1)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

plt.title("Part 1 Training Curve")
plt.tight_layout()
plt.savefig("part1.png", dpi=200)
plt.show()