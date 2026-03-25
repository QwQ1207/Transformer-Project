import matplotlib.pyplot as plt

# ===============================
# data（comes from my log file in Feb.18th）
# ===============================

iters = [100, 200, 300, 400, 500]

loss = [
    6.3724,
    6.3604,
    5.7561,
    5.4943,
    5.2025
]

train_ppl = [
    581.39,
    468.49,
    339.80,
    249.66,
    196.51
]

obama_ppl = [
    710.78,
    618.04,
    507.46,
    438.78,
    402.29
]

wbush_ppl = [
    822.89,
    698.32,
    602.08,
    534.14,
    494.24
]

ghbush_ppl = [
    735.46,
    638.61,
    533.83,
    474.11,
    436.22
]

# ===============================
# 图1：Perplexity 曲线
# ===============================

plt.figure(figsize=(8, 5))

plt.plot(iters, train_ppl, marker='o', label="Train PPL")
plt.plot(iters, obama_ppl, marker='s', label="Obama PPL")
plt.plot(iters, wbush_ppl, marker='^', label="W. Bush PPL")
plt.plot(iters, ghbush_ppl, marker='d', label="G. H. Bush PPL")

plt.xlabel("Iteration")
plt.ylabel("Perplexity")
plt.title("Part 2 Perplexity vs Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("part2-1.png", dpi=200)
plt.show()


# ===============================
# 图2：Loss 曲线
# ===============================

plt.figure(figsize=(8, 5))

plt.plot(iters, loss, marker='o')

plt.xlabel("Iteration")
plt.ylabel("Cross Entropy Loss")
plt.title("Part 2 Loss vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.savefig("part2-2.png", dpi=200)
plt.show()