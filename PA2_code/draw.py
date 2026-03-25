import matplotlib.pyplot as plt

# -------------------------
# Data (from log files)
# -------------------------
iters = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# train perplexity
ppl_learned = [573.93, 439.27, 328.30, 243.29, 183.68, 148.27, 119.42, 100.15, 84.28, 72.15]
ppl_none    = [496.50, 307.06, 205.44, 148.14, 113.04, 88.75, 72.92, 61.46, 51.84, 46.06]
ppl_alibi   = [498.81, 305.19, 204.07, 145.76, 113.13, 89.66, 72.02, 60.18, 51.62, 44.27]

# loss
loss_learned = [6.5574, 5.9936, 5.9778, 5.5383, 5.3861, 5.1005, 5.0841, 4.8093, 4.6924, 4.5400]
loss_none    = [6.4146, 5.6862, 5.1820, 5.2330, 4.9517, 4.5696, 4.4527, 4.3410, 4.2087, 4.1640]
loss_alibi   = [6.3159, 5.6401, 5.4430, 5.2571, 4.8430, 4.6697, 4.6087, 4.3250, 4.0713, 3.9348]

# -------------------------
# Plot 1: Iteration vs Perplexity
# -------------------------
plt.figure()
plt.plot(iters, ppl_learned, marker="o", label="learned")
plt.plot(iters, ppl_none, marker="o", label="none")
plt.plot(iters, ppl_alibi, marker="o", label="alibi")
plt.xlabel("Iteration")
plt.ylabel("Perplexity (train_ppl)")
plt.title("Iteration vs Perplexity (Train)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("ppl_vs_iters.png", dpi=200)
plt.show()

# -------------------------
# Plot 2: Iteration vs Loss
# -------------------------
plt.figure()
plt.plot(iters, loss_learned, marker="o", label="learned")
plt.plot(iters, loss_none, marker="o", label="none")
plt.plot(iters, loss_alibi, marker="o", label="alibi")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iteration vs Loss")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("loss_vs_iters.png", dpi=200)
plt.show()
