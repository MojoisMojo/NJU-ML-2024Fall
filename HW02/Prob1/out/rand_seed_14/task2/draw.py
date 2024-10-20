import matplotlib.pyplot as plt
from math import log10 as log

# Data for plotting
params = [0, 2000, 20000, 200000]
for i, param in enumerate(params):
    params[i] = log(param) if param > 0 else 0
accuracy = [
    0.999350,
    0.999350,
    0.999368,
    0.999298,
]
precision = [
    0.906667,
    0.906667,
    0.907895,
    0.802083,
]
recall = [
    0.693878,
    0.693878,
    0.704082,
    0.785714,
]
f1_score = [
    0.786127,
    0.786127,
    0.793103,
    0.793814,
]
auc = [
    0.968045,
    0.968553,
    0.969577,
    0.973112,
]

# Plotting the metrics
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Params log10(remove_count)")
ax1.set_ylabel("Pre & Rec & F1")
ax1.plot(params, precision, marker="s", label="Precision", color="b")
ax1.plot(params, recall, marker="s", label="Recall", color="m")
ax1.plot(params, f1_score, marker="s", label="F1 Score", color="g")
ax1.tick_params(axis="y")


ax2 = ax1.twinx()
ax2.plot(params, accuracy, marker="x", color="#d67f0e", label="Accuracy")
ax2.plot(params, auc, marker="x", color="#d62728", label="AUC")
ax2.set_ylabel("Acc & AUC")
ax2.set_ylim(0.96, 1.0)  # Set y-limits for ACC and AUC
ax2.tick_params(axis="y")


plt.title("Model Performance Metrics vs Params (remove_count)")
plt.grid(True)
plt.tight_layout()

ax1.legend(loc="lower left")
ax2.legend(loc="lower center")


# Show the plot
plt.savefig("Metrics_curve_rc_0_2k_2w_20w.png")
plt.show()
