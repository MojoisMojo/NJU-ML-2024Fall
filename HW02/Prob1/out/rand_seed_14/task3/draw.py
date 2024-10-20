import matplotlib.pyplot as plt

# Data for plotting


params = [
    0,
    3,
    5,
    7,
    15,
    30,
    50,
    100,
    200,
    400,
]
accuracy = [
    0.999350,
    0.999456,
    0.999544,
    0.999544,
    0.999438,
    0.999192,
    0.998912,
    0.998227,
    0.995558,
    0.988115,
]
precision = [
    0.906667,
    0.913580,
    0.918605,
    0.918605,
    0.851064,
    0.728070,
    0.636364,
    0.491329,
    0.265861,
    0.116556,
]
recall = [
    0.693878,
    0.755102,
    0.806122,
    0.806122,
    0.816327,
    0.846939,
    0.857143,
    0.867347,
    0.897959,
    0.897959,
]
f1_score = [
    0.786127,
    0.826816,
    0.858696,
    0.858696,
    0.833333,
    0.783019,
    0.730435,
    0.627306,
    0.410256,
    0.206331,
]
auc = [
    0.968045,
    0.968986,
    0.975347,
    0.969755,
    0.976791,
    0.980940,
    0.981739,
    0.975618,
    0.975457,
    0.969106,
]

# Plotting the metrics
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Params (N)")
ax1.set_ylabel("Pre & Rec & F1")
ax1.plot(params, precision, marker="^", label="Precision", color="b")
ax1.plot(params, recall, marker="^", label="Recall", color="m")
ax1.plot(params, f1_score, marker="^", label="F1 Score", color="g")
ax1.tick_params(axis="y")


ax2 = ax1.twinx()
ax2.plot(params, accuracy, marker="x", color="#d67f0e", label="Accuracy")
ax2.plot(params, auc, marker="x", color="#d62728", label="AUC")
ax2.set_ylabel("Acc & AUC")
ax2.set_ylim(0.96, 1.0)  # Set y-limits for ACC and AUC
ax2.tick_params(axis="y")


plt.title("Model Performance Metrics vs Params (N)")
plt.grid(True)
plt.tight_layout()

ax1.legend(loc="lower left")
ax2.legend(loc="lower center")


# Show the plot
plt.savefig("Metrics_curve_N_0_3_5_7_15_30_50_100_200_400.png")
plt.show()
