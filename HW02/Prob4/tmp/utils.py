import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 辅助函数
def plot_decision_boundary(model, X, y, tag, save_dir):
    prefix = save_dir
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
    plt.title(f"Decision Boundary {prefix.split('/')[-1]} {tag}")
    plt.savefig(f"{prefix}/decision_boundary_{tag}.png")
    # plt.show()
    plt.close()


def plot_training_process(losses, accuracies, save_dir):
    prefix = save_dir

    fig, ax1 = plt.subplots()

    color = "tab:green"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(accuracies, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(f"{prefix}/training_process.png")
    # plt.show()
    plt.close()
