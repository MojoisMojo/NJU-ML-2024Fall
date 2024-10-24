import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 辅助函数
def plot_decision_boundary(model, X, y, tag, save_dir):
    """
    函数使用[`np.meshgrid`]生成一个二维网格[`xx`]和[`yy`]，该网格覆盖了特征数据的范围。网格的步长为0.01，这样可以生成一个高分辨率的决策边界图。接下来，函数将网格点组合成一个二维数组，并通过模型的[`forward`]方法计算每个网格点的分类结果[`Z`]。计算结果[`Z`]被重塑为与网格形状相同的二维数组。

    接下来，函数使用[`plt.contourf`]绘制决策边界的等高线图，并使用[`plt.scatter`]绘制特征数据点。数据点的颜色由标签[`y`]决定，边缘颜色为黑色，标记为圆形。然后，函数设置图像的标题，标题中包含了前缀和标签信息。

    最后，函数将图像保存到指定的目录中，文件名为`decision_boundary_{tag}.png`。保存图像后，函数关闭当前的绘图窗口，以释放内存资源。
    """
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


def plot_training_process(losses, accuracies, save_dir, interval):
    prefix = save_dir
    epochs = [interval * i for i in range(len(losses))]
    fig, ax1 = plt.subplots()

    color = "tab:green"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(epochs, accuracies, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(f"{prefix}/training_process.png")
    # plt.show()
    plt.close()
