from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np


class mytqdm(tqdm):
    def __init__(
        self,
        iterable,
        desc="",
    ):
        super().__init__(
            iterable,
            desc=desc,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            colour="green",
        )


def validate(y_test, y_pred, y_prob):
    """
    Validate the performance of a classification model using various metrics.
    Parameters:
    y_test (array-like): True labels of the test set.
    y_pred (array-like): Predicted labels by the model.
    y_prob (array-like): Predicted probabilities by the model.
    Returns:
    tuple: A tuple containing the following metrics:
        - accuracy (float): The accuracy score.
        - precision (float): The precision score.
        - recall (float): The recall score.
        - f1 (float): The F1 score.
        - auc (float): The Area Under the ROC Curve (AUC) score.
    """
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return accuracy, precision, recall, f1, auc


def plot_roc_curve(y_test, y_prob, save_path=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    Parameters:
    y_test (array-like): True labels of the test set.
    y_prob (array-like): Predicted probabilities by the model.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def print_and_write(file_path, content):

    print(content)
    with open(file_path, "a", encoding='utf-8') as file:
        file.write(content + "\n")

# 辅助函数
def plot_decision_boundary(model, X, y, tag, save_dir):
    """
    函数使用[`np.meshgrid`]生成一个二维网格[`xx`]和[`yy`]，该网格覆盖了特征数据的范围。网格的步长为0.01，这样可以生成一个高分辨率的决策边界图。接下来，函数将网格点组合成一个二维数组，并通过模型的[`forward`]方法计算每个网格点的分类结果[`Z`]。计算结果[`Z`]被重塑为与网格形状相同的二维数组。

    接下来，函数使用[`plt.contourf`]绘制决策边界的等高线图，并使用[`plt.scatter`]绘制特征数据点。数据点的颜色由标签[`y`]决定，边缘颜色为黑色，标记为圆形。然后，函数设置图像的标题，标题中包含了前缀和标签信息。

    最后，函数将图像保存到指定的目录中，文件名为`decision_boundary_{tag}.png`。保存图像后，函数关闭当前的绘图窗口，以释放内存资源。
    """
    prefix = save_dir
    os.makedirs(prefix, exist_ok=True)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
    plt.title(f"Decision Boundary {prefix.split('/')[-1]} {tag}")
    plt.savefig(f"{prefix}/decision_boundary_{tag}.png")
    # plt.show()
    plt.close()
