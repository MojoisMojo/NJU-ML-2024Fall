from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


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
    with open(file_path, "a") as file:
        file.write(content + "\n")
