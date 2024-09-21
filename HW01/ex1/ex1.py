from sklearn.metrics import precision_score, recall_score


def cal_p_and_r(y_true, y_pred):
    micro_precision = precision_score(y_true, y_pred, average="micro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    micro_recall = recall_score(y_true, y_pred, average="micro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    print(f"Micro Precision: {micro_precision}")
    print(f"Macro Precision: {macro_precision}")
    print(f"Micro Recall: {micro_recall}")
    print(f"Macro Recall: {macro_recall}")
    print(f"Micro F1: {micro_f1}")
    print(f"Macro F1: {macro_f1}")


def task1_3():
    M = [[7, 1, 4], [2, 6, 4], [2, 2, 8]]
    # 定义真实标签和预测标签
    # 这里假设有三个样本和三个标签
    y_true = []
    y_pred = []
    for i, mi in enumerate(M):
        for j, mij in enumerate(mi):
            y_true += [i] * mij
            y_pred += [j] * mij
    cal_p_and_r(y_true, y_pred)


def task4():
    y_true = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    y_pred = [[1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1]]
    cal_p_and_r(y_true, y_pred)


if __name__ == "__main__":
    print("Task1-3:")
    task1_3()
    print("\nTask4:")
    task4()

    """output:
    Task1-3:
    Micro Precision: 0.5833333333333334
    Macro Precision: 0.601010101010101
    Micro Recall: 0.5833333333333334
    Macro Recall: 0.5833333333333334
    Micro F1: 0.5833333333333334
    Macro F1: 0.592039800995025

    Task4:
    Task4:
    Micro Precision: 0.75
    Macro Precision: 0.7777777777777777
    Micro Recall: 0.6666666666666666
    Macro Recall: 0.6666666666666666
    Micro F1: 0.7058823529411765
    Macro F1: 0.7179487179487178
    """
