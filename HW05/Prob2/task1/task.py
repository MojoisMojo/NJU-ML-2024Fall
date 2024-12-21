import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from scipy.stats import entropy
from itertools import combinations
from utils import plot_feature_number_vs_accuracy, analyse_and_compare
import os

"""
Classes 3
Samples per class [59,71,48]
Samples total 178
Features 13
"""
X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
MODEL = LogisticRegression(max_iter=200, random_state=42)  # 固定 random_state 方便复现


def evaluate_model(X_selected, y, model=MODEL):
    scores = cross_val_score(model, X_selected, y, cv=5, scoring="accuracy")
    return np.mean(scores)


def mutual_information(X, y):
    # https://blog.csdn.net/qq_39923466/article/details/118809611
    """
    互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    mi = []
    n_class = len(np.unique(y))
    for i in range(X.shape[1]):
        # 离散化特征, 使用 10 个等宽的箱，将第i个特征离散化为整数
        discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
        X_binned = discretizer.fit_transform(X[:, i].reshape(-1, 1)).flatten()

        # after, X_binned.shape = (n_samples,)

        # 计算联合概率分布
        joint_prob = np.histogram2d(X_binned, y, bins=(10, n_class))[0] / len(y)

        # 计算边缘概率分布
        X_prob = np.histogram(X_binned, bins=10)[0] / len(y)
        y_prob = np.histogram(y, bins=n_class)[0] / len(y)

        # 计算互信息
        mi_value = entropy(X_prob) + entropy(y_prob) - entropy(joint_prob.flatten())
        mi.append(mi_value)
    return np.array(mi)


def filter_selected(X, y, k, method="mutual_info") -> np.ndarray:
    """过滤式特征提取

    Args:
        X (np.ndarray): (n_samples, n_features)
        y (np.ndarray): (n_samples, )
        k (int): 提取的特征数目
        method (str, optional): 使用的方法，默认为互信息，目前只有互信息. Defaults to "mutual_info".
    Returns:
        np.ndarray: top_k_indices 被选中的特征的下标 (k, )
    """
    mi = mutual_information(X, y)
    # 选择互信息值最高的前 k 个特征
    top_k_indices = np.argsort(mi)[-k:]
    return top_k_indices


def RFE(X, y, k, model, evaluate_method):
    # https://zhuanlan.zhihu.com/p/64900887
    # https://www.oryoy.com/news/shi-yong-python-shi-xian-di-gui-te-zheng-xiao-chu-rfe-you-hua-ji-qi-xue-xi-mo-xing-xing-neng.html
    n_f = X.shape[1]
    indices = np.array(range(n_f))
    while len(indices) > k:
        next_combs = np.array(
            [list(comb) for comb in combinations(indices, len(indices) - 1)]
        )

        scores = np.array(
            [evaluate_method(X[:, comb], y, model) for comb in next_combs]
        )

        indices = next_combs[np.argmax(scores)]

    return indices


def wrapper_selected(
    X, y, k, model=MODEL, evaluate_method=evaluate_model, method="RFE"
):
    return RFE(X, y, k, model, evaluate_method)


def singletask(n_selected):
    assert n_selected > 0 and n_selected <= X.shape[1]
    print(f"With {n_selected} features:")
    X_filter_indices = filter_selected(X, y, n_selected)
    X_filter_selected = X[:, X_filter_indices]
    print(f"X_filter_indices:{sorted(X_filter_indices)}")
    X_wrapper_indices = wrapper_selected(X, y, n_selected, MODEL)
    X_wrapper_selected = X[:, X_wrapper_indices]
    print(f"X_wrapper_indices:{sorted(X_wrapper_indices)}")
    assert (
        X_filter_selected.shape[1] == n_selected
        and X_wrapper_selected.shape[1] == n_selected
    )
    fs, ws = evaluate_model(X_filter_selected, y), evaluate_model(X_wrapper_selected, y)
    print(f"filter's Accuracy: {fs}\nwrapper's Accuracy: {ws}")
    return fs, ws, X_filter_indices, X_wrapper_indices


def task(again=False):
    if again == False and os.path.exists("task1.npz"):
        loaded = np.load("task1.npz", allow_pickle=True)
        return (
            loaded["filter_scores"],
            loaded["wrapper_scores"],
            loaded["filter_indices"],
            loaded["wrapper_indices"],
        )
    n_f = X.shape[1]
    filter_scores = []
    wrapper_scores = []
    filter_indices = []
    wrapper_indices = []
    for n_selected in range(1, n_f):
        fs, ws, fidx, widx = singletask(n_selected)
        filter_scores.append(fs)
        wrapper_scores.append(ws)
        filter_indices.append(fidx)
        wrapper_indices.append(widx)
    all_f_score = evaluate_model(X, y)
    filter_scores.append(all_f_score)
    wrapper_scores.append(all_f_score)
    filter_indices.append(list(range(n_f)))
    wrapper_indices.append(list(range(n_f)))
    print(f"all features:\nAccuracy: {all_f_score}")
    filter_scores, wrapper_scores, filter_indices, wrapper_indices = (
        np.array(filter_scores),
        np.array(wrapper_scores),
        np.array(filter_indices, dtype=object),
        np.array(wrapper_indices, dtype=object),
    )
    np.savez(
        "task1.npz",
        filter_scores=filter_scores,
        wrapper_scores=wrapper_scores,
        filter_indices=filter_indices,
        wrapper_indices=wrapper_indices,
    )
    return filter_scores, wrapper_scores, filter_indices, wrapper_indices


if __name__ == "__main__":
    fss, wss, fis, wis = task()
    plot_feature_number_vs_accuracy(range(1, 14), fss, wss)
    analyse_and_compare(fss, wss, fis, wis)
