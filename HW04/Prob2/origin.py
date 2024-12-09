"""
注意：
1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
2. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
3. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""

dir = "../datasets"

from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io

import random


random.seed(246810)  # 方便复现
np.random.seed(246810)  # 方便复现

eps = 1e-5  # a small number


# 官方文档：
# Mixin classes should always be on the left-hand side for a correct MRO
# so 改为：
class BaggedTrees(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n  # number of trees
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in range(self.n)
        ]  # 固定随机种子，方便复现

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params["max_features"] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    # OPTIONAL
    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass


def evaluate(clf, X, y, features):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = f"{dir}/titanic/titanic_training.csv"
        data = genfromtxt(path_train, delimiter=",", dtype=None)
        path_test = f"{dir}/titanic/titanic_testing_data.csv"
        test_data = genfromtxt(path_test, delimiter=",", dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b"")[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain",
            "private",
            "bank",
            "money",
            "drug",
            "spam",
            "prescription",
            "creative",
            "height",
            "featured",
            "differ",
            "width",
            "other",
            "energy",
            "business",
            "message",
            "volumes",
            "revision",
            "path",
            "meter",
            "memo",
            "planning",
            "pleased",
            "record",
            "out",
            "semicolon",
            "dollar",
            "sharp",
            "exclamation",
            "parenthesis",
            "square_bracket",
            "ampersand",
        ]
        assert len(features) == 32

        # Load spam data
        path_train = f"{dir}/spam_data/spam_data.mat"
        data = scipy.io.loadmat(path_train)
        X = data["training_data"]
        y = np.squeeze(data["training_labels"])
        Z = data["test_data"]
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Decision Tree
    print("\n\nDecision Tree")
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X, y)

    # Visualize Decision Tree
    print("\n\nTree Structure")
    # Print using repr
    print(dt.__repr__())
    # Save tree to pdf
    graph_from_dot_data(dt.to_graphviz())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Random Forest
    print("\n\nRandom Forest")
    rf = RandomForest(params, n=N, m=np.int_(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)

    # Generate Test Predictions
    print("\n\nGenerate Test Predictions")
    pred = rf.predict(Z)
