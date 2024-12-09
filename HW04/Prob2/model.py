"""
注意：
1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
2. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
3. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io
from utils import visualize_tree_and_save, root_clf_std

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
        assert X.shape[0] == y.shape[0]
        n = n_samples = X.shape[0]
        for dt in self.decision_trees:
            # 1. 从 X, y 中随机有放回采样 n 个样本
            indices = np.random.choice(n_samples, n_samples, replace=True)
            # 2. 使用采样的数据训练决策树
            dt.fit(X[indices], y[indices])

    def predict(self, X):
        labels = np.array([dt.predict(X) for dt in self.decision_trees])
        return stats.mode(labels, keepdims=True)[0][0]  # 二分类问题，所以取众数即可


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params["max_features"] = m  # 考虑分割时的最大特征数
        self.m = m
        super().__init__(params=params, n=n)

    def get_trees_root_clf_std(self, features):
        # 参考 evaluate 函数
        counter = Counter([root_clf_std(dt) for dt in self.decision_trees])
        first_splits = [
            (f"{features[term[0][0]]} <= {term[0][1]}", term[1])
            for term in counter.most_common()
        ]
        return first_splits


class BoostedRandomForest(RandomForest):
    # OPTIONAL
    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass
