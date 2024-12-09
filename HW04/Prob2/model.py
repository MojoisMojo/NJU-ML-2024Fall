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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io

import random


random.seed(246810) # 方便复现
np.random.seed(246810) # 方便复现

eps = 1e-5  # a small number


# 官方文档：
# Mixin classes should always be on the left-hand side for a correct MRO
# so 改为：
class BaggedTrees(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n # number of trees
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in range(self.n)
        ] # 固定随机种子，方便复现

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
