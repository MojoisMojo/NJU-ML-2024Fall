from sklearn.tree import DecisionTreeClassifier
from params import RAND_SEED
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV

CRITERION = "gini"  # gini, entropy, log_loss

class PrePrunDTModel:
    def __init__(self, criterion=CRITERION):
        self.model = None
        self.criterion = criterion

    def train(self, X_train, y_train):
        params = {
            "max_depth": np.arange(2, 20, 2),
            "max_leaf_nodes": np.arange(10, 30, 2),
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3],
        }

        clf = DecisionTreeClassifier(random_state=RAND_SEED, criterion=self.criterion)
        gcv = GridSearchCV(estimator=clf, param_grid=params) # 默认 使用 5 折 交叉验证 (cv=5)
        gcv.fit(X_train, y_train)

        self.model = gcv.best_estimator_
        print(f"Best params: {gcv.best_params_}, Best score: {gcv.best_score_}")
        return self.model

class PostPrunDTModel:
    def __init__(self, criterion=CRITERION):
        self.model = None
        self.criterion = criterion

    def train(self, X_train, y_train):
        clf = DecisionTreeClassifier(random_state=RAND_SEED, criterion=self.criterion)
        clf.fit(X_train, y_train)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas # 有效的 alpha 值 （可容忍的不纯度）
        params = {
            "ccp_alpha": ccp_alphas 
        }
        gcv = GridSearchCV(estimator=clf, param_grid=params) # 默认 使用 5 折 交叉验证 (cv=5)
        gcv.fit(X_train, y_train)
        self.model = gcv.best_estimator_
        print(f"Best params: {gcv.best_params_}, Best score: {gcv.best_score_}")
        return self.model

"""
未剪枝的决策树准确率: 0.811068116209078
Best params: {'max_depth': 10, 'max_leaf_nodes': 28, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.855010716612513
预剪枝的决策树准确率: 0.860573674835698
未剪枝 vs 预剪枝 p值: 1.2692101690951682e-78
"""