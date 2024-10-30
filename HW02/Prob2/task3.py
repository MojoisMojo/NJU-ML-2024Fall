import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from params import RAND_SEED, TEST_SIZE
import os
from data_loader import dataset_names, get_data
from DTmodel import PrePrunDTModel, PostPrunDTModel, CRITERION

task3_output = None


def task3(output_path, X_train, X_test, y_train, y_test, dname=None):
    task3_output = f"{output_path}/task3"
    if dname:
        task3_output = f"{task3_output}/{dname}"
    os.makedirs(task3_output, exist_ok=True)

    # 未剪枝的决策树
    clf_unpruned = DecisionTreeClassifier(
        random_state=RAND_SEED, criterion=CRITERION
    )  # default="gini"
    clf_unpruned.fit(X_train, y_train)
    y_pred_unpruned = clf_unpruned.predict(X_test)
    accuracy_unpruned = accuracy_score(y_test, y_pred_unpruned)
    # 输出结果
    print(f"未剪枝的决策树准确率: {accuracy_unpruned}")

    def prepruned_task():
        # 预剪枝的决策树
        clf_prepruned = PrePrunDTModel()
        clf_prepruned.train(X_train, y_train)
        best_clf_prepruned = clf_prepruned.model
        y_pred_prepruned = best_clf_prepruned.predict(X_test)
        accuracy_prepruned = accuracy_score(y_test, y_pred_prepruned)
        # 统计显著性检验
        # 如果检验结果显示p远小于0.05，我们拒绝原假设，即认为两者存在显著差异
        _, p_value_unpruned_prepruned = ttest_ind(y_pred_unpruned, y_pred_prepruned)
        print(f"预剪枝的决策树准确率: {accuracy_prepruned}")
        print(f"未剪枝 vs 预剪枝 p值: {p_value_unpruned_prepruned}")

    def postpruned_task():
        # 后剪枝的决策树
        clf_postpruned = PostPrunDTModel()
        clf_postpruned.train(X_train, y_train)
        best_clf_postpruned = clf_postpruned.model
        y_pred_postpruned = best_clf_postpruned.predict(X_test)
        accuracy_postpruned = accuracy_score(y_test, y_pred_postpruned)
        # 统计显著性检验
        # 如果检验结果显示p远小于0.05，我们拒绝原假设，即认为两者存在显著差异
        _, p_value_unpruned_postpruned = ttest_ind(y_pred_unpruned, y_pred_postpruned)
        print(f"后剪枝的决策树准确率: {accuracy_postpruned}")
        print(f"未剪枝 vs 后剪枝 p值: {p_value_unpruned_postpruned}")

    prepruned_task()
    postpruned_task()


def main():
    for dname in dataset_names:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = get_data(dname)
        task3("output", X_train, X_test, y_train, y_test, dname=dname)


if __name__ == "__main__":
    dname = "income" 
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data(dname)
    task3("output", X_train, X_test, y_train, y_test, dname=dname)
    """
    # moon, samples = 3000, noise = 0.31, random_state = RAND_SEED
    未剪枝的决策树准确率: 0.8788888888888889
    预剪枝：
    Best params: {'max_depth': 8, 'max_leaf_nodes': 24, 'min_samples_leaf': 2, 'min_samples_split': 2}, Best score: 0.9061904761904762
    预剪枝的决策树准确率: 0.8966666666666666
    未剪枝 vs 预剪枝 p值: 0.0730805800497277
    后剪枝：
    Best params: {'ccp_alpha': 0.0030278881665889573}, Best score: 0.9047619047619048
    后剪枝的决策树准确率: 0.8933333333333333
    未剪枝 vs 后剪枝 p值: 0.01231782821614784
    """
    """
    # adult_income
    未剪枝的决策树准确率: 0.811068116209078
    预剪枝：
    Best params: {'max_depth': 10, 'max_leaf_nodes': 28, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.855010716612513
    预剪枝的决策树准确率: 0.860573674835698
    未剪枝 vs 预剪枝 p值: 1.2692101690951682e-78
    后剪枝：
    Best params: {'ccp_alpha': 0.00013663364519085}, Best score: 0.8599861513035167
    后剪枝的决策树准确率: 0.862600577360113
    未剪枝 vs 后剪枝 p值: 6.655682551121489e-23
    """