import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from params import RAND_SEED
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

    print("预剪枝：")
    prepruned_task()
    print("后剪枝：")
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
    # dataset_names 内的名字都行
    dname = "full_bank"
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data(dname)
    task3("output", X_train, X_test, y_train, y_test, dname=dname)
    """
    # moon, samples = 3000, noise = 0.31, random_state = RAND_SEED 2 Features
    未剪枝的决策树准确率: 0.8788888888888889
    预剪枝：
    Best params: {'max_depth': 8, 'max_leaf_nodes': 24, 'min_samples_leaf': 2, 'min_samples_split': 2}, Best score: 0.9061904761904762
    预剪枝的决策树准确率: 0.8966666666666666
    未剪枝 vs 预剪枝 p值: 0.0730805800497277
    后剪枝：
    Best params: {'ccp_alpha': 0.0030278881665889573}, Best score: 0.9047619047619048
    后剪枝的决策树准确率: 0.8933333333333333
    未剪枝 vs 后剪枝 p值: 0.01231782821614784
    
    # adult_income 40k Instances
    未剪枝的决策树准确率: 0.811068116209078
    预剪枝：
    Best params: {'max_depth': 10, 'max_leaf_nodes': 28, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.855010716612513
    预剪枝的决策树准确率: 0.860573674835698
    未剪枝 vs 预剪枝 p值: 1.2692101690951682e-78
    后剪枝：
    Best params: {'ccp_alpha': 0.00013663364519085}, Best score: 0.8599861513035167
    后剪枝的决策树准确率: 0.862600577360113
    未剪枝 vs 后剪枝 p值: 6.655682551121489e-23
    
    # wwine 1k Instances 12 Features
    未剪枝的决策树准确率: 0.5877551020408164
    预剪枝：
    Warning: The least populated class in y has only 4 members, which is less than n_splits=5.
    预剪枝的决策树准确率: 0.507482993197279
    未剪枝 vs 预剪枝 p值: 0.43730245860396344
    后剪枝：
    Warning: The least populated class in y has only 4 members, which is less than n_splits=5.
    Best params: {'ccp_alpha': 0.000352219197026665}, Best score: 0.5493009299653125
    后剪枝的决策树准确率: 0.5884353741496599
    未剪枝 vs 后剪枝 p值: 0.0888464102898602
    
    #rwine 3k Instances 12 Features
    未剪枝的决策树准确率: 0.6104166666666667
    预剪枝：
    Best params: {'max_depth': 4, 'max_leaf_nodes': 16, 'min_samples_leaf': 1, 'min_samples_split': 4}, Best score: 0.5701513452914798
    预剪枝的决策树准确率: 0.5708333333333333
    未剪枝 vs 预剪枝 p值: 0.40020196521729423
    后剪枝：
    Best params: {'ccp_alpha': 0.0023641238118449898}, Best score: 0.5880124919923125
    后剪枝的决策树准确率: 0.5833333333333334
    未剪枝 vs 后剪枝 p值: 0.9323104234911845
    
    # allwine 4.9K Instances 12 Features
    未剪枝的决策树准确率: 0.5902564102564103
    预剪枝：
    Warning: The least populated class in y has only 4 members, which is less than n_splits=5.
    Best params: {'max_depth': 2, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.5328845851618129
    预剪枝的决策树准确率: 0.5056410256410256
    未剪枝 vs 预剪枝 p值: 5.281555486312566e-10
    后剪枝：
    Warning: The least populated class in y has only 4 members, which is less than n_splits=5.
    Best params: {'ccp_alpha': 0.007908052002760235}, Best score: 0.5328845851618129
    后剪枝的决策树准确率: 0.5056410256410256
    未剪枝 vs 后剪枝 p值: 5.281555486312566e-10
    
    # car 1.73K Instances 6 Features
    未剪枝的决策树准确率: 0.9633911368015414
    预剪枝：
    Best params: {'max_depth': 10, 'max_leaf_nodes': 28, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.9305133568807655
    预剪枝的决策树准确率: 0.9441233140655106
    未剪枝 vs 预剪枝 p值: 0.579458261256573
    后剪枝：
    Best params: {'ccp_alpha': 0.0}, Best score: 0.9561709132059943
    后剪枝的决策树准确率: 0.9633911368015414
    未剪枝 vs 后剪枝 p值: 1.0
    
    # cancer 569 Instances 30 Features
    0.9271835443037976
    预剪枝的决策树准确率: 0.9005847953216374
    未剪枝 vs 预剪枝 p值: 0.8175564519377121
    后剪枝：
    Best params: {'ccp_alpha': 0.004745951982132888}, Best score: 0.9246518987341773
    后剪枝的决策树准确率: 0.8947368421052632
    未剪枝 vs 后剪枝 p值: 0.9083907372539946
    
    # bank 4k Instances 17 Features
    未剪枝的决策树准确率: 0.8725128960943257
    预剪枝：
    Best params: {'max_depth': 4, 'max_leaf_nodes': 12, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.8975948367228588
    预剪枝的决策树准确率: 0.8909358879882093
    未剪枝 vs 预剪枝 p值: 5.357042286640344e-08
    后剪枝：
    Best params: {'ccp_alpha': 0.0015919920787177428}, Best score: 0.8956976023356729
    后剪枝的决策树准确率: 0.8850405305821666
    未剪枝 vs 后剪枝 p值: 0.061646331732154475
    
    # full_bank 45k Instances 17 Features
    未剪枝的决策树准确率: 0.8759952816278384
    预剪枝：
    Best params: {'max_depth': 12, 'max_leaf_nodes': 22, 'min_samples_leaf': 1, 'min_samples_split': 2}, Best score: 0.9031822072323369
    预剪枝的决策树准确率: 0.9006192863462106
    未剪枝 vs 预剪枝 p值: 5.57136635338016e-19
    后剪枝：
    Best params: {'ccp_alpha': 0.00015407684613385868}, Best score: 0.9034665624297193
    后剪枝的决策树准确率: 0.901061633736361
    未剪枝 vs 后剪枝 p值: 2.1258511279610506e-09
    """
