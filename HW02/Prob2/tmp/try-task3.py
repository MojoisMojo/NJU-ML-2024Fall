import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from params import RAND_SEED, TEST_SIZE

# 数据准备
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_SEED)

# 未剪枝的决策树
clf_unpruned = DecisionTreeClassifier(random_state=RAND_SEED)
clf_unpruned.fit(X_train, y_train)
y_pred_unpruned = clf_unpruned.predict(X_test)
accuracy_unpruned = accuracy_score(y_test, y_pred_unpruned)

# 预剪枝的决策树
clf_prepruned = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=RAND_SEED)
clf_prepruned.fit(X_train, y_train)
y_pred_prepruned = clf_prepruned.predict(X_test)
accuracy_prepruned = accuracy_score(y_test, y_pred_prepruned)

# 后剪枝的决策树
clf_postpruned = DecisionTreeClassifier(random_state=RAND_SEED)
clf_postpruned.fit(X_train, y_train)
path = clf_postpruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=RAND_SEED, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 选择最佳的后剪枝模型
clfs = clfs[:-1]  # 去掉最后一个模型，因为它是完全剪枝的
alpha_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
best_alpha_index = np.argmax(alpha_scores)
best_clf_postpruned = clfs[best_alpha_index]
accuracy_postpruned = alpha_scores[best_alpha_index]

# 统计显著性检验
_, p_value_unpruned_prepruned = ttest_ind(y_pred_unpruned, y_pred_prepruned)
_, p_value_unpruned_postpruned = ttest_ind(y_pred_unpruned, y_pred_postpruned)
_, p_value_prepruned_postpruned = ttest_ind(y_pred_prepruned, y_pred_postpruned)

# 输出结果
print(f"未剪枝的决策树准确率: {accuracy_unpruned}")
print(f"预剪枝的决策树准确率: {accuracy_prepruned}")
print(f"后剪枝的决策树准确率: {accuracy_postpruned}")
print(f"未剪枝 vs 预剪枝 p值: {p_value_unpruned_prepruned}")
print(f"未剪枝 vs 后剪枝 p值: {p_value_unpruned_postpruned}")
print(f"预剪枝 vs 后剪枝 p值: {p_value_prepruned_postpruned}")