"""
使用 L1 正则化的 Logistic 回归（LASSO）进行特征选择
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
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

lasso = Lasso(alpha=0.1, max_iter=200)
lasso.fit(X, y)
for coef in lasso.coef_:
    print(coef, end=", ")
print()
X_lasso_indices = np.where(lasso.coef_ != 0)[0]
X_lasso_selected = X[:, X_lasso_indices]
X_lasso_selected_number = len(X_lasso_indices)
X_lasso_indices = sorted(X_lasso_indices, key=lambda x: abs(lasso.coef_[x]), reverse=True)
print("selected indices(sorted):", X_lasso_indices)
print("selected:", X_lasso_selected_number)
print("penalty:", sum(map(lambda x: abs(x),lasso.coef_)))
print("score:", evaluate_model(X_lasso_selected, y))