import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
"""
Classes 3
Samples per class [59,71,48]
Samples total 178
Features 13
"""
X, y = load_wine(return_X_y=True)

def evaluate_model(X_selected , y):
    model = LogisticRegression(max_iter=200)
    scores = cross_val_score(model, X_selected , y,cv=5, scoring='accuracy')
    return np.mean(scores)