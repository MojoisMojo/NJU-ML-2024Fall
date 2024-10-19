import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import datetime
import os
from utils import mytqdm

def train(model:SVC, X_train, y_train):
    # 使用 tqdm 显示训练进度
    for _ in mytqdm(range(1), desc="Training SVM model"):
        model.fit(X_train, y_train)

def predict(model:SVC, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

