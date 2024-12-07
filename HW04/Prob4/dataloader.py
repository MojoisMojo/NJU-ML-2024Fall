import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 加载数据
def load_mall_data():
    df = pd.read_csv("../datasets/Mall_Customers.csv")
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, df

def deScale(X, tagets):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.inverse_transform(tagets)