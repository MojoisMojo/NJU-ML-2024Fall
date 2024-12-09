"""
https://blog.csdn.net/Sjxjdnidjenff/article/details/143107465 独立热编码参考
"""
from collections import Counter
dir = "../datasets"
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


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b""] = "-1"

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b"-1":
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = "0"
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO
        pass

    return data, onehot_features


def load_spam():
    features = [
        "pain",
        "private",
        "bank",
        "money",
        "drug",
        "spam",
        "prescription",
        "creative",
        "height",
        "featured",
        "differ",
        "width",
        "other",
        "energy",
        "business",
        "message",
        "volumes",
        "revision",
        "path",
        "meter",
        "memo",
        "planning",
        "pleased",
        "record",
        "out",
        "semicolon",
        "dollar",
        "sharp",
        "exclamation",
        "parenthesis",
        "square_bracket",
        "ampersand",
    ]
    assert len(features) == 32

    # Load spam data
    path_train = f"{dir}/spam_data/spam_data.mat"
    data = scipy.io.loadmat(path_train)
    X = data["training_data"]
    y = np.squeeze(data["training_labels"])
    Z = data["test_data"]
    class_names = ["Ham", "Spam"]
    return X, y, Z, class_names, features


def load_titanic():
    # Load titanic data
    path_train = f"{dir}/titanic/train.csv"
    df = pd.read_csv(path_train)
    df = df.drop(columns=["Name"])
    print(df.head())
    data = df.drop(columns=["Survived"]).values
    labels = df["Survived"].values
    path_test = f"{dir}/titanic/test.csv"
    test_df = pd.read_csv(path_test)
    test_df = test_df.drop(columns=["Name"])
    print(test_df.head())
    test_data = test_df.values
    y = labels
    class_names = ["Died", "Survived"]
    labeled_idx = np.where(y != b"")[0]
    y = np.array(y[labeled_idx], dtype=float).astype(int)
    print("\n\nPart (b): preprocessing the titanic dataset")
    X, onehot_features = preprocess(data, onehot_cols=[2, 6, 8, 9])
    X = X[labeled_idx, :]
    Z, _ = preprocess(test_data, onehot_cols=[2, 6, 8, 9])
    assert X.shape[1] == Z.shape[1]
    features_name = [
        "PassengerId",  # int
        # survived string
        "Pclass",  # int
        # name string
        "Sex",  # string
        "Age",  # int
        "SibSp",  # int
        "Parch",  # int
        "Ticket",  # string
        "Fare",  # float
        "Cabin",  # string
        "Embarked",  # string
    ]
    return X, y, Z, class_names, (features_name, onehot_features)


if __name__ == "__main__":
    X, y, Z, clns, fs = load_titanic()
    print(fs)
    print("X[0]:\n",X[0])
    print("y[0]:\n",y[0])
    print("Z[0]:\n",Z[0])
    
