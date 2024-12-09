"""
https://blog.csdn.net/Sjxjdnidjenff/article/details/143107465 独立热编码参考
"""

from collections import Counter

dir = "../datasets"
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random


def myHotEncoder(X):
    assert isinstance(X, pd.DataFrame)
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    # 对类别型特征进行独热编码
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_numeric = X[numerical_features].values
    X_prepared = np.hstack((X_numeric, X_encoded.toarray()))
    return X_prepared


def myLabelEncoder(X):
    assert isinstance(X, pd.DataFrame)
    for col in X.columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    return X


def preprocess(data: np.ndarray, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == " "] = None
    data[data == ""] = None

    n_samples, n_features = data.shape

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == None:
                continue
            if term[-1] <= min_freq:
                break
            print(col, term)
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = "0"
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for col in range(n_features):
            column_data = data[:, col]
            non_missing = column_data != -1
            if np.sum(non_missing) == 0:
                continue
            valid_data = column_data[non_missing]
            counts = Counter(valid_data)
            mode, cnt = counts.most_common(1)[0]
            column_data[~non_missing] = mode
            data[:, col] = column_data
    return data, onehot_features


def load_spam(encoder="onehot"):
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
    print(data.keys())

    X = data["training_data"]
    y = np.squeeze(data["training_labels"])
    # 0.8 train, 0.2 test
    X, Z, y, Zy = train_test_split(X, y, test_size=0.2, random_state=14)
    class_names = ["Ham", "Spam"]
    assert X.shape[1] == len(features)
    assert Z.shape[1] == len(features)
    return X, y, Z, Zy, class_names, features


preprocesser = None


def load_titanic(encoder="onehot"):

    if encoder == "onehot":
        preprocesser = myHotEncoder
    elif encoder == "label":
        preprocesser = myLabelEncoder
    else:
        raise ValueError(f"Invalid encoder: {encoder}")

    # Load titanic data
    path_train = f"{dir}/titanic/train.csv"
    path_test = f"{dir}/titanic/test.csv"
    path_all_label = f"{dir}/titanic/gender_submission.csv"

    df_train = pd.read_csv(path_train).drop(columns=["Name"])
    df_train_X = df_train.drop(columns=["Survived"])
    df_train_y = df_train["Survived"]
    df_test_X = pd.read_csv(path_test).drop(columns=["Name"])
    df_train_X.fillna(" ", inplace=True)
    df_test_X.fillna(" ", inplace=True)
    print(df_train_X.head())
    print(df_test_X.head())
    string_feature_indexs = [2, 6, 8, 9]
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
    df_test_y = pd.read_csv(path_all_label)
    class_names = ["Died", "Survived"]

    train_X = df_train_X.values
    train_y = df_train_y.values
    test_X = df_test_X.values
    test_y = df_test_y.values[:, 1]

    combined_X = np.vstack([train_X, test_X])

    combined_X, onehot_features = preprocess(
        combined_X, onehot_cols=string_feature_indexs
    )

    train_X = combined_X[: len(train_X)]
    test_X = combined_X[len(train_X) :]

    total_features = features_name + onehot_features

    return train_X, train_y, test_X, test_y, class_names, total_features

    # V2

    combined_X = pd.concat([df_train_X, df_test_X], keys=["train", "test"])

    onehot_features = [features_name[i] for i in string_feature_indexs]
    combined_X = pd.get_dummies(
        combined_X, columns=onehot_features, prefix=onehot_features
    )

    df_onehot_train_X = combined_X.xs("train")
    df_onehot_test_X = combined_X.xs("test")

    X = df_onehot_train_X.values
    y = df_train_y.values
    Z = df_onehot_test_X.values
    Zy = df_test_y.values[:, 1]

    assert X.shape[1] == Z.shape[1]
    return X, y, Z, Zy, class_names, combined_X.columns


if __name__ == "__main__":
    X, y, Z, Zy, clns, fs = load_spam()
    # X, y, Z, Zy, clns, fs = load_titanic()
    print(fs)
    print("X[0]:\n", X[0])
    print("y[0]:\n", y[0])
    print("Z[0]:\n", Z[0])
    if Zy:
        print("Zy[0]:\n", Zy[0])
