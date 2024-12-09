from data_loader import load_spam, load_titanic
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from model import RandomForest
from collections import Counter
import numpy as np
from numpy import genfromtxt
from pydot import graph_from_dot_data
from utils import visualize_tree_and_save
from sklearn.model_selection import cross_val_score


def load_data(dataset_name):
    if dataset_name == "spam":
        return load_spam()
    elif dataset_name == "titanic":
        return load_titanic()
    else:
        raise ValueError("Invalid dataset name")


def task1(dataset_name="spam"):
    X, y, Z, Zy, class_names, features = load_data(dataset_name)
    print("Data loaded from dataset:", dataset_name)
    print("Class Names:", class_names)
    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)
    assert len(X) == len(y), "X and y must have the same length"
    print("\n\nDecision Tree")
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X, y)
    dt_train_acc = dt.score(X, y)
    dt_test_acc = dt.score(Z, Zy)
    dt_test_cross = cross_val_score(dt, Z, Zy, cv=5).mean()
    print("Decision Tree Train Accuracy: ", dt_train_acc)
    print("Decision Tree Test Accuracy: ", dt_test_acc)
    print("Decision Tree Test 5 Cross Validation Accuracy: ", dt_test_cross)

    print("\n\nTree Structure")
    print(dt.__repr__(), "\n\n")

    visualize_tree_and_save(dt, dataset_name, features, class_names)

def task2(dataset_name):
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    X, y, Z, Zy, class_names, features = load_data(dataset_name)
    print("Data loaded from dataset:", dataset_name)
    print("Class Names:", class_names)
    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)
    assert len(X) == len(y), "X and y must have the same length"
    print("\n\nRandom Forest")
    # Random Forest
    N = 100
    rf = RandomForest(params, n=N, m=np.int_(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    rf_train_acc = rf.score(X, y)
    rf_test_acc = rf.score(Z, Zy)
    rf_test_cross = cross_val_score(rf, Z, Zy, cv=5).mean()
    print("Random Forest Train Accuracy: ", rf_train_acc)
    print("Random Forest Test Accuracy: ", rf_test_acc)
    print("Random Forest Test 5 Cross Validation Accuracy: ", rf_test_cross)
    print("\n\n Root Clf Std")
    for s,c in rf.get_trees_root_clf_std(features):
        print(s,"(",c,"trees)")


if __name__ == "__main__":
    name = "spam"
    # name = "titanic"
    # task1(name)
    task2(name)
