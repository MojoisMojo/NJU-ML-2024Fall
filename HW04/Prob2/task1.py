from data_loader import load_spam, load_titanic
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import model
from collections import Counter
import numpy as np
from numpy import genfromtxt
from pydot import graph_from_dot_data
from utils import evaluate, visualize_tree_and_save


def task1(dataset_name="spam"):  # for spam dataset
    if dataset_name == "spam":
        X, y, Z, class_names, features = load_spam()
    elif dataset_name == "titanic":
        X, y, Z, class_names, features = load_titanic()
    else:
        raise ValueError("Invalid dataset name")
    print("Data loaded")
    print("train first data:", X[0])
    print("train first label:", y[0])
    print("test first data:", Z[0])
    print("Class Names:", class_names)
    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)
    assert len(X) == len(y), "X and y must have the same length"
    N = 100

    print("\n\nDecision Tree")
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X, y)

    print("\n\nTree Structure")
    print(dt.__repr__(), "\n\n")

    visualize_tree_and_save(dt, dataset_name, features, class_names)


if __name__ == "__main__":
    # name = "spam"
    name = "titanic"
    task1(name)
