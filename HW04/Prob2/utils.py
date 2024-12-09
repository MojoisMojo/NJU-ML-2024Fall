from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier


# def evaluate(clf, X, y, features):
#     print("Cross validation", cross_val_score(clf, X, y))
#     if hasattr(clf, "decision_trees"):
#         counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
#         first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
#         print("First splits", first_splits)


def visualize_tree_and_save(dt, dataset_name, features, class_names):
    # Visualize Decision Tree
    outfile = f"images/{dataset_name}-basic-tree"
    print("Visualize Decision Tree")
    dot_data = export_graphviz(
        dt,
        out_file=None,
        feature_names=features,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    # Save tree to pdf
    graph = Source(dot_data)
    graph.render(outfile, format="pdf")
    print("Tree saved to", f"{outfile}.pdf")
    print("\n\n")

def root_clf_std(dt:DecisionTreeClassifier)->str:
    """
    :param dt: DecisionTreeClassifier
    :return: Tuple[int, number]
    """
    tree = dt.tree_
    root_node = 0
    feature_index  = tree.feature[root_node]
    threshold = tree.threshold[root_node]
    return feature_index, threshold