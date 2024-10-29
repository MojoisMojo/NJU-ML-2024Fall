from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from constants import RAND_SEED
import logging


class PrePrunDTModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        # 划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RAND_SEED
        )
        models = []
        for md in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 30]:
            tree = DecisionTreeClassifier(max_depth=md, random_state=RAND_SEED)
            tree.fit(X_train, y_train)
            models.append((tree, tree.score(X_val, y_val)))
            logging.info(f"Max depth: {md}, Accuracy: {models[-1][1]}")
        m, bestscore = max(models, key=lambda x: x[1])
        self.model: DecisionTreeClassifier = m
        print(f"Best max depth: {self.model.get_depth()}, Accuracy: {bestscore}")
        return self.model, bestscore
