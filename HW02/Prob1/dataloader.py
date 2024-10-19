import pandas as pd
from sklearn.model_selection import train_test_split
from params import RAND_STATE_SEED


class DataLoader:
    def __init__(self, csv_path) -> None:
        df = pd.read_csv(csv_path)
        self.X = df.drop("Class", axis=1).values
        self.y = df["Class"].values

    # 默认：80% 训练集，20% 测试集
    # 针对 不平衡数据集 使用 stratify=y
    def split(self, test_size=0.2, stratify=False):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=RAND_STATE_SEED,
            stratify=self.y if stratify else None,
        )
        return X_train, X_test, y_train, y_test

    def split_undersampling(self):
        raise NotImplementedError

    def reduce_negatives(self, X_train, y_train, remove_count):
        X_train_pos = X_train[y_train == 1]
        X_train_neg = X_train[y_train == 0]

        X_train_neg_reduced = X_train_neg.sample(
            n=len(X_train_neg) - remove_count, random_state=RAND_STATE_SEED
        )

        X_train_reduced = pd.concat([X_train_neg_reduced, X_train_pos])
        y_train_reduced = pd.concat(
            [
                pd.Series(0, index=X_train_neg_reduced.index),
                pd.Series(1, index=X_train_pos.index),
            ]
        )

        return X_train_reduced, y_train_reduced
