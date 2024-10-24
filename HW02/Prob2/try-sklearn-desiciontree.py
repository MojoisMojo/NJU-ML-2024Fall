import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_loader import DataLoader
from constants import RAND_SEED, TEST_SIZE


# 加载数据集
data = load_digits()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND_SEED
)

# 定义不同的 max_depth 值
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 30]

# 存储训练集和测试集的精度
train_accuracies = []
test_accuracies = []

# 训练并评估决策树模型
for max_depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RAND_SEED)
    clf.fit(X_train, y_train)

    # 计算训练集精度
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)

    # 计算测试集精度
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

    print(
        f"Max Depth: {max_depth}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

# 绘制训练集和测试集精度随 max_depth 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(max_depths, test_accuracies, label="Test Accuracy", marker="o")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs. Max Depth")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f"DecisionTreeAccuracyVsMaxDepth_{max_depth}.png")
