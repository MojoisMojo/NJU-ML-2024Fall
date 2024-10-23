"""
注意：
1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
2. 确保正确实现前向传播、反向传播和梯度更新。
3. 在比较不同初始化方法时，保持其他超参数不变。
4. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
5. 尝试使用不同的学习率（例如 0.01, 0.1, 1），并比较结果。
6. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""

RANDOM_SEED = 42
learning_rate = None
epochs = None
batch_size = None
init_method = ""
prefix = ""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os

# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


class NeuralNetwork:
    def __init__(
        self, input_size=2, hidden_size=4, output_size=1, init_method="random"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        np.random.seed(RANDOM_SEED)

        # 初始化权重和偏置
        if init_method == "random":  # 随机初始化：
            """
            W_ij ~ U(-0.5, 0.5)
            """
            self.W1 = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
            self.b1 = np.random.uniform(-0.5, 0.5, (1, self.hidden_size))
            self.W2 = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
            self.b2 = np.random.uniform(-0.5, 0.5, (1, self.output_size))
        elif init_method == "xavier":  # Xavier 初始化：根据前一层的节点数进行缩放。
            """
            W_ij ~ U(-sqrt(6/(ni+no)), sqrt(6/(ni+no)))
            """
            cal = lambda ni, no: (lambda x: (-x, x))(np.sqrt(6 / (ni + no)))
            self.W1 = np.random.uniform(
                *cal(self.input_size, self.hidden_size),
                (self.input_size, self.hidden_size),
            )
            self.b1 = np.random.uniform(
                *cal(self.input_size, self.hidden_size),
                (1, self.hidden_size),
            )
            self.W2 = np.random.uniform(
                *cal(self.hidden_size, self.output_size),
                (self.hidden_size, self.output_size),
            )
            self.b2 = np.random.uniform(
                *cal(self.hidden_size, self.output_size),
                (1, self.output_size),
            )
        elif (
            init_method == "he"
        ):  # He 初始化假设每一层都是线性的，并且考虑了 ReLU 激活函数的特性
            """
            W_ij ~ N(0, sqrt(2/ni))
            """
            self.W1 = np.random.normal(
                0, np.sqrt(2 / self.input_size), (self.input_size, self.hidden_size)
            )
            self.b1 = np.random.normal(
                0, np.sqrt(2 / self.input_size), (1, self.hidden_size)
            )
            self.W2 = np.random.normal(
                0, np.sqrt(2 / self.hidden_size), (self.hidden_size, self.output_size)
            )
            self.b2 = np.random.normal(
                0, np.sqrt(2 / self.hidden_size), (1, self.output_size)
            )
        else:
            raise ValueError("Unsupported initialization method")

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # 隐藏层使用 ReLU 激活函数
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 输出层使用 Sigmoid 激活函数
        return self.a2

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def loss(self, y, y_pred): #  均方误差
        return np.mean((y - y_pred) ** 2)

    def train(
        self, X, y, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size
    ):
        self.learning_rate = learning_rate
        losses = []
        accuracies = []

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                y_pred = self.forward(X_batch)
                loss = self.loss(y_batch, y_pred)
                self.backward(X_batch, y_batch, y_pred)

            if epoch % 100 == 0:
                y_pred_train = self.forward(X)
                train_loss = self.loss(y, y_pred_train)
                train_acc = np.mean((y_pred_train > 0.5) == y.reshape(-1, 1))
                losses.append(train_loss)
                accuracies.append(train_acc)
                print(f"Epoch {epoch}, Loss: {train_loss}, Accuracy: {train_acc}")
            if epoch % 1000 == 0:
                plot_decision_boundary(self, X, y, epoch)
        return losses, accuracies


# 辅助函数
def plot_decision_boundary(model, X, y, tag=epochs):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
    plt.title(f"Decision Boundary {prefix.split('/')[-1]}")
    plt.savefig(f"{prefix}/decision_boundary_ep-{tag}.png")
    # plt.show()
    plt.close()


def plot_training_process(losses, accuracies):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(accuracies, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(f"{prefix}/training_process.png")
    # plt.show()
    plt.close()


# 主函数
def main():
    nn = NeuralNetwork(init_method=init_method)
    losses, accuracies = nn.train(
        X_train,
        y_train,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )
    plot_training_process(losses, accuracies)
    plot_decision_boundary(nn, X_test, y_test, tag="test")


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 3000
    batch_size = 32
    init_method = "random"
    prefix = f"output/lr-{learning_rate}_bs-{batch_size}_im-{init_method}"
    os.makedirs(f"{prefix}", exist_ok=True)
    main()
