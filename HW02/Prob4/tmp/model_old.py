import numpy as np
import logging
from utils import relu, sigmoid

from params import RANDOM_SEED
from utils import plot_decision_boundary, plot_training_process


def bi_cross_entropy_loss(label, prob):
    """_summary_

    Args:
        label (np.ndarray): batch_size x 1 (value = 0 or 1)
        prob (np.ndarray): batch_size x 1

    Returns:
        float: bi_cross_entropy_loss
    """
    loss = -np.mean(np.log(prob + 1e-8) * label + np.log(1 - prob + 1e-8) * (1 - label))
    return loss


class NeuralNetwork:
    def __init__(
        self,
        learning_rate,
        epochs,
        batch_size,
        print_iter,
        input_size=2,
        hidden_size=4,
        output_size=1,
        init_method="random",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.print_iter = print_iter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_f = bi_cross_entropy_loss  # label, pred
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

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)  # 隐藏层使用 ReLU 激活函数
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # 输出层使用 Sigmoid 激活函数
        return self.a2

    def backward(self, X, y, y_pred):
        batch_size = X.shape[0]
        # loss = np.ones_like(y_pred)
        # dz2 = self.loss_f(y, y_pred) * loss
        dz2 = y_pred - y
        dW2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0) / batch_size
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, save_dir):

        epochs = self.epochs
        batch_size = self.batch_size
        print_iter = self.print_iter
        train_shape_0 = X.shape[0]

        losses = []
        accuracies = []

        y_labels = y.reshape(-1, 1)

        def cal_loss_acc(tag):
            nonlocal losses, accuracies
            prob = self.forward(X)
            loss = self.sigmoid.get_loss(y)
            pred_labels = np.round(prob)
            accuracy = np.mean(pred_labels == y_labels)
            losses.append(loss)
            accuracies.append(accuracy)
            logging.info(f"{tag}, loss: {loss}, accuracy: {accuracy}")
            plot_decision_boundary(self, X, y, tag, save_dir)

        cal_loss_acc("Init")

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(train_shape_0)
            X_shuffled = X[permutation]
            y_shuffled = y_labels[permutation]

            for i in range(0, train_shape_0, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            if epoch % print_iter == 0:
                cal_loss_acc(f"Epoch {epoch}")
        plot_training_process(losses, accuracies, save_dir)
        return losses, accuracies

    def evaluate(self, X, y):
        y_label = y.reshape(-1, 1)
        y_pred = self.forward(X)
        test_acc = np.mean(np.round(y_pred) == y_label)
        logging.info(f"Accuracy in test set: {test_acc}")
        return test_acc
