"""
注意：
1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
2. 确保正确实现前向传播、反向传播和梯度更新。
3. 在比较不同初始化方法时，保持其他超参数不变。
4. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
5. 尝试使用不同的学习率（例如 0.01, 0.1, 1），并比较结果。
6. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""

import numpy as np
import os
import time
from layers import FullyConnectedLayer, ReLULayer, SigmoidLossLayer
import logging

from utils import plot_training_process, plot_decision_boundary

from params import RANDOM_SEED


from dataloader import X_train, X_test, y_train, y_test


class NeuralNetwork(object):
    def __init__(
        self,
        batch_size=16,
        input_size=2,
        hidden_size=4,
        output_size=1,
        max_epoch=1000,
        lr=0.01,
        init_method="random",
        print_iter=100,
    ):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.init_method = init_method
        self.print_iter = print_iter
        np.random.seed(RANDOM_SEED)

    def load_data(self):
        self.train_X = X_train
        self.train_y = y_train.reshape(-1, 1)
        self.test_X = X_test
        self.test_y = y_test.reshape(-1, 1)

    def build_model(self):
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden_size, self.output_size)
        self.sigmoid = SigmoidLossLayer()
        self.update_layer_list = [self.fc1, self.fc2]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param(self.init_method)

    def load(self, param_file):
        params = np.load(param_file).item()
        self.fc1.load_param(params["w1"], params["b1"])
        self.fc2.load_param(params["w2"], params["b2"])

    def save(self, param_file):
        params = {}
        params["w1"], params["b1"] = self.fc1.save_param()
        params["w2"], params["b2"] = self.fc2.save_param()
        np.save(param_file, params)

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        prob = self.sigmoid.forward(h2)
        return prob

    def backward(self):
        dloss = self.sigmoid.backward()
        assert dloss.shape == (self.batch_size, self.output_size)
        assert np.any(dloss != np.zeros_like(dloss))
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)
        for layer in self.update_layer_list:
            layer.update_param(self.lr)

    def train(self, save_dir):
        train_shape_0 = self.train_X.shape[0]
        batch_size = self.batch_size
        logging.info("Start training...")

        losses = []
        accuracies = []

        prob = self.forward(self.train_X)
        loss = self.sigmoid.get_loss(self.train_y)
        pred_labels = np.round(prob)
        accuracy = np.mean(pred_labels == self.train_y)
        logging.info(f"Init, loss: {loss}, accuracy: {accuracy}")
        losses.append(loss)
        accuracies.append(accuracy)
        plot_decision_boundary(
            self, self.train_X, self.train_y, "epoch-0 before", save_dir=save_dir
        )

        for idx_epoch in range(1, 1 + self.max_epoch):
            permutation = np.random.permutation(train_shape_0)
            X_shuffled = self.train_X[permutation]
            y_shuffled = self.train_y[permutation]
            for idx_batch in range(0, train_shape_0, batch_size):
                batch_X = X_shuffled[idx_batch : idx_batch + batch_size]
                batch_y = y_shuffled[idx_batch : idx_batch + batch_size]
                prob = self.forward(batch_X)
                loss = self.sigmoid.get_loss(batch_y)
                self.backward()
            if idx_epoch % self.print_iter == 0:
                prob = self.forward(self.train_X)
                loss = self.sigmoid.get_loss(self.train_y)
                pred_labels = np.round(prob)
                accuracy = np.mean(pred_labels == self.train_y)
                losses.append(loss)
                accuracies.append(accuracy)
                logging.info(f"Epoch {idx_epoch}, loss: {loss}, accuracy: {accuracy}")
                plot_decision_boundary(
                    self, self.train_X, self.train_y, f"epoch-{idx_epoch}", save_dir
                )
        return losses, accuracies

    def evaluate(self):
        prob = self.forward(self.test_X)
        pred_labels = np.round(prob)
        accuracy = np.mean(pred_labels == self.test_y)
        logging.info("Accuracy in test set: %f" % accuracy)
        return accuracy


def main(time_stemp, e, lr, btz, inmethod):
    mdir = f"./output/{time_stemp}/e{e}_lr{lr}_btz{btz}_{inmethod}"
    os.makedirs(mdir, exist_ok=True)
    print_iter = e // 20
    nn = NeuralNetwork(
        max_epoch=e, lr=lr, batch_size=btz, init_method=inmethod, print_iter=print_iter
    )

    nn.load_data()
    nn.build_model()
    nn.init_model()

    losses, accuracies = nn.train(save_dir=mdir)

    plot_training_process(losses, accuracies, mdir, print_iter)
    nn.save(f"{mdir}/model.npy")
    logging.info(f"Model saved to {mdir}/model.npy")
    acc = nn.evaluate()
    print(f"Test Accuracy: {acc}")
    plot_decision_boundary(nn, nn.test_X, nn.test_y, "test", save_dir=mdir)
    return nn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    time_stemp = time.strftime("%m%d_%H%M%S", time.localtime())
    btz = 16
    nn = main(time_stemp=time_stemp, e=1000, lr=0.01, btz=btz, inmethod="he")
    # for e,lr in [(200,0.01),(100,0.1),(50,1)]:
    #     for inmethod in ["random","xavier","he"]:
    #         print("#"*50,f"\ne={e},lr={lr},btz={btz},inmethod={inmethod}")
    #         nn = main(time_stemp=time_stemp, e=e, lr=lr, btz=btz, inmethod=inmethod)
