# coding=utf-8
import numpy as np
import struct
import os
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from layers import FullyConnectedLayer, ReLULayer, SigmoidLossLayer
import logging

RANDOM_SEED = 42
MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


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

    def train(self):
        train_shape_0 = self.train_X.shape[0]
        batch_size = self.batch_size
        logging.info("Start training...")

        prob = self.forward(self.train_X)
        loss = self.sigmoid.get_loss(self.train_y)
        pred_labels = np.round(prob)
        accuracy = np.mean(pred_labels == self.train_y)
        logging.info(f"Init, loss: {loss}, accuracy: {accuracy}")

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
            prob = self.forward(self.train_X)
            loss = self.sigmoid.get_loss(self.train_y)
            pred_labels = np.round(prob)
            accuracy = np.mean(pred_labels == self.train_y)
            if idx_epoch % self.print_iter == 0:
                logging.info(f"Epoch {idx_epoch}, loss: {loss}, accuracy: {accuracy}")

    def evaluate(self):
        prob = self.forward(self.test_X)
        pred_labels = np.round(prob)
        accuracy = np.mean(pred_labels == self.test_y)
        logging.info("Accuracy in test set: %f" % accuracy)
        return accuracy


def build_neural_network(e, lr, btz, inmethod):
    time_stemp = time.strftime("%m%d_%H%M%S", time.localtime())
    nn = NeuralNetwork(max_epoch=e, lr=lr, batch_size=btz, init_method=inmethod, print_iter=10)
    nn.load_data()
    nn.build_model()
    nn.init_model()
    nn.train()
    mdir = f"./model/{time_stemp}"
    os.makedirs(mdir, exist_ok=True)
    nn.save(f"{mdir}/e{e}_lr{lr}_btz{btz}_me{inmethod}.npy")
    logging.info(f"Model saved to {mdir}/e{e}_lr{lr}_btz{btz}_me{inmethod}.npy")
    return nn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    nn = build_neural_network(e=200, lr=0.1, btz=16, inmethod="random")
    acc = nn.evaluate()
    print(f"Accuracy: {acc}")
