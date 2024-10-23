# coding=utf-8
import numpy as np
import struct
import os
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from layers import FullyConnectedLayer, ReLULayer, SigmoidLossLayer
import logging

from params import RANDOM_SEED, learning_rate, epochs, batch_size, print_iter

from utils import plot_decision_boundary, plot_training_process

from dataloader import X_train, X_test, y_train, y_test


class NeuralNetwork(object):
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
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.print_iter = print_iter
        np.random.seed(RANDOM_SEED)
        self.build_model()
        self.init_model()

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
            layer.update_param(self.learning_rate)

    def train(self, X, y, save_dir):

        epochs = self.epochs

        train_shape_0 = X.shape[0]
        batch_size = self.batch_size
        logging.info("Start training...")

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

        for epoch in range(1, 1 + epochs):
            permutation = np.random.permutation(train_shape_0)
            X_shuffled = X[permutation]
            y_shuffled = y_labels[permutation]
            for i in range(0, train_shape_0, batch_size):
                batch_X = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]
                # 这里必须先forward,get_loss再backward
                # 因为backward中会用到forward的结果
                prob = self.forward(batch_X)
                loss = self.sigmoid.get_loss(batch_y)
                self.backward()
            if epoch % self.print_iter == 0:
                cal_loss_acc(f"Epoch{epoch}")
        plot_training_process(losses, accuracies, save_dir)
        return losses, accuracies

    def evaluate(self, X, y):
        y_label = y.reshape(-1, 1)
        y_pred = self.forward(X)
        test_acc = np.mean(np.round(y_pred) == y_label)
        logging.info(f"Accuracy in test set: {test_acc}")
        return test_acc


def main_task(e, lr, btz, inmethod):
    time_stemp = time.strftime("%m%d_%H%M%S", time.localtime())
    nn = NeuralNetwork(
        learning_rate=lr,
        epochs=e,
        batch_size=btz,
        print_iter=e // 20,
        init_method=inmethod,
    )

    mdir = f"./output/{time_stemp}"
    os.makedirs(mdir, exist_ok=True)
    nn.train(X_train, y_train, save_dir=mdir)
    nn.save(f"{mdir}/e{e}_lr{lr}_btz{btz}_me{inmethod}.npy")
    logging.info(f"Model saved to {mdir}/e{e}_lr{lr}_btz{btz}_me{inmethod}.npy")
    acc = nn.evaluate(X_test, y_test)
    print(f"Accuracy: {acc}")
    return nn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    nn = main_task(e=200, lr=0.1, btz=16, inmethod="random")

