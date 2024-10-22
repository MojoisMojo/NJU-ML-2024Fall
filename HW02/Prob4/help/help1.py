import numpy as np


class Model:
    def __init__(self):
        self.prob = np.array([[0.4], [0.3], [0.8]])

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        print("Batch size:", self.batch_size)
        loss = -np.mean(
            np.log(self.prob + 1e-8) * label
            + np.log(1 - self.prob + 1e-8) * (1 - label)
        )
        return loss


model = Model()
label = np.array([[0], [0], [1]])  # 真实标签
loss = model.get_loss(label)
print("Loss:", loss)
l = (-np.log(0.6 + 1e-8) - np.log(0.7 + 1e-8) - np.log(0.8 + 1e-8)) / 3
print("Expected loss:", l)
