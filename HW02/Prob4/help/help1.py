import numpy as np

class Model:
    def __init__(self):
        self.prob = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.3, 0.4, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25]
        ])

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        print("Batch size:", self.batch_size)
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss2 = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss1,loss2

model = Model()
label = np.array([2, 1, 0])  # 真实标签
loss1,loss2 = model.get_loss(label)
print("Loss:", loss1, loss2)