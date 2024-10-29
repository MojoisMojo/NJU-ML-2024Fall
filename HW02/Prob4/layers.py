import numpy as np

"""
实现参考
    知乎:  https://zhuanlan.zhihu.com/p/377634925
    CS188: Introduction to Artificial Intelligence
"""


def init_param(method, input_size, output_size, shape):
    if method == "random":
        return random_init(shape)
    elif method == "xavier":
        return xavier_init(input_size, output_size, shape)
    elif method == "he":
        return he_init(input_size, shape)


def random_init(shape):
    return np.random.uniform(-0.5, 0.5, shape)


xavier_func = lambda ni, no: (lambda x: (-x, x))(np.sqrt(6 / (ni + no)))


def xavier_init(input_size, output_size, shape):
    return np.random.uniform(*xavier_func(input_size, output_size), shape)


def he_init(input_size, shape):
    return np.random.normal(0, np.sqrt(2 / input_size), shape)


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print(
            "\tFully connected layer with input %d, output %d."
            % (self.num_input, self.num_output)
        )

    def init_param(self, init_method):
        self.weight = init_param(
            init_method,
            self.num_input,
            self.num_output,
            (self.num_input, self.num_output),
        )
        self.bias = init_param(
            init_method, self.num_input, self.num_output, (1, self.num_output)
        )

    def forward(self, input):
        self.input = input
        self.output = (input @ self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        # input x w + b = output  
        # dtdiff/dw = input
        self.d_weight = np.dot(self.input.T, top_diff)
        # dtdiff/db = 1
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        # bdiff * w = tdiff
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def __init__(self):
        print("\tReLU layer.")

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input < 0] = 0
        return bottom_diff


class SigmoidLossLayer(object):
    def __init__(self):
        print("\tSigmoid loss layer.")

    def forward(self, input):  # 前向传播的计算
        self.prob = 1 / (1 + np.exp(-input))
        return self.prob

    def get_loss(self, label):  # 计算损失
        """
        label: batch_size * 1
        """
        self.batch_size = self.prob.shape[0]
        self.label_onehot = label.copy()
        loss = -np.mean(
            np.log(self.prob + 1e-8) * label
            + np.log(1 - self.prob + 1e-8) * (1 - label)
        )
        return loss

    def backward(self):
        """
        很有趣的一点是，在交叉熵的条件下，
        sigmoidLoss的反向传播和softmax的反向传播是一样的
        dE/din = dE/dout * dout/din
        dout/din = out(1-out) sigmoid函数的导数
        dE/dout = [-1/pred_i if label_i = 1 else 1-pred_i]
        so:
            dE/din = pred(1-pred) * (-1/pred if label = 1 else (1-pred))
            pred - 1 if lavel = 1 else pred
            也就是 pred - label
        """
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
