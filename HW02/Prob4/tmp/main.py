"""
注意：
1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
2. 确保正确实现前向传播、反向传播和梯度更新。
3. 在比较不同初始化方法时，保持其他超参数不变。
4. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
5. 尝试使用不同的学习率（例如 0.01, 0.1, 1），并比较结果。
6. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""

import os

from utils import plot_decision_boundary, plot_training_process

from dataloader import X_train, X_test, y_train, y_test

# from model_old import NeuralNetwork
from model import NeuralNetwork


# 主函数
def main(
    learning_rate=0.01,
    epochs=200,
    batch_size=16,
    init_method="random",
    save_dir="output",
):
    nn = NeuralNetwork(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        init_method=init_method,
        print_iter=epochs // 20,
    )
    losses, accuracies = nn.train(
        X_train,
        y_train,
        save_dir=save_dir,
    )
    accuracies = nn.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracies}")
    plot_decision_boundary(nn, X_test, y_test, tag="test", save_dir=save_dir)


from datetime import datetime
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    learning_rate = 0.01
    epochs = 200
    batch_size = 16
    print_iter = 20
    init_method = "random"
    timestemp = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = f"output/{timestemp}/lr-{learning_rate}_bs-{batch_size}_im-{init_method}"
    os.makedirs(f"{save_dir}", exist_ok=True)
    main(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        init_method=init_method,
        save_dir=save_dir,
    )
