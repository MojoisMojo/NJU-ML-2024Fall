from model import SVCModel
from sklearn.neighbors import NearestNeighbors
from dataloader import DataLoader
from datetime import datetime
import numpy as np
import os
from utils import mytqdm, print_and_write

"""
% 注意:
% 1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
% 2. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
% 3. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""


class SMOTE(object):
    def __init__(self, X, y, N, K, random_state=None):
        self.N = N  # 每个小类样本合成样本个数
        self.K = K  # 近邻个数
        self.label = y  # 进行数据增强的类别
        self.sample = X  # 进行数据增强的样本
        self.n_sample, self.n = X.shape  # 小样本个数, 特征个数
        if random_state is not None:
            np.random.seed(random_state)

    def over_sampling(self):
        # 计算需要生成的合成样本数量
        N = self.N
        n_synthetic_samples = N * self.n_sample

        # 初始化合成样本数组
        synthetic_samples = np.zeros((n_synthetic_samples, self.n))

        # 使用 K 近邻算法找到每个样本的 K 个最近邻
        neigh = NearestNeighbors(
            n_neighbors=self.K,
        )
        neigh.fit(self.sample)
        neighbors = neigh.kneighbors(self.sample, return_distance=False)

        # 生成合成样本
        for i in mytqdm(range(self.n_sample), desc="Generating synthetic samples"):
            for n in range(N):
                nn = np.random.choice(neighbors[i][neighbors[i] != i])
                diff = self.sample[nn] - self.sample[i]
                gap = np.random.rand()
                synthetic_samples[i * N + n] = self.sample[i] + gap * diff
        print("Synthetic samples generated")
        return synthetic_samples, np.ones(n_synthetic_samples) * self.label


def task3(
    run_time,
    data_loader: DataLoader,
    loadpath=None,
    is_train=True,
    params={"N": 50, "K": 7},
):
    N, K = params["N"], params["K"]
    task_name = "task3"
    dir_path = f"./output/{run_time}/{task_name}"
    savepath = f"{dir_path}/svm_model.pkl"
    output_path = f"{dir_path}/out.out"
    curve_path = f"{dir_path}/roc_curve.png"

    os.makedirs(dir_path, exist_ok=True)
    # outfile = open(output_path, "w")  # 清空文件内容
    # outfile.close()

    X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=False)
    y_pos_label = 1
    X_train_pos = X_train[y_train == y_pos_label]

    print_and_write(output_path, "#" * 30, f"\nSMOTE N={N}, K={K}")

    if is_train:
        print(f"X_train shape: {X_train.shape}")

        smote = SMOTE(X_train_pos, y_pos_label, N=N, K=K)

        synthetic_samples, y_reduced = smote.over_sampling()
        X_train = np.vstack([X_train, synthetic_samples])
        y_train = np.hstack([y_train, y_reduced])

        print_and_write(output_path, f"X_train shape: {X_train.shape}")

    svm_model = SVCModel(loadpath=loadpath, savepath=savepath)

    if loadpath is None or is_train:
        svm_model.train(X_train, y_train)

    y_pred, y_prob = svm_model.predict(X_test)

    svm_model.validate_and_print(y_test, y_pred, y_prob, output_path, curve_path)


if __name__ == "__main__":
    data_loader = DataLoader("../data/creditcard.csv")
    run_timestemp = datetime.now().strftime("%m%d_%H%M%S")
    # load_path = "./out/rand_seed_14_N_400_K_7/task3/svm_model.pkl"
    load_path = None
    params = {"N": 30, "K": 7}
    task3(run_timestemp, data_loader, load_path, params=params)
