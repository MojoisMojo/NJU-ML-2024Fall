import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE(object):
    def __init__(self, X, y, N, K, random_state=None):
        self.N = N  # 每个小类样本合成样本个数
        self.K = K  # 近邻个数
        self.label = y  # 进行数据增强的类别
        self.sample = X
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
        neigh = NearestNeighbors(n_neighbors=self.K)
        neigh.fit(self.sample)
        neighbors = neigh.kneighbors(self.sample, return_distance=False)

        # 生成合成样本
        for i in range(self.n_sample):
            for n in range(N):
                nn = np.random.choice(neighbors[i][neighbors[i] != i])
                diff = self.sample[nn] - self.sample[i]
                gap = np.random.rand()
                synthetic_samples[i * N + n] = self.sample[i] + gap * diff
        return synthetic_samples, np.ones(n_synthetic_samples) * self.label


# 示例使用
if __name__ == "__main__":
    # 示例数据
    X = np.array(
        [
            [1, 1, 4],
            [1, 3, 2],
            [2, 4, 1],
            [3, 1, 2],
            [1, 2, 3],
            [2, 5, 2],
        ]
    )
    oy = np.array([1, 1, 1, 0, 0, 0])
    y = 1
    N = 2  # 每个小类样本合成的样本个数
    K = 2  # 近邻个数
    random_state = 53

    smote = SMOTE(X, y, N, K, random_state)
    synthetic_samples, y_reduced = smote.over_sampling()
    samples = np.vstack([X, synthetic_samples])
    oy = np.hstack([oy, y_reduced])
    print("样本:\n", samples)
    print("样本标签:\n", oy)
