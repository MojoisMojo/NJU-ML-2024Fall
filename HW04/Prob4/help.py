from kmeans_pp import KMeansPP
from utils import euclidean_distance
import numpy as np


class ConstrainedKMeans(KMeansPP):
    def __init__(
        self,
        n_clusters=3,
        max_iters=200,
        rand=None,
        weights=None,
        size_constraints=None,
    ):
        """
        n_clusters: 聚类中心的个数
        max_iters: 最大迭代次数
        rand: 随机种子
        weights: 每个特征的权重，shape (n_features,)
        size_constraints: 每个聚类的最小和最大大小约束，shape (2, ) float表示比例
        """
        super().__init__(n_clusters, max_iters, rand)
        self.weights = None
        self.size_constraints = None
        if weights is not None:
            assert len(weights) == n_clusters, "weights shape should be (n_clusters,)"
            self.weights = weights * n_clusters / np.sum(weights)  # 归一化权重
        if size_constraints is not None:
            assert len(size_constraints) == 2, "size_constraints shape should be (2,)"
            min_, max_ = size_constraints
            if min_ is None:
                min_ = 0.0
            if max_ is None:
                max_ = 1.0
            assert (
                min_ <= max_
            ), "size_constraints[0] should be less than size_constraints[1]"
            assert isinstance(min_, float), "size_constraints[0] should be float"
            assert isinstance(max_, float), "size_constraints[1] should be float"
            assert (
                0 <= min_ <= (1.0 / n_clusters)
            ), "size_constraints[0] should be in [0, 1/n_clusters]"
            assert (
                (1.0 / n_clusters) <= max_ <= 1.0
            ), "size_constraints[1] should be in [1.0 / n_clusters, 1]"
            self.size_constraints = [min_, max_]

    def get_distance(self, X, Y):
        """计算两个点之间的距离 考虑权重
        X: shape (n_features)
        Y: shape (n_features)
        @return number
        """
        if self.weights is None:
            return super().get_distance(X, Y)
        return euclidean_distance(X * self.weights, Y * self.weights)

    def assign_labels(self, distances, past_labels) -> np.ndarray:
        """将每个样本分配到最近的聚类中心 保证每个簇的大小约束
        @param distances: shape (n_samples, n_clusters)
        """
        new_labels = super().assign_labels(distances, past_labels)
        if self.size_constraints is None:
            return new_labels
        n_samples = distances.shape[0]
        n_clusters = distances.shape[1]
        min_size, max_size = self.size_constraints
        min_size = max(1, int(min_size * n_samples))
        max_size = min(n_samples, int(max_size * n_samples))
        for i in range(self.n_clusters):
            cluster_size = np.sum(new_labels == i)
            if cluster_size < min_size or cluster_size > max_size:
                # 重新分配
                farthest_indices = np.where(new_labels == i)[0] # 找到当前簇的所有样本
                farthest_distances = distances[farthest_indices, i] # 计算到当前簇的距离
                for idx in farthest_indices[np.argsort(farthest_distances)[::-1]]: # 从最远的样本开始重新分配
                    found = False
                    for j in range(self.n_clusters):
                        if j != i and np.sum(new_labels == j) < max_size:
                            new_labels[idx] = j
                            found = True
                            break
                    assert found, "Can't find a suitable cluster"
        return new_labels
        
        