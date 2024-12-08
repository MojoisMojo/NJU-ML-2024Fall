from kmeans_pp import KMeansPP
from utils import euclidean_distance
import numpy as np


class MConstrainedKMeans(KMeansPP):
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
        self.weights = weights
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

    def fit(self, X: np.ndarray):
        if self.size_constraints is None and self.weights is None:
            super().fit(X)
            return
        n_samples, n_features = X.shape
        weighed_X = X
        if self.weights is not None:
            assert (
                len(self.weights) == n_features
            ), "weights shape should be (n_features,)"
            self.weights = (
                self.weights * n_features / np.sum(self.weights)
            )  # 归一化权重
            weighed_X = X * self.weights

        super().fit(weighed_X)
        if self.weights is not None:
            self.centroids = self.centroids / self.weights
        self.origin_inertia_ = sum(
            [
                ((X[self.labels == i] - self.centroids[i]) ** 2).sum()
                for i in range(self.n_clusters)
            ]
        )
        return

    def assign_labels(self, distances, past_labels) -> np.ndarray:
        """将每个样本分配到最近的聚类中心 保证每个簇的大小约束
        @param distances: shape (n_clusters, n_samples)
        """
        if self.size_constraints is None:
            return super().assign_labels(distances, past_labels)
        n_clusters = distances.shape[0]
        n_samples = distances.shape[1]
        min_size, max_size = self.size_constraints
        min_size = max(0, int(min_size * n_samples))
        max_size = min(n_samples, int(max_size * n_samples))
        print(min_size, max_size)

        # 对distance排序
        labels = np.array([-1] * n_samples)
        cnts = np.array([0] * n_clusters)
        sorted_indices = sorted(
            [
                (distances[j][i], j, i)
                for j in range(n_clusters)
                for i in range(n_samples)
            ]
        )
        # for d, j, i in sorted_indices:
        #     if labels[i] != -1:
        #         continue
        #     if cnts[j] >= min_size:
        #         continue
        #     labels[i] = j
        #     cnts[j] += 1
        # # 注意这里结束后，label可能有-1 但是我们这里确保了每个簇大小为min_size
        # assert np.all(cnts == min_size), "Some clusters are smaller than min_size"

        for d, j, i in sorted_indices:
            if labels[i] != -1:
                continue
            if cnts[j] >= max_size:
                continue
            labels[i] = j
            cnts[j] += 1

        assert np.all(labels != -1), "Some samples are not assigned to any cluster"
        assert np.all(cnts <= max_size), "Some clusters are larger than max_size"

        for d, j, i in sorted_indices:
            if cnts[j] >= min_size:
                continue
            past = labels[i]
            if cnts[past] <= min_size:
                continue
            labels[i] = j
            cnts[j] += 1
            cnts[past] -= 1

        assert np.all(cnts >= min_size), "Some clusters are smaller than min_size"

        return labels
