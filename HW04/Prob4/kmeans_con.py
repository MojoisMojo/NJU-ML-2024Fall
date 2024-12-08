from kmeans_pp import KMeansPP
from utils import euclidean_distance
import numpy as np
from k_means_constrained import KMeansConstrained


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
        self.model = None

    def fit(self, X: np.ndarray):
        """
        使用约束条件训练模型
        """
        if self.size_constraints is None and self.weights is None:
            super().fit(X)
            return
        n_samples, n_features = X.shape
        size_min = 0
        size_max = n_samples
        weighed_X = X
        if self.weights is not None:
            assert (
                len(self.weights) == n_features
            ), "weights shape should be (n_features,)"
            self.weights = (
                self.weights * n_features / np.sum(self.weights)
            )  # 归一化权重
            weighed_X = weighed_X * self.weights
        if self.size_constraints is not None:
            size_min, size_max = self.size_constraints
            size_min = max(0, int(size_min * n_samples))
            size_max = min(n_samples, int(size_max * n_samples))
            print(f"size_min: {size_min}, size_max: {size_max}")
        self.model = KMeansConstrained(
            n_clusters=self.n_clusters,
            max_iter=self.max_iters,
            random_state=self.rand,
            n_init=1,
            size_max=size_max,
            size_min=size_min,
            init="k-means++",
        )
        self.model.fit(weighed_X)
        self.labels = np.array(self.model.labels_)
        if self.weights is not None:
            self.centroids = self.model.cluster_centers_ / self.weights
        else:
            self.centroids = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        self.origin_inertia_ = np.sum(
            [
                ((X[self.labels == i] - self.centroids[i]) ** 2).sum()
                for i in range(self.n_clusters)
            ]
        )
