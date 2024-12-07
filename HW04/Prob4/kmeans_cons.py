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
        size_constraints: 每个聚类的最小和最大大小约束，shape (2, ) int表示数目, float表示比例
        """
        super().__init__(n_clusters, max_iters, rand)
        if weights is None:
            weights = np.ones(n_clusters)
        self.weights = weights  * n_clusters / np.sum(weights)# 归一化权重
        self.size_constraints = size_constraints

    def get_distances(self, X, centroids):
        """计算每个样本到每个聚类中心的距离 考虑权重
        @return shape (n_samples, n_clusters)
        """
        weighed_X = X * self.weights
        weighed_centroids = centroids * self.weights
        return np.array([euclidean_distance(weighed_X, centroid) for centroid in weighed_centroids])
        

    def assign_labels(self, distances):
        """将每个样本分配到最近的聚类中心 保 证每个簇的大小约束
        @param distances: shape (n_samples, n_clusters)
        """
        pass
