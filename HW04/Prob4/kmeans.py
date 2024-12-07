from dataloader import load_mall_data, deScale
from utils import euclidean_distance, plot_clusters
import numpy as np

"""
在不借助外部实现的情况下，手动实现 KMeans 方法，
• 在数据集上进行聚类，可视化聚类结果；  ---- 绘制三维散点图 + 二维散点图
• 并解决下列问题 :
    • 如何使用肘部法则确定合适的 k 值，绘图说明 ---- 绘制一个 y轴为所有簇内平方和，x轴为k值的折线图，找到拐点
    • 简单分析每个客户群的特征        ---- 
    • 计算和分析簇内平方和 (inertia)
"""


class KMeans:
    def __init__(self, n_clusters=3, max_iters=200, rand=None):
        if rand is not None:
            np.random.seed(rand)
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.iter_count = None
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # 簇内平方和

    def initialize_centroids(self, X):
        """使用随机算法初始化聚类中心"""
        n_clusters = self.n_clusters
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def get_distances(self, X, centroids):
        """计算每个样本到每个聚类中心的距离
        @return shape (n_samples, n_clusters)
        """
        return np.array([euclidean_distance(X, centroid) for centroid in centroids])

    def assign_labels(self, distances):
        """将每个样本分配到最近的聚类中心
        @param distances: shape (n_samples, n_clusters)
        """
        return np.argmin(distances, axis=0)

    def fit(self, X: np.ndarray):
        """
        实现K-means算法
        参数:
            X: shape (n_samples, n_features)
        返回:
            self
        """
        # TODO: 实现K-means算法
        inertias_ = np.zeros(self.n_clusters)
        labels = np.zeros(X.shape[0])
        iter_count = 0
        # 1. 随机初始化聚类中心 2种方法，随机选择和KMeans++
        centroids = self.initialize_centroids(X)
        # 2. 迭代优化直到收敛
        while iter_count < self.max_iters:
            iter_count += 1
            # 2.1 计算每个样本到聚类中心的距离
            distance = self.get_distances(X, centroids)
            # 2.2 将每个样本分配到最近的聚类中心
            labels = self.assign_labels(distance)
            # 2.3 更新聚类中心
            new_centroids = np.array(
                [
                    (
                        X[labels == i].mean(axis=0)
                        if len(X[labels == i]) > 0
                        else centroids[i]
                    )
                    for i in range(self.n_clusters)
                ]
            )
            if np.all(new_centroids == centroids):
                break
            centroids = new_centroids

        self.iter_count = iter_count

        # 3 计算簇内平方和
        for i in range(self.n_clusters):
            inertias_[i] = ((X[labels == i] - centroids[i]) ** 2).sum()

        self.centroids = centroids
        self.labels = labels
        self.inertia_ = inertias_.sum()

    def predict(self, X):
        """返回每个样本的聚类标签"""
        return self.labels


def task(k):
    print(f"KMeans, k  = {k}")
    X_scaled, X, _ = load_mall_data()  # 三维的数据
    kmeans = KMeans(n_clusters=k, rand=14)
    kmeans.fit(X_scaled)
    centroids = deScale(X, kmeans.centroids)
    plot_clusters(X, kmeans.labels, centroids=centroids)
    print("inertia:", kmeans.inertia_)
    print("centroids:", centroids)
    print("####" * 5)
    return kmeans


if __name__ == "__main__":
    task(5)
