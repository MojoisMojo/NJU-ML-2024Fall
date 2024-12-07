from dataloader import load_mall_data, deScale
from utils import (
    euclidean_distance,
    plot_clusters,
    compute_silhouette_score,
    compute_cluster_stability_and_iters,
)
from kmeans import KMeans
import numpy as np

"""
在不借助外部实现的情况下，手动实现 KMeans 方法，
• 在数据集上进行聚类，可视化聚类结果；  ---- 绘制三维散点图 + 二维散点图
• 并解决下列问题 :
    • 如何使用肘部法则确定合适的 k 值，绘图说明 ---- 绘制一个 y轴为所有簇内平方和，x轴为k值的折线图，找到拐点
    • 简单分析每个客户群的特征        ---- 
    • 计算和分析簇内平方和 (inertia)
"""


class KMeansPP(KMeans):
    def initialize_centroids(self, X):
        """使用KMeans++算法初始化聚类中心"""
        n_clusters = self.n_clusters
        centroids = np.zeros((n_clusters, X.shape[1]))
        # 随机选择第一个质心
        centroids[0] = X[np.random.choice(X.shape[0])]

        for i in range(1, n_clusters):
            # 计算每个样本到最近质心的距离
            distances = np.min(
                [np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]],
                axis=0,
            )
            # 根据距离的平方选择下一个质心
            probabilities = distances**2 / np.sum(distances**2)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = X[j]
                    break
        return centroids


def task2(k):
    print(f"KMeans, k  = {k}")
    X_scaled, X, _ = load_mall_data()  # 三维的数据
    kmeans = KMeansPP(n_clusters=k, rand=14)
    kmeans.fit(X_scaled)
    centroids = deScale(X, kmeans.centroids)
    plot_clusters(X, kmeans.labels, centroids=centroids)
    silhouette_score = compute_silhouette_score(X, kmeans.labels) if k > 1 else None
    print("inertia:", kmeans.inertia_)
    print("centroids:", centroids)
    print(f"Silhouette Score: {silhouette_score}")
    print("####" * 5)
    return kmeans


if __name__ == "__main__":
    task2(5)
