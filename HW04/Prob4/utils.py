import numpy as np
from mPCA import plot_PCA
import matplotlib.pyplot as plt


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def plot_clusters(X, labels, centroids=None):
    """绘制聚类结果的散点图"""
    # 根据 聚类结果 为 每个簇绘制散点图 颜色不同
    # 图片存放到 ./images/ 文件夹下

    k = np.max(labels) + 1

    def plot_3d():
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(k):
            ax.scatter(
                X[labels == i, 0],
                X[labels == i, 1],
                X[labels == i, 2],
                label=f"Cluster {i}",
            )

        if centroids is not None:
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2],
                c="black",
                s=50,
                marker="x",
                label="Centroids",
            )

        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Income (k$)")
        ax.set_zlabel("Spending Score (1-100)")

        ax.legend()
        plt.savefig(f"./images/cluster_result_k{k}_3d.png")
        plt.close(fig)

    feature_names = ["Age", "Annual-Income(k$)", "Spending-Score(1-100)"]

    def plot_2d(i, j):
        fig = plt.figure(figsize=(6, 6))
        for c in range(k):
            plt.plot(X[labels == c, i], X[labels == c, j], "o", label=f"Cluster {c}")
        if centroids is not None:
            plt.plot(centroids[:, i], centroids[:, j], "kx", markersize=10)
        plt.xlabel(f"{feature_names[i]}")
        plt.ylabel(f"{feature_names[j]}")
        plt.legend()
        plt.savefig(f"./images/cluster_result_k{k}_{i}_{j}.png")
        plt.close(fig)

    plot_3d()
    plot_2d(0, 1)
    plot_2d(0, 2)
    plot_2d(1, 2)
    plot_PCA(X, labels)


from sklearn.metrics import silhouette_score, adjusted_rand_score


def compute_silhouette_score(X, labels):
    """计 算 轮 廓 系 数
    值越大表示聚类效果越好"""
    return silhouette_score(X, labels)


def compute_cluster_stability_and_iters(X, k, model, n_runs=50):
    """评 估 聚 类 结 果 的 稳 定 性
    值越大：表示结果越一致，稳定性越好
    """
    labels = []
    iters = []
    for _ in range(n_runs):
        kmeans = model(n_clusters=k)
        kmeans.fit(X)
        labels.append(kmeans.labels)
        iters.append(kmeans.iter_count)
    stability_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            score = adjusted_rand_score(labels[i], labels[j])
            stability_scores.append(score)

    return np.mean(stability_scores), np.mean(iters)
