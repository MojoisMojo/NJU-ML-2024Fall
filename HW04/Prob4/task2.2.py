from kmeans_pp import KMeansPP
from dataloader import load_mall_data
from kmeans import KMeans
from utils import compute_silhouette_score, compute_cluster_stability_and_iters
import numpy as np
import matplotlib.pyplot as plt

def task2_2(k):
    # 比较两种方法的 稳定性 和 收敛速度
    print(f"k = {k}")
    X_scaled, X, _ = load_mall_data()  # 三维的数据
    print(f"KMean")
    # 聚类稳定性评估
    cluster_stability, iter = compute_cluster_stability_and_iters(X, k, KMeans)
    print(f"KMeans++")
    cluster_stability_pp, iter_pp = compute_cluster_stability_and_iters(X, k, KMeansPP)
    print(f"KMeans Cluster Stability: {cluster_stability}")
    print(f"KMeans++ Cluster Stability: {cluster_stability_pp}")
    print("####" * 5)
    return cluster_stability, cluster_stability_pp, iter, iter_pp

def plot_stabilities(stabilities, stabilities_pp):
    # 绘制稳定性对比图 存放在 images/stabilities_k{k}.png
    fig = plt.figure(figsize=(8, 6))
    plt.plot(list(range(2, 21)), stabilities, label="KMeans")
    plt.plot(list(range(2, 21)), stabilities_pp, label="KMeans++")
    plt.xlabel("k")
    plt.ylabel("Cluster Stability")
    plt.legend()
    plt.title("Cluster Stability vs K")
    plt.grid()
    plt.savefig(f"./images/stabilities_k{k}.png")
    plt.close(fig)

def plot_iters(iters, iters_pp):
    # 绘制收敛速度对比图 存放在 images/iters_k{k}.png
    fig = plt.figure(figsize=(8, 6))
    plt.plot(list(range(2, 21)), iters, label="KMeans")
    plt.plot(list(range(2, 21)), iters_pp, label="KMeans++")
    plt.xlabel("k")
    plt.ylabel("Iterations")
    plt.legend()
    plt.title("Iterations vs K")
    plt.grid()
    plt.savefig(f"./images/iters_k{k}.png")
    plt.close(fig)

if __name__ == "__main__":
    stabilities, stabilities_pp, iters, iters_pp = [], [], [], []
    for k in list(range(2, 21)):
        stability, stability_pp, iter, iter_pp = task2_2(k)
        stabilities.append(stability)
        stabilities_pp.append(stability_pp)
        iters.append(iter)
        iters_pp.append(iter_pp)
    
    plot_stabilities(stabilities, stabilities_pp)
    plot_iters(iters, iters_pp)
    
