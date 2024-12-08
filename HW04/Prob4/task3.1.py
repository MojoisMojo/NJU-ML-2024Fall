from kmeans_mcons import MConstrainedKMeans
from kmeans_con import ConstrainedKMeans
from dataloader import load_mall_data, deScale
from utils import euclidean_distance, plot_clusters
from plot_see_k import plot_SEE_K
import numpy as np
def task3_1(
    k,
    weights,
    mod="con",
):
    print(f"k = {k}")
    X_scaled, X, _ = load_mall_data()  # 三维的数据
    modelClass = ConstrainedKMeans if mod == "con" else MConstrainedKMeans
    model = modelClass(n_clusters=k, weights=weights, rand=14)
    model.fit(X_scaled)
    centroids = model.centroids
    centroids = deScale(X, centroids)
    plot_clusters(X, model.labels, centroids=centroids)
    print("inertia:", model.inertia_)
    print("centroids:", centroids)
    print("####" * 5)
    return model

if __name__ == "__main__":
    intertia = []
    for k in range(1, 11):
        kmeans = task3_1(k, np.array([1,2,1]), mod="con")
        intertia.append(kmeans.inertia_)
    plot_SEE_K(list(range(1, 11)), intertia)
    