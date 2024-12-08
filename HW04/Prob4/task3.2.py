from kmeans_mcons import MConstrainedKMeans
from kmeans_con import ConstrainedKMeans
from dataloader import load_mall_data, deScale
from utils import euclidean_distance, plot_clusters
from plot_see_k import plot_SEE_K
import numpy as np


def task3_2(
    k,
    min_size=0.2,
    mod="con",
):
    print(f"k = {k}")
    X_scaled, X, _ = load_mall_data()  # 三维的数据
    modelClass = ConstrainedKMeans if mod == "con" else MConstrainedKMeans
    model = modelClass(
        max_iters = 400,
        n_clusters=k, size_constraints=np.array([min_size, 1.0]), rand=14
    )
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
    # 由于最小值约束0.2, 所以 k <= 5
    for k in range(1, 6):
        kmeans = task3_2(k, mod="mcon")
        intertia.append(kmeans.inertia_)
        print("iter_count:", kmeans.iter_count)
    plot_SEE_K(list(range(1, 6)), intertia)
