from dataloader import load_mall_data
from kmeans_pp import KMeansPP, task2
import numpy as np

from plot_see_k import plot_SEE_K

if __name__ == '__main__':
    Ks = list(range(1, 21))
    intertia = []
    for k in Ks:
        kmeans = task2(k)
        intertia.append(kmeans.inertia_)
    plot_SEE_K(Ks, intertia)