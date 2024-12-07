from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_PCA(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    k = np.max(y) + 1
    for i in range(k):
        plt.scatter(
            X_pca[y == i, 0],
            X_pca[y == i, 1],
            label=f"Cluster {i}",
        )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA")
    
    print("PCAs is: ")
    print(pca.components_)
    
    print("explained variance ratio is: ")
    print(pca.explained_variance_ratio_)
    
    plt.legend()
    plt.grid()
    plt.savefig(f"./images/PCA_k{k}.png")
    plt.close()
