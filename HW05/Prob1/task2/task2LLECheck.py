import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

N_SAMPLES = 3000
RANDOM_SEED = 0
X, t = make_swiss_roll(n_samples=N_SAMPLES, random_state=RANDOM_SEED)

def plt_ori():
    # 画出原始数据的三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral)
    plt.title('Original data')
    plt.savefig('task2OriginalData.png')
    plt.close()

def task(neighbors):
    f = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=2)
    X_r = f.fit_transform(X)
    plt.scatter(X_r[:, 0], X_r[:, 1], c=t, cmap=plt.cm.Spectral)
    plt.title('LLE')
    plt.savefig(f'task2LLECheck_nei_{neighbors}.png')
    plt.close()

if __name__ == '__main__':
    plt_ori()
    for nei in [5, 7, 12, 15, 20, 30, 50, 75, 100]:
        task(nei)
