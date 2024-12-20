from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np

# 1. 加 载 LFW 数 据 集
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
# 2. 获 取 数 据
X = lfw_people.data


# Todo1: 写 出PCA函 数
def PCA(X, n_components=5):
    # step1: 中心化
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    # step2: 计算协方差矩阵
    cov_matrix = X_centered.T @ X_centered
    # step3: 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # step4: 选取最大的n_components个特征值所对应的特征向量
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    eigenfaces = eigenvectors[:, idx]
    # step5: 将(n_features,n_components) 转为 (n_components, n_features)
    eigenfaces = eigenfaces.T
    return eigenfaces


eigenfaces = PCA(X)
# Todo2: 可视化5个主成分对应的特征脸
fig, axes = plt.subplots(1, 5, figsize=(15, 8), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(5):
    print(eigenfaces[i])
    # 默认情况下，数值越大，颜色越亮（接近白色）；数值越小，颜色越暗（接近黑色）
    axes[i].imshow(eigenfaces[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    axes[i].set_title(f'Component {i+1}')
plt.savefig('task1_Eigenfaces.png')
plt.close()
