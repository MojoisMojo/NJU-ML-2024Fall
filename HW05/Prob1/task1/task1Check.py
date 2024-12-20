from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

# 1. 加载LFW数据集
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
# 2. 获取数据
X = lfw_people.data

# Todo1: 写出PCA函数
def PCA(X, n_components=5):
    pca = sklearnPCA(n_components=n_components)
    pca.fit(X)
    return pca.components_

eigenfaces = PCA(X)

# Todo2: 可视化5个主成分对应的特征脸
fig, axes = plt.subplots(1, 5, figsize=(15, 8), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(5):
    print(eigenfaces[i])
    # 默认情况下，数值越大，颜色越亮（接近白色）；数值越小，颜色越暗（接近黑色）
    axes[i].imshow(eigenfaces[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    axes[i].set_title(f'Component {i+1}')
plt.savefig('task1Check.png')
plt.close()