from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from params import RANDOM_SEED

# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
