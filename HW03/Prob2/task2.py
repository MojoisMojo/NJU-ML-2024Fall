from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os
from utils import print_and_write

from params import CACHE_SIZE, RAND_STATE_SEED

from dataloader import load_data

X_train, y_train = load_data()

# 定义参数网格
param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.1, 1, 10], "kernel": ["rbf"]}

# 创建 SVM 分类器
svc = SVC(cache_size=CACHE_SIZE, random_state=RAND_STATE_SEED)

write_dir = "output/task2"
os.makedirs(write_dir, exist_ok=True)

write_file = os.path.join(write_dir, "out.txt")
with open (write_file, "w") as f:
    f.write("")

# 网格搜索
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 最佳参数
best_params = grid_search.best_params_
print_and_write(
    write_file, f"最佳参数：C={best_params['C']}, gamma={best_params['gamma']}"
)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_

# 计算训练准确率
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print_and_write(write_file, f"训练准确率：{train_accuracy:.4f}")

# 输出支持向量数量
n_support = best_model.n_support_
print_and_write(write_file, f"每类的支持向量数量：{n_support}")
print_and_write(write_file, f"总支持向量数量：{sum(n_support)}")
