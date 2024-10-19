### `SVC` 类的构造函数参数

```python
class sklearn.svm.SVC(
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None
)
```

### 参数说明

- `C`: 正则化参数。默认值为 1.0。C 值越大，对误分类的惩罚越大。
- `kernel`: 核函数类型。可选值有 `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'precomputed'`。默认值为 `'rbf'`。
- `degree`: 多项式核函数的维度（`kernel='poly'` 时有效）。默认值为 3。
- `gamma`: 核函数系数。可选值有 `'scale'`, `'auto'` 或浮点数。默认值为 `'scale'`。
- `coef0`: 核函数中的独立项。对 `'poly'` 和 `'sigmoid'` 核函数有效。默认值为 0.0。
- `shrinking`: 是否使用启发式收缩。默认值为 `True`。
- `probability`: 是否启用概率估计。启用后会增加计算开销。默认值为 `False`。
- `tol`: 停止训练的误差容忍度。默认值为 `1e-3`。
- `cache_size`: 缓存大小（以 MB 为单位）。默认值为 200。
- `class_weight`: 类别权重。可以是字典或 `'balanced'`。默认值为 `None`。
- `verbose`: 是否启用详细输出。默认值为 `False`。
- `max_iter`: 最大迭代次数。默认值为 -1（表示无限制）。
- `decision_function_shape`: 决策函数形状。可选值有 `'ovr'`（一对多）和 `'ovo'`（一对一）。默认值为 `'ovr'`。
- `break_ties`: 是否在预测时打破平局。默认值为 `False`。
- `random_state`: 随机数种子。默认值为 `None`。

### 常用方法

- `fit(X, y)`: 训练模型。
- `predict(X)`: 使用训练好的模型进行预测。
- `predict_proba(X)`: 返回每个类的概率估计（需要 `probability=True`）。
- `decision_function(X)`: 计算样本到决策边界的距离。
- `score(X, y)`: 返回模型在给定测试集上的准确率。

### 参考链接

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

https://zhuanlan.zhihu.com/p/134091240

https://www.cnblogs.com/linjingyg/p/15708635.html

### 示例代码

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 SVC 模型
model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
```

以上是 `SVC` 的主要接口和参数说明，以及一个简单的示例代码。你可以根据需要调整参数来优化模型。