import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from params import RAND_SEED
# 假设您的数据加载如下
# data = pd.read_csv('your_data.csv', header=None)

# 读取训练和测试数据
data_train = pd.read_csv('../data/adult/adult.data', header=None, skipinitialspace=True)
data_test = pd.read_csv('../data/adult/adult.test', header=None, skiprows=1, skipinitialspace=True)

# 合并训练和测试数据以确保编码一致
data = pd.concat([data_train, data_test], ignore_index=True)

# 列名称（根据数据集实际情况调整）
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data.columns = columns

# 分离特征和标签
X = data.drop('income', axis=1)
y = data['income']
y = y.apply(lambda s: 1 if (s == '>50K' or s == '>50K.') else 0)
# 识别类别型特征和数值型特征
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# 对类别型特征进行独热编码
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[categorical_features])

# 将数值型特征和编码后的类别型特征合并
import numpy as np
X_numeric = X[numerical_features].values
X_prepared = np.hstack((X_numeric, X_encoded.toarray()))

# 将数据分回训练集和测试集
X_train = X_prepared[:len(data_train)]
X_test = X_prepared[len(data_train):]
y_train = y[:len(data_train)].values
y_test = y[len(data_train):].values



# # 训练决策树模型
# clf = DecisionTreeClassifier(random_state=RAND_SEED, criterion='gini')
# clf.fit(X_train, y_train)

# # 预测并计算准确率
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f'测试集准确率: {accuracy:.4f}')