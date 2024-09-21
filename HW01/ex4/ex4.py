# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
# X,y = load_boston(return_X_y=True)

try:
    if os.path.exists('boston_housing.csv'):
        df = pd.read_csv('boston_housing.csv')
        y = df['MEDV'] # 标签-房价
        X = df.drop(['MEDV'], axis=1) #去掉标签（房价）的数据子集
        trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)
    else:
        raise FileNotFoundError("boston_housing.csv not found")
except Exception as e:
    print(e)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    """ 长这样：
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.

    Variables in order:
    CRIM     per capita crime rate by town
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS    proportion of non-retail business acres per town
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX      nitric oxides concentration (parts per 10 million)
    RM       average number of rooms per dwelling
    AGE      proportion of owner-occupied units built prior to 1940
    DIS      weighted distances to five Boston employment centres
    RAD      index of accessibility to radial highways
    TAX      full-value property-tax rate per $10,000
    PTRATIO  pupil-teacher ratio by town
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT    % lower status of the population
    MEDV     Median value of owner-occupied homes in $1000's

    0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30
    396.90   4.98  24.00
    0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80
    396.90   9.14  21.60
    """
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    """
    raw_df.values[::2, :]：选择 raw_df 中所有偶数行（从第0行开始，每隔一行取一次），并选择所有列。
    raw_df.values[1::2, :2]：选择 raw_df 中所有奇数行（从第1行开始，每隔一行取一次），并选择前两列。
    np.hstack([...])：将上述两个数组水平堆叠（按列拼接）在一起，形成一个新的数组 X。
    这里也就是去掉目标房价MEDV后的内容
    """
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # 选择 raw_df 中所有奇数行（从第1行开始，每隔一行取一次），并选择第3列（索引为2） 也就是目标房价MEDV
    y = raw_df.values[1::2, 2]
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)
    # 将特征矩阵和目标向量组合成一个 DataFrame
    df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=[
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ])

    # 保存为 CSV 文件
    df.to_csv('boston_housing.csv', index=False)
    print("boston_housing.csv saved")

def linear_regression(X_train, y_train):
    """线性回归

    Args:
        X_train (np.ndarray): n X d
        y_train (np.ndarray): n X 1
    返回：权重矩阵W
    """
    # 使用方程计算 W
    # W = (X^T * X)^-1 * X^T * y
    W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    return W


def MSE(X_train, y_train, X_test, y_test):
    """调用linear_regression得到权重矩阵W后计算均方误差MSE

    Args:
        X_train (np.ndarray): n X d
        y_train (np.ndarray): n X 1
        X_test (np.ndarray): n X d
        y_test (np.ndarray): n X 1
    返回：标量，均方误差MSE的值
    """
    W = linear_regression(X_train, y_train)
    y_pred = X_test @ W
    MSE = np.mean((y_pred - y_test) ** 2)
    return MSE


linear_regression_MSE = MSE(trainx, trainy, testx, testy)
print(linear_regression_MSE)  # 输出: 23.60384463049043
