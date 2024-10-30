from sklearn.datasets import load_iris, load_breast_cancer, load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from params import RAND_SEED, TEST_SIZE
import pandas as pd


dataset_names = ["iris", "cancer", "digits", "car", "moon", "income"]


def get_data(dataset_name):
    dataset_getters = [
        get_iris_data,
        get_breast_cancer_data,
        get_digits_data,
        get_car_eval_data,
        get_moon_data,
        get_adult_income_data,
    ]
    table = dict(zip(dataset_names, dataset_getters))
    if dataset_name not in table:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return table[dataset_name]()


def get_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_SEED
    )
    return X_train, X_test, y_train, y_test


def get_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_SEED
    )
    return X_train, X_test, y_train, y_test


def get_digits_data():
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_SEED
    )
    return X_train, X_test, y_train, y_test


def get_moon_data():
    X, y = make_moons(n_samples=3000, noise=0.31, random_state=RAND_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_SEED
    )
    return X_train, X_test, y_train, y_test


def get_car_eval_data():
    pass


def get_adult_income_data():
    # 读取训练和测试数据
    data_train = pd.read_csv(
        "../data/adult/adult.data", header=None, skipinitialspace=True
    )
    data_test = pd.read_csv(
        "../data/adult/adult.test", header=None, skiprows=1, skipinitialspace=True
    )
    data = pd.concat([data_train, data_test], ignore_index=True)

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data.columns = columns

    # 分离特征和标签
    X = data.drop("income", axis=1)
    y = data["income"]
    y = y.apply(lambda s: 1 if (s == ">50K" or s == ">50K.") else 0)
    # 识别类别型特征和数值型特征
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    # 对类别型特征进行独热编码
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_features])

    # 将数值型特征和编码后的类别型特征合并
    import numpy as np

    X_numeric = X[numerical_features].values
    X_prepared = np.hstack((X_numeric, X_encoded.toarray()))

    # 将数据分回训练集和测试集
    X_train = X_prepared[: len(data_train)]
    X_test = X_prepared[len(data_train) :]
    y_train = y[: len(data_train)].values
    y_test = y[len(data_train) :].values

    return X_train, X_test, y_train, y_test
