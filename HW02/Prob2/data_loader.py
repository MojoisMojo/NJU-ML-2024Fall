from sklearn.datasets import load_iris, load_breast_cancer, load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from params import RAND_SEED, TEST_SIZE
import pandas as pd
import numpy as np
from typing import Tuple, List


def myHotEncoder(X):
    assert isinstance(X, pd.DataFrame)
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    # 对类别型特征进行独热编码
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_numeric = X[numerical_features].values
    X_prepared = np.hstack((X_numeric, X_encoded.toarray()))
    return X_prepared


def myLabelEncoder(X):
    assert isinstance(X, pd.DataFrame)
    for col in X.columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    return X


data_encoder = myHotEncoder

dataset_names = [
    "iris",
    "cancer",
    "digits",
    "moon",
    "income",
    "wwine",
    "rwine",
    "allwine",
    "car",
    "bank",
    "full_bank",
]


def get_size(dataset_name) -> Tuple[int, int]:
    dateset_wLabel_details: List[Tuple[int, int]] = [
        (4, 3),  # iris
        (30, 2),  # cancer
        (64, 10),  # digits
        (2, 2),  # moon
        (14, 2),  # income
        (11, 11),  # wwine
        (11, 11),  # rwine
        (11, 11),  # allwine
        (6, 4),  # car
        (16, 2),  # bank
        (16, 2),  # full_bank
    ]
    table = dict(zip(dataset_names, dateset_wLabel_details))
    if dataset_name not in table:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return table[dataset_name]


def get_data(dataset_name, encoder="onehot"):
    global data_encoder
    if encoder == "onehot":
        data_encoder = myHotEncoder
    elif encoder == "label":
        data_encoder = myLabelEncoder
    dataset_getters = [
        get_iris_data,
        get_breast_cancer_data,
        get_digits_data,
        get_moon_data,
        get_adult_income_data,
        get_white_wine_data,
        get_red_wine_data,
        get_all_wine_data,
        get_car_eval_data,
        get_bank_data,
        get_full_bank_data,
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
    """
    | class values (不平衡的数据集)
    unacc, acc, good, vgood
    """
    data = pd.read_csv("../data/car_evaluation/car.data", header=None)
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data.columns = columns
    ytable = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
    # 分离特征和标签
    X = data.drop("class", axis=1)
    y = data["class"]
    y = y.apply(lambda s: ytable[s]).values
    X_prepared = data_encoder(X)
    # 将数据分回训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y, test_size=TEST_SIZE, random_state=RAND_SEED
    )
    return X_train, X_test, y_train, y_test


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
    X_prepared = data_encoder(X)

    # 将数据分回训练集和测试集
    X_train = X_prepared[: len(data_train)]
    X_test = X_prepared[len(data_train) :]
    y_train = y[: len(data_train)].values
    y_test = y[len(data_train) :].values

    return X_train, X_test, y_train, y_test


def get_white_wine_data():
    data = pd.read_csv("../data/wine_q/winequality-white.csv", sep=";")
    X = data.drop("quality", axis=1).values
    y = data["quality"].values
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_SEED)


def get_red_wine_data():
    data = pd.read_csv("../data/wine_q/winequality-red.csv", sep=";")
    X = data.drop("quality", axis=1).values
    y = data["quality"].values
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_SEED)


def get_all_wine_data():
    X_white_train, X_white_test, y_white_train, y_white_test = get_white_wine_data()
    X_red_train, X_red_test, y_red_train, y_red_test = get_red_wine_data()
    X_train = np.vstack((X_white_train, X_red_train))
    X_test = np.vstack((X_white_test, X_red_test))
    y_train = np.hstack((y_white_train, y_red_train))
    y_test = np.hstack((y_white_test, y_red_test))
    indices_train = np.arange(len(X_train))
    np.random.shuffle(indices_train)
    indices_test = np.arange(len(X_test))
    np.random.shuffle(indices_test)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]
    return X_train, X_test, y_train, y_test


def get_bank_data():
    data = pd.read_csv("../data/bank_mark/bank/bank.csv", sep=";")
    X = data.drop("y", axis=1)
    y = data["y"]
    y = y.apply(lambda s: 1 if s == "yes" else 0).values
    X_prepared = data_encoder(X)
    return train_test_split(X_prepared, y, test_size=TEST_SIZE, random_state=RAND_SEED)


def get_full_bank_data():
    data = pd.read_csv("../data/bank_mark/bank/bank-full.csv", sep=";")
    X = data.drop("y", axis=1)
    y = data["y"]
    y = y.apply(lambda s: 1 if s == "yes" else 0).values
    X_prepared = data_encoder(X)
    return train_test_split(X_prepared, y, test_size=TEST_SIZE, random_state=RAND_SEED)
