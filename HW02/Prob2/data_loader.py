from sklearn.datasets import load_iris, load_breast_cancer, load_digits

dataset_names = ["iris", "breast_cancer", "digits", "car_eval"]


def get_data(dataset_name):
    dataset_getters = [
        get_iris_data,
        get_breast_cancer_data,
        get_digits_data,
        get_car_eval_data,
    ]
    table = dict(zip(dataset_names, dataset_getters))
    if dataset_name not in table:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return table[dataset_name]()


def get_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    return X, y


def get_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    return X, y


def get_digits_data():
    data = load_digits()
    X, y = data.data, data.target
    return X, y


def get_car_eval_data():
    pass
