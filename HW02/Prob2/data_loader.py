from sklearn.datasets import load_iris, load_breast_cancer, load_digits

# class DataLoader:
#     def __init__(self, data_path = None):
#         self.data_path = data_path

#     def load_data(self):
#         if self.data_path is None:
#             print("Data path is not specified.")
#             return None
#         with open(self.data_path, 'r') as f:
#             data = f.readlines()
#         return data


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
