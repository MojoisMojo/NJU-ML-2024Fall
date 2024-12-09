import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y):
        # TODO
        self.is_fitted_ = True

    def predict(self, X):
        # TODO
        return np.full(shape = X.shape[0], fill_value=self.param)

estimator = MyEstimator(param=1)
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

estimator.fit(X, y)
print(estimator.predict(X))
print(estimator.score(X,y,sample_weight=[1,1,2]))