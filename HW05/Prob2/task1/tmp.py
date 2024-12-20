from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
print(X[0])