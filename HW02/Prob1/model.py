import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from params import CACHE_SIZE
import matplotlib.pyplot as plt
import datetime
import os
from utils import mytqdm, validate, print_and_write, plot_roc_curve
import joblib


class SVCModel:
    def __init__(
        self,
        loadpath=None,
        savepath=None,
        probability=True,
        cache_size=CACHE_SIZE,
    ):
        self.savepath = savepath
        if loadpath:
            try:
                self.load(loadpath)
            except:
                print("Model not found, creating a new model")
                self.model = SVC(probability=probability, cache_size=cache_size)
        else:
            self.model = SVC(probability=probability, cache_size=cache_size)

    def train(self, X_train, y_train, tqdm=True):
        if not tqdm:
            self.model.fit(X_train, y_train)
            return
        for _ in mytqdm(range(1), desc="Training SVM model"):
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test), self.model.predict_proba(X_test)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def run(self, X_train, y_train, X_test):
        self.train(X_train, y_train)
        if self.savepath:
            self.save(self.savepath)
            print(f"Model saved to {self.savepath}")
        return self.predict(X_test)

    def validate(self, y_test, y_pred, y_prob):
        return validate(y_test, y_pred, y_prob)

    def validate_and_print(self, y_test, y_pred, y_prob, out_file, curve_path):
        accuracy, recall, f1, auc = self.validate(y_test, y_pred, y_prob)
        print_and_write(out_file, f"Accuracy: {accuracy:.4f}")
        print_and_write(out_file, f"Recall: {recall:.4f}")
        print_and_write(out_file, f"F1 Score: {f1:.4f}")
        print_and_write(out_file, f"AUC: {auc:.4f}")
        plot_roc_curve(y_test, y_prob, curve_path)
        return accuracy, recall, f1, auc
