from sklearn.svm import SVC
from params import CACHE_SIZE
from utils import mytqdm, validate, print_and_write, plot_roc_curve
import joblib
import os


class SVCModel:
    def __init__(
        self,
        loadpath=None,
        savepath=None,
        probability=True,
        cache_size=CACHE_SIZE,
    ):
        self.haveprob = probability
        self.savepath = savepath
        if loadpath:
            try:
                self.load(loadpath)
                print(f"Model loaded from {loadpath}")
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
        if self.savepath:
            self.save(self.savepath)
            print(f"Model saved to {self.savepath}")

    def predict(self, X_test):
        if self.haveprob:
            return self.model.predict_proba(X_test)[:, 1]
        return self.model.predict(X_test)
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        assert os.path.exists(path), "Model not found"
        self.model = joblib.load(path)

    def validate(self, y_test, y_pred, y_prob):
        return validate(y_test, y_pred, y_prob)
    
    def get_support_vectors(self):
        return self.model.support_vectors_

    def validate_and_print(self, y_test, y_pred, y_prob, file_path, curve_path):
        accuracy, precision, recall, f1, auc = self.validate(y_test, y_pred, y_prob)
        print_and_write(file_path, f"Accuracy: {accuracy:.6f}")
        print_and_write(file_path, f"Precision: {precision:.6f}")
        print_and_write(file_path, f"Recall: {recall:.6f}")
        print_and_write(file_path, f"F1 Score: {f1:.6f}")
        print_and_write(file_path, f"AUC: {auc:.6f}")
        plot_roc_curve(y_test, y_prob, curve_path)
        return accuracy, precision, recall, f1, auc
