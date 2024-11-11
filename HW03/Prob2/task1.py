from dataloader import load_data
from model import SVCModel
from utils import plot_decision_boundary, print_and_write
if __name__ == "__main__":
    X_train, y_train = load_data()
    svc_model = SVCModel(
        loadpath=None,
        savepath="models/task1.pkl",
        # probability=False
    )
    svc_model.train(X_train, y_train)
    svs = svc_model.get_support_vectors()
    print_and_write("output/task1/out.txt", "支持向量："+str(svs))
    plot_decision_boundary(svc_model, X_train, y_train, "trainProb", "output/task1")