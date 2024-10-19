import datetime
import os
from model import SVCModel
from dataloader import DataLoader


def task1(run_time, data_loader: DataLoader, loadpath=None, is_test=False):
    task_name = "task1"
    dir_path = f"./output/{run_time}/{task_name}"
    savepath = f"{dir_path}/svm_model.pkl"
    output_path = f"{dir_path}/out.out"
    os.makedirs(dir_path, exist_ok=True)
    out_file = open(output_path, "w")  # 清空文件内容
    out_file.close()
    curve_path = f"{dir_path}/roc_curve.png"

    X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=False)
    # X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=True)

    svm_model = SVCModel(loadpath=loadpath, savepath=savepath)

    if loadpath == None or not is_test:
        # 训练、保存 SVM 模型
        svm_model.train(X_train, y_train)
    # 预测 SVM 模型
    y_pred, y_prob = svm_model.predict(X_test)

    # 计算评估指标 并输出 & 画图
    svm_model.validate_and_print(y_test, y_pred, y_prob, output_path, curve_path)


if __name__ == "__main__":
    run_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    data_loader = DataLoader("../data/creditcard.csv")
    load_path = None
    task1(run_time, data_loader)
