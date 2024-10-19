import datetime
import os
from dataloader import DataLoader
from model import SVCModel
from utils import print_and_write


def task2(
    run_time,
    data_loader: DataLoader,
    loadpaths=[None, None, None],
    is_test=False
):
    task_name = "task2"
    dir_path = f"./output/{run_time}/{task_name}"
    task_path = f"{dir_path}/out.out"
    os.makedirs(dir_path, exist_ok=True)
    out_file = open(task_path, "w")  # 清空文件内容
    out_file.close()

    remove_cnts = [2000, 20000, 200000]

    X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=False)
    # X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=True)

    def undersampling_train_and_test(i, X_train, y_train):

        remove_cnt = remove_cnts[i]
        loadpath = loadpaths[i]
        savepath = f"{dir_path}/svm_model_remove{remove_cnt}.pkl"

        curve_path = f"{dir_path}/roc_curve_remove{remove_cnt}.png"
        print_and_write(
            task_path,
            "######################################" * 2
            + f"\nRemove {remove_cnt} positive samples",
        )
        X_train_reduced, y_train_reduced = data_loader.reduce_positives(
            X_train, y_train, remove_cnt
        )

        svm_model = SVCModel(loadpath=loadpath, savepath=savepath)

        if loadpath == None or not is_test:
            # 训练、保存 SVM 模型
            svm_model.train(X_train, y_train)
        # 预测 SVM 模型
        y_pred, y_prob = svm_model.predict(X_test)

        # 计算评估指标 并输出 & 画图
        svm_model.validate_and_print(y_test, y_pred, y_prob, task_path, curve_path)

    for i in range(3):
        undersampling_train_and_test(i, X_train, y_train)


if __name__ == "__main__":
    run_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    data_loader = DataLoader("../data/creditcard.csv")
    task2(run_time, data_loader)
