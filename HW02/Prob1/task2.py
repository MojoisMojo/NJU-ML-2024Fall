import datetime
import os
from dataloader import DataLoader
from model import SVCModel
from utils import print_and_write


def task2(
    out_dir, data_loader: DataLoader, loadpath=None, is_train=True, remove_cnt=0
):
    task_name = "task2"
    dir_path = f"./output/{out_dir}/{task_name}"
    output_path = f"{dir_path}/out.out"
    os.makedirs(dir_path, exist_ok=True)

    X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=False)
    # X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=True)

    loadpath = loadpath
    savepath = f"{dir_path}/svm_model_remove{remove_cnt}.pkl"

    curve_path = f"{dir_path}/roc_curve_remove{remove_cnt}.png"

    print_and_write(
        output_path,
        "######################################" * 2
        + f"\nRemove {remove_cnt} negative samples",
    )
    if is_train:
        if remove_cnt > 0:
            X_train_reduced, y_train_reduced = data_loader.reduce_negatives(
                X_train, y_train, remove_cnt
            )
        else:
            X_train_reduced, y_train_reduced = X_train, y_train
            print("No negative samples removed")

        print_and_write(output_path, f"X_train shape: {X_train_reduced.shape}")

    svm_model = SVCModel(loadpath=loadpath, savepath=savepath)

    if loadpath == None or is_train:
        # 训练、保存 SVM 模型
        svm_model.train(X_train_reduced, y_train_reduced)
    # 预测 SVM 模型
    y_pred, y_prob = svm_model.predict(X_test)

    # 计算评估指标 并输出 & 画图
    svm_model.validate_and_print(y_test, y_pred, y_prob, output_path, curve_path)


if __name__ == "__main__":
    run_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    data_loader = DataLoader("../data/creditcard.csv")
    task2(run_time, data_loader, is_train=True, remove_cnt=200000)
    # for remove_count in [2000, 20000, 200000]:
        # task2(run_time, data_loader, remove_cnt=remove_count)
