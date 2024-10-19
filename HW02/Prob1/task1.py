import numpy as np
import pandas as pd
from sklearn.svm import SVC

import datetime
import os
from utils import validate, plot_roc_curve, print_and_write
from model import train, predict
from dataloader import DataLoader


run_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
dir_path = f"../output/{run_time}"
task_path = f"{dir_path}/task1.out"
os.makedirs(dir_path, exist_ok=True)
out_file = open(task_path, "w")  # 清空文件内容
out_file.close()
out_file = open(task_path, "a")  # 追加写入

# 读取数据
dataloader = DataLoader("../data/creditcard.csv")
X_train, X_test, y_train, y_test = dataloader.split(test_size=0.2, stratify=True)

svm_model = SVC(
    probability=True,
    cache_size=1000,
)

# 训练 SVM 模型
train(svm_model, X_train, y_train)
# 预测
y_pred, y_prob = predict(svm_model, X_test)

# 计算评估指标
accuracy, recall, f1, auc = validate(y_test, y_pred, y_prob)
print_and_write(f"Accuracy: {accuracy:.4f}")
print_and_write(f"Recall: {recall:.4f}")
print_and_write(f"F1 Score: {f1:.4f}")
print_and_write(f"AUC: {auc:.4f}")

# 绘制 ROC 曲线
plot_roc_curve(y_test, y_prob, f"{dir_path}/roc_curve_task1.png")
