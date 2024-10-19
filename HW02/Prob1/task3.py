from params import RAND_STATE_SEED
from model import SVCModel
from dataloader import DataLoader
from datetime import datetime
import os
"""
% 注意:
% 1. 这个框架提供了基本的结构，您需要完成所有标记为 'pass' 的函数。
% 2. 记得处理数值稳定性问题，例如在计算对数时避免除以零。
% 3. 在报告中详细讨论您的观察结果和任何有趣的发现。
"""
class SMOTE(object):
    def __init__(self , X, y, N, K, random_state = RAND_STATE_SEED):
        self.N = N # 每个小类样本合成样本个数
        self.K = K # 近邻个数
        self.label = y # 进行数据增强的类别
        self.sample = X
        self.n_sample , self.n = X.shape # 获得样本个数, 特征个数
    
    def over_sampling(self):
        pass

def task3(
    run_time,
    data_loader: DataLoader,
    loadpath=None,
):
    task_name = "task3"
    dir_path = f"./output/{run_time}/{task_name}"
    savepath = f"{dir_path}/svm_model.pkl"
    output_path = f"{dir_path}/out.out"
    curve_path = f"{dir_path}/roc_curve.png"
    
    os.makedirs(dir_path, exist_ok=True)
    outfile = open(output_path, "w")  # 清空文件内容
    outfile.close()

    X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2, stratify=False)
    
    smote = SMOTE(X_train, y_train, N=100, K=5)

if __name__ == "__main__":
    data_loader = DataLoader("../data/creditcard.csv")
    run_timestemp = datetime.now().strftime("%m%d_%H%M%S")
    load_path = None
    task3(data_loader, run_timestemp, load_path)