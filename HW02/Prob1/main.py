from task1 import task1
from task2 import task2
from task3 import task3
from dataloader import DataLoader
from datetime import datetime


def train(out_dir, data_loader: DataLoader):
    task1(out_dir, data_loader)
    return 
    for remove_count in [2000, 20000, 200000]:
        task2(out_dir, data_loader, remove_cnt=remove_count)
    for n in [30, 50, 100, 200, 400]:
        task3(out_dir, data_loader, params={"N": n})


def test(out_dir, data_loader: DataLoader, model_dir="./model/rand_seed_14"):
    task1(
        out_dir,
        data_loader,
        is_train=False,
        loadpath=f"{model_dir}/task1/svm_model.pkl",
    )
    return 
    for c in [2000, 20000, 200000]:
        loadpath = f"{model_dir}/task2/svm_model_remove{c}.pkl"
        task2(
            out_dir,
            data_loader,
            is_train=False,
            loadpath=loadpath,
            remove_cnt=c,
        )
    for n, k in [
        (30, 7),
        (50, 7),
        (100, 7),
        (200, 7),
        (400, 7),  # 400 的 耗时太长了而且效果很烂 建议不要运行
    ]:
        loadpath = f"{model_dir}/task3/N_{n}_K_{k}/svm_model.pkl"
        task3(
            out_dir,
            data_loader,
            is_train=False,
            loadpath=loadpath,
            params={"N": n, "K": k},
        )


import argparse

if __name__ == "__main__":
    run_timestemp = datetime.now().strftime("%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_dir", type=str, default="./model/rand_seed_14")
    parser.add_argument("--out_dir", type=str, default=run_timestemp)
    data_loader = DataLoader("../data/creditcard.csv")
    is_train = parser.parse_args().mode == "train"
    model_dir = parser.parse_args().model_dir
    out_dir = parser.parse_args().out_dir
    if is_train:
        print("Training")
        print(f"Output dir: {out_dir}")
        train(out_dir, data_loader)
    else:
        print("Testing")
        print(f"Output dir: {out_dir}")
        print(f"Model dir: {model_dir}")
        test(out_dir, data_loader, model_dir=model_dir)
