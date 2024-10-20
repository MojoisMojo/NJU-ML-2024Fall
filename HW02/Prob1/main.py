from task1 import task1
from task2 import task2
from task3 import task3
from dataloader import DataLoader
from datetime import datetime


def train(run_time, data_loader: DataLoader):
    task1(run_time, data_loader)
    for remove_count in [2000, 20000, 200000]:
        task2(run_time, data_loader, remove_cnt=remove_count)
    for n in [30, 50, 100, 200, 400]:
        task3(run_time, data_loader, params={"N": n})


def test(run_time, data_loader: DataLoader):
    task1(
        run_time,
        data_loader,
        is_train=False,
        loadpath=f"./out/rand_seed_14/task1/svm_model.pkl",
    )
    for c in [2000, 20000, 200000]:
        loadpath = f"./out/rand_seed_14/task2/svm_model_remove{c}.pkl"
        task2(
            run_time,
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
        # (400, 7), # 400 的 耗时太长了而且效果很烂
    ]:
        loadpath = f"./out/rand_seed_14/task3/N_{n}_K_{k}/svm_model.pkl"
        print("test loadpath:", loadpath)
        task3(
            run_time,
            data_loader,
            is_train=False,
            loadpath=loadpath,
        )


if __name__ == "__main__":
    data_loader = DataLoader("../data/creditcard.csv")
    run_timestemp = datetime.now().strftime("%m%d_%H%M%S")
    # train(run_timestemp, data_loader)
    test(run_timestemp, data_loader)
