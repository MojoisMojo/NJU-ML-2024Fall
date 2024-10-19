from task1 import task1
from task2 import task2
from dataloader import DataLoader
from datetime import datetime
if __name__ == "__main__":
    data_loader = DataLoader("../data/creditcard.csv")
    run_timestemp = datetime.now().strftime("%m%d_%H%M%S")
    task1(run_timestemp, data_loader)
    task2(run_timestemp, data_loader)