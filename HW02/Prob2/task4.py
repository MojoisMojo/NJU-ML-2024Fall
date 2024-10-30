import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_loader import dataset_names, get_data, get_size
from params import TEST_SIZE, RAND_SEED
import os, logging
from datetime import datetime

task4_output = None
# 实验参数
BACTH_SIZE = 4096
# 这里事实上不应该设置这么大，但是为了加快训练速度，这里设置为 4096 （）
LR = 0.01


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, num_classes):
        assert num_layers == len(hidden_sizes)
        assert num_layers >= 1
        torch.manual_seed(RAND_SEED)
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)
        self.num_layers = num_layers
        print(f"MLP with layers: {hidden_sizes} created")

    def forward(self, x):
        return self.network(x)


def paint_and_save(epochs, hidden_sizes_list, train_accuracies, test_accuracies):
    # 绘制测试集精度
    assert len(hidden_sizes_list) == len(test_accuracies)
    assert len(epochs) == len(test_accuracies[0])
    max_epoch = epochs[-1]
    plt.figure(figsize=(8, 6))
    for i, hidden_sizes in enumerate(hidden_sizes_list):
        la = ",".join(map(str, hidden_sizes))
        plt.plot(epochs, test_accuracies[i], label=f"Layers:{la}-test", marker="o")
        plt.plot(epochs, train_accuracies[i], label=f"Layers:{la}-train", marker="x")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Layers")
    plt.legend()
    plt.grid(True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    plt.savefig(f"{task4_output}/AccVsLayer_maxe{max_epoch}_layer_{timestamp}.png")
    plt.close()


def train(model: MLP, X_train, y_train, X_test, y_test, lr, max_epoch, interval):
    torch.manual_seed(RAND_SEED)
    epochs = []
    test_accs = []
    train_accs = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Training model with {model.num_layers} layers")
    # test_acc = test(model, X_test, y_test, model.num_layers, tag="Test")
    # epochs.append(0)
    # test_accs.append(test_acc)
    # 训练模型
    X_Sample_num = X_train.shape[0]
    for epoch in range(1, 1 + max_epoch):
        epoch_loss = 0
        for i in range(0, X_Sample_num, BACTH_SIZE):
            X_batch = X_train[i : i + BACTH_SIZE]
            y_batch = y_train[i : i + BACTH_SIZE]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % interval == 0:
            logging.info(f"Epoch {epoch}, Loss: {epoch_loss}")
            train_acc = test(model, X_train, y_train, model.num_layers, tag="Train")
            test_acc = test(model, X_test, y_test, model.num_layers, tag="Test")
            print(
                f"Epoch {epoch}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
            )
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)
    return epochs, train_accs, test_accs


def test(model: MLP, X_test, y_test, num_layers, tag):
    torch.manual_seed(RAND_SEED)
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy


def task4(
    output_path,
    max_epoch,
    hidden_sizes_list,
    X_train,
    X_test,
    y_train,
    y_test,
    dname=None,
):
    # 记录测试集精度
    global task4_output
    task4_output = f"{output_path}/task4"
    if dname:
        task4_output = f"{task4_output}/{dname}"
    os.makedirs(task4_output, exist_ok=True)

    train_accuracies = []
    test_accuracies = []
    input_size, output_size = get_size(dname)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    for hidden_sizes in hidden_sizes_list:
        num_layers = len(hidden_sizes)
        model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_layers=num_layers,
            num_classes=output_size,
        )
        epochs, train_accs, test_accs = train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            lr=LR,
            max_epoch=max_epoch,
            interval=max_epoch // 20,
        )

        final_test_acc = test(model, X_test, y_test, num_layers, tag="Test")
        print(f"Final Test Accuracy: {final_test_acc:.4f}")
        if epochs[-1] != max_epoch:
            final_train_acc = test(model, X_train, y_train, num_layers, tag="Train")
            epochs.append(max_epoch)
            train_accs.append(final_train_acc)
            test_accs.append(final_test_acc)
        train_accuracies.append(train_accs)
        test_accuracies.append(test_accs)

    paint_and_save(epochs, hidden_sizes_list, train_accuracies, test_accuracies)


def run_digits():
    print("Processing dataset: digits")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("digits", encoder="label")
    task4(
        "output",
        300,
        [[87, 60], [24], [24, 12]],
        X_train,
        X_test,
        y_train,
        y_test,
        "digits",
    )


def run_income():
    print("Processing dataset: adult_income")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("income", encoder="label")
    task4(
        "output",
        800,
        [[8], [8, 4], [19, 13], [24, 8]],
        X_train,
        X_test,
        y_train,
        y_test,
        "income",
    )


def run_wine():
    print("Processing dataset: wine")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("allwine", encoder="label")
    task4(
        "output",
        800,
        [[11, 11], [20], [20, 14], [14, 20]],
        X_train,
        X_test,
        y_train,
        y_test,
        "allwine",
    )


def run_car():
    print("Processing dataset: car")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("car", encoder="label")
    task4(
        "output",
        800,
        [[12, 8], [12, 16, 8], [12, 8, 4], [32, 16, 8]],
        X_train,
        X_test,
        y_train,
        y_test,
        "car",
    )


def run_bank():
    print("Processing dataset: bank")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("bank", encoder="label")
    task4(
        "output",
        400,
        [[8], [8, 4], [12, 6], [12, 6, 3], [32, 16, 8]],
        X_train,
        X_test,
        y_train,
        y_test,
        "bank",
    )


def run_full_bank():
    print("Processing dataset: full_bank")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_data("full_bank", encoder="label")
    task4(
        "output",
        250,
        [[8], [8, 4], [12, 6, 3], [32, 16, 8]],
        X_train,
        X_test,
        y_train,
        y_test,
        "full_bank",
    )


def main():
    run_digits()
    run_income()
    run_wine()
    run_car()
    run_bank()
    run_full_bank()


if __name__ == "__main__":
    # run_digits()
    # run_income()
    # run_wine()
    # run_car()
    # run_bank()
    # run_full_bank()
    main()
