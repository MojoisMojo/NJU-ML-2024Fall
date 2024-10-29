import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_loader import dataset_names, get_data
from params import TEST_SIZE, RAND_SEED

dname = "moon"
# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=RAND_SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND_SEED
)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, num_classes):
        assert num_layers == len(hidden_sizes)
        assert num_layers >= 1
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

    def forward(self, x):
        return self.network(x)


# 实验参数
hidden_sizes = [8, 16, 4]
num_layers_list = [1, 2, 3]
num_epochs = 100
learning_rate = 0.01

# 记录测试集精度
test_accuracies = []

for num_layers in num_layers_list:
    model = MLP(
        input_size=2,
        hidden_sizes=hidden_sizes[:num_layers],
        num_layers=num_layers,
        num_classes=2,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    # 测试模型
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        test_accuracies.append(accuracy)
        print(f"Num Layers: {num_layers}, Test Accuracy: {accuracy:.4f}")

# 绘制测试集精度
plt.figure()
plt.plot(num_layers_list, test_accuracies, marker="o")
plt.xlabel("Number of Layers")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Number of Layers")
# plt.show()
plt.savefig("output/task4.png")
plt.close()
