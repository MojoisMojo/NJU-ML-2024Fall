import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_from_disk  # SST-2 数 据 集
from scipy.sparse import vstack
import matplotlib.pyplot as plt

dataset_path = "../datasets/sst2"
# 设 置 随 机 种 子， 确 保 结 果 可 复 现
random.seed(42)
np.random.seed(42)
# 加 载 SST-2 数 据 集
datasets = load_from_disk(dataset_path)
train_data = datasets["train"]
valid_data = datasets["validation"]
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Example of a training instance: {train_data[0]}")
# 提 取 文 本 和 标 签
train_texts = [example["sentence"] for example in train_data]
train_labels = [example["label"] for example in train_data]
valid_texts = [example["sentence"] for example in valid_data]
valid_labels = [example["label"] for example in valid_data]
# 划 分 标 注 数 据 和 未 标 注 数 据
labeled_size = int(0.1 * len(train_texts))  # 仅 保 留10%的 标 注 数 据
indices = np.arange(len(train_texts))
np.random.shuffle(indices)
labeled_indices = indices[:labeled_size]
unlabeled_indices = indices[labeled_size:]
labeled_texts = [train_texts[i] for i in labeled_indices]
labeled_labels = [train_labels[i] for i in labeled_indices]
unlabeled_texts = [train_texts[i] for i in unlabeled_indices]
unlabeled_ground_truth = [train_labels[i] for i in unlabeled_indices]
# 向 量 化 文 本 数 据
vectorizer = CountVectorizer()
X_labeled = vectorizer.fit_transform(labeled_texts)
X_unlabeled = vectorizer.transform(unlabeled_texts)
X_valid = vectorizer.transform(valid_texts)


def task(epochs, confidence_threshold, if_static=True, if_plus=True):
    # 半 监 督EM算 法 - 使 用 朴 素 贝 叶 斯 模 型
    model = MultinomialNB()
    model.fit(X_labeled, labeled_labels)
    unlabeled_scores = np.zeros(epochs + 1)
    valid_scores = np.zeros(epochs + 1)
    unlabeled_scores[0] = accuracy_score(
        unlabeled_ground_truth, model.predict(X_unlabeled)
    )
    valid_scores[0] = accuracy_score(valid_labels, model.predict(X_valid))

    print(f"unlabelled accuracy before em: {unlabeled_scores[0]}")
    print(f"Valid accuracy before em: {valid_scores[0]}")

    print(
        f"Training model with EM algorithm, epochs = {epochs}, confidence_threshold = {confidence_threshold}:\n"
    )

    for epoch in range(1, 1 + epochs):
        # E步，计算每个未标注样本的标签概率
        probs = model.predict_proba(X_unlabeled)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        print(f"Epoch {epoch}:")
        print(f"Confidence threshold: {confidence_threshold}")

        # 筛选出高置信度的样本
        high_confidence_indices = max_probs >= confidence_threshold
        X_high_confidence = X_unlabeled[high_confidence_indices]
        y_high_confidence = preds[high_confidence_indices]

        X_combined = vstack((X_labeled, X_high_confidence))
        assert X_combined.shape[0] == len(labeled_labels) + len(y_high_confidence)
        y_combined = np.hstack((labeled_labels, y_high_confidence))

        # M步，使用高置信度样本更新模型
        model.fit(X_combined, y_combined)
        if if_static == False:
            confidence_threshold = confidence_threshold + (0.02 if if_plus else -0.02)
            confidence_threshold = max(0.45, min(0.9, confidence_threshold))
        unlabeled_scores[epoch] = accuracy_score(
            unlabeled_ground_truth, model.predict(X_unlabeled)
        )
        valid_scores[epoch] = accuracy_score(valid_labels, model.predict(X_valid))
        print(f"unlabeled accuracy: {unlabeled_scores[epoch]}")
        print(f"Valid accuracy: {valid_scores[epoch]}")
        print(f"Epoch {epoch} completed\n")

    # 预测验证集
    y_valid_preds = model.predict(X_valid)
    accuracy = accuracy_score(valid_labels, y_valid_preds)
    print(f"Accuracy on validation set: {accuracy}\n")
    
    return accuracy, unlabeled_scores, valid_scores


def task_static_vs_dynamic():
    print()
    confidence_threshold = 0.7
    epochs = 5
    methods = [(True, None), (False, True), (False, False)]
    for if_static, if_plus in methods:
        desc = f"{'Static' if if_static else 'Dynamic'} confidence threshold {'' if if_static else ('+' if if_plus else '-')}"
        print(
            f"\n{desc}"
        )
        acc, unlabeled_accs, val_accs = task(epochs, confidence_threshold, if_static, if_plus)
        plt.title(desc)
        plt.plot(range(epochs+1), unlabeled_accs, label="Unlabeled accuracy")
        plt.plot(range(epochs+1), val_accs, label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"task1_{desc}.png")
        plt.close()


def task_epochs():
    print()
    confidence_threshold = 0.7
    acc,unlabeled_accs,val_accs = task(20, confidence_threshold)
    plt.title("Task 1 semi-supervised EM algorithm")
    plt.plot(range(21), unlabeled_accs, label="Unlabeled accuracy")
    plt.plot(range(21), val_accs, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("task1_epochs.png")
    plt.close()

if __name__ == "__main__":
    task_static_vs_dynamic()
    # task_epochs()
