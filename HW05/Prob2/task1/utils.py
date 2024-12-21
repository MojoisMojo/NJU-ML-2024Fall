import matplotlib.pyplot as plt
import numpy as np


def plot_feature_number_vs_accuracy(feature_numbers, filter_scores, wrapper_scores):
    """绘制特征数量与准确率的关系图"""
    plt.plot(feature_numbers, filter_scores, label="filter")
    plt.plot(feature_numbers, wrapper_scores, label="wrapper")
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("feature_number_vs_accuracy.png")
    plt.close()


def analyse_and_compare(
    filter_scores: np.ndarray,
    wrapper_scores: np.ndarray,
    filter_indices: np.ndarray,
    wrapper_indices: np.ndarray,
):
    """两种方法的最佳特征数量和对应准确率, 计算并解释每个特征被选择的频率（特征重要性）"""
    # 计算最佳特征数量、哪些特征和对应准确率
    n_f = len(filter_scores)
    assert (
        n_f == len(wrapper_scores)
        and n_f == len(filter_indices)
        and n_f == len(wrapper_indices)
    )
    filter_best_idx = np.argmax(filter_scores)
    wrapper_best_idx = np.argmax(wrapper_scores)
    filter_best_score = filter_scores[filter_best_idx]
    wrapper_best_score = wrapper_scores[wrapper_best_idx]
    filter_best_features = filter_indices[filter_best_idx]
    wrapper_best_features = wrapper_indices[wrapper_best_idx]
    print(
        f"Filter method:\n\tBest number of features: {len(filter_best_features)}\n\ttheir idxs are: {filter_best_features}\n\tBest accuracy: {filter_best_score}"
    )
    print(
        f"Wrapper method:\n\tBest number of features: {len(wrapper_best_features)}\n\ttheir idxs are: {wrapper_best_features}\n\tBest accuracy: {wrapper_best_score}"
    )
    filter_feature_freq = np.zeros(n_f)
    wrapper_feature_freq = np.zeros(n_f)
    cnt = 0
    for i in range(n_f):
        assert len(filter_indices[i]) == len(wrapper_indices[i])
        for idx in filter_indices[i]:
            filter_feature_freq[idx] += 1
        for idx in wrapper_indices[i]:
            wrapper_feature_freq[idx] += 1
    filter_feature_freq /= filter_feature_freq.sum()
    wrapper_feature_freq /= wrapper_feature_freq.sum()
    print(f"Features idx: {range(n_f)}")
    print(f"Filter method feature frequency: {filter_feature_freq}")
    print(f"Wrapper method feature frequency: {wrapper_feature_freq}")
