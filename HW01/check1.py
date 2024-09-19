from sklearn.metrics import precision_score, recall_score

TASK = 4
assert TASK in [1,2,3,4], "TASK must be 1, 2, 3 or 4"
if TASK !=4:
    M = [
        [7,1,4],
        [2,6,4],
        [2,2,8]
    ]
    # 定义真实标签和预测标签
    # 这里假设有三个样本和三个标签
    y_true = []
    y_pred = []
    for i,mi in enumerate(M):
        for j, mij in enumerate(mi):
            y_true+=[i]*mij
            y_pred+=[j]*mij
else:
    # Task 4 多标签分类问题的查准率和查全率
    y_true = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    y_pred = [[1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1]]

# 计算微查准率 (micro-P)
micro_precision = precision_score(y_true, y_pred, average='micro')

# 计算宏查准率 (macro-P)
macro_precision = precision_score(y_true, y_pred, average='macro')

# 计算微查全率 (micro-R)
micro_recall = recall_score(y_true, y_pred, average='micro')

# 计算宏查全率 (macro-R)
macro_recall = recall_score(y_true, y_pred, average='macro')

# 计算微F1 (micro-F1)
micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall)

# 计算宏F1 (macro-F1)
macro_f1 = 2*macro_precision*macro_recall/(macro_precision+macro_recall)

print(f"Micro Precision: {micro_precision}")
print(f"Macro Precision: {macro_precision}")
print(f"Micro Recall: {micro_recall}")
print(f"Macro Recall: {macro_recall}")
print(f"Micro F1: {micro_f1}")
print(f"Macro F1: {macro_f1}")

