# 导入必要的库
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_true = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]  # 真实标签
y_scores = [0.92, 0.75, 0.62, 0.55, 0.49, 0.4, 0.31, 0.28, 0.2, 0.1]  # 预测概率

font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

assert len(y_true) == len(y_scores), "标签和概率的长度不一致"

# 计算精确率和召回率
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# 计算平均精确率
average_precision = average_precision_score(y_true, y_scores)

# 绘制PR图
plt.figure()
plt.step(recall, precision, where="post")

# 添加标题和标签
plt.xlabel("查全率", fontproperties=font)
plt.ylabel("查准率", fontproperties=font)
plt.title(f"Precision-Recall 曲线: AP={average_precision:0.2f}", fontproperties=font)

plt.savefig("PR.png")

# 显示图形
plt.show()
