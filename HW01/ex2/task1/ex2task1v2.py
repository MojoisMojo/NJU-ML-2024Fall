import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# 数据
recall = [1 / 5, 2 / 5, 2 / 5, 3 / 5, 4 / 5, 4 / 5, 5 / 5, 5 / 5, 5 / 5, 5 / 5]
precision = [1 / 1, 2 / 2, 2 / 3, 3 / 4, 4 / 5, 4 / 6, 5 / 7, 5 / 8, 5 / 9, 5 / 10]

plt.figure()

# 绘制P-R曲线
plt.grid(True)
plt.plot(recall, precision, "-o", linewidth=2, markersize=6)

# 添加轴标签
plt.xlabel("查全率", fontproperties=font)
plt.ylabel("查准率", fontproperties=font)

# 添加标题
plt.title("Precision-Recall 曲线", fontproperties=font)

# 设置轴范围
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])

# 设置边框的zorder为1
ax = plt.gca()
# 取消边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 保存图形
plt.savefig("PR.png")

# # 显示图形
# plt.show()
