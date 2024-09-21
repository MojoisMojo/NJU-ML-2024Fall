import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# 数据
FPR = [0, 0 / 5, 0 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 5, 2 / 5, 3 / 5, 4 / 5, 5 / 5]
TRR = [0, 1 / 5, 2 / 5, 2 / 5, 3 / 5, 4 / 5, 4 / 5, 5 / 5, 5 / 5, 5 / 5, 5 / 5]
plt.figure()

# 绘制P-R曲线
plt.plot(FPR, TRR, "-o", linewidth=2, markersize=6, zorder=10)
plt.grid(True)

# 添加轴标签
plt.xlabel("假正例率", fontproperties=font)
plt.ylabel("真正例率", fontproperties=font)

# 添加标题
plt.title("ROC曲线", fontproperties=font)

# 设置轴范围
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

ax = plt.gca()

# 获取你想要挪动的坐标轴，这里只有顶部、底部、左、右四个方向参数
ax.xaxis.set_ticks_position('bottom')  #  要挪动底部的X轴，所以先目光锁定底部！
ax.spines['bottom'].set_position(('data',0))
ax.spines['top'].set_position(('data',1))
ax.yaxis.set_ticks_position('left')  #  要挪动底部的Y轴，所以先目光锁定左部！
ax.spines['left'].set_position(('data',0))
ax.spines['right'].set_position(('data',1))
# 保存图形
plt.savefig("ROC.png")

# # 显示图形
# plt.show()
