from PIL import Image, ImageDraw

# 数据
recall = [
    1 / 5,
    2 / 5,
    2 / 5,
    3 / 5,
    4 / 5,
    4 / 5,
    5 / 5,
    5 / 5,
    5 / 5,
    5 / 5,
]
precision = [1 / 1, 2 / 2, 2 / 3, 3 / 4, 4 / 5, 4 / 6, 5 / 7, 5 / 8, 5 / 9, 5 / 10]

# 图像尺寸
width, height = 400, 400
margin = 50

# 创建图像
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# 绘制坐标轴
draw.line((margin, height - margin, margin, margin), fill="black")
draw.line((margin, height - margin, width - margin, height - margin), fill="black")

# 添加轴标签
draw.text((margin - 40, margin - 20), "查准率", fill="black")
draw.text((width - margin + 10, height - margin + 10), "查全率", fill="black")

# 绘制刻度和标签
for i in range(6):
    x = margin + i * (width - 2 * margin) / 5
    y = height - margin - i * (height - 2 * margin) / 5
    draw.line((x, height - margin, x, height - margin + 5), fill="black")
    draw.line((margin - 5, y, margin, y), fill="black")
    draw.text((x - 5, height - margin + 10), f"{i / 5:.1f}", fill="black")
    draw.text((margin - 30, y - 5), f"{i / 5:.1f}", fill="black")

# 绘制P-R曲线
points = [
    (margin + r * (width - 2 * margin), height - margin - p * (height - 2 * margin))
    for r, p in zip(recall, precision)
]
draw.line(points, fill="blue", width=2)

# 保存图像
image.save("precision_recall_curve.png")
