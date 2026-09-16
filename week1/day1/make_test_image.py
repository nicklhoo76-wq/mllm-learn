"""生成一张内容完全已知的测试图，用来验证模型是不是真"看"到了东西。

不用网图有两个原因：
1. huggingface.co 在这台机器上不通，少一个网络依赖少一个坑；
2. 内容自己定的，模型答得对不对一目了然，不用你去猜标准答案。
"""
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shapes.png")

W, H = 448, 448
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

# 左上：红色圆
d.ellipse([40, 40, 200, 200], fill="red")
# 右上：蓝色正方形
d.rectangle([250, 40, 410, 200], fill="blue")
# 左下：绿色三角形
d.polygon([(120, 250), (40, 410), (200, 410)], fill="green")
# 右下：黄色圆
d.ellipse([250, 250, 410, 410], fill="yellow")

img.save(OUT)
print(f"已生成 {OUT}")
print("真实内容：左上红圆、右上蓝方、左下绿三角、右下黄圆")
