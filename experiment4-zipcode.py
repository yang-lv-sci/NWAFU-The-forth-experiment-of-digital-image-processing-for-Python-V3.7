# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:23:27 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = "zipcode_number.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 使用形态学操作
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# 轮廓检测
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选和排序轮廓
valid_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if 100 < h < 130 and 0.80<aspect_ratio<1:  # 修改筛选条件，考虑高度和宽高比
        valid_contours.append((x, y, w, h))

valid_contours = sorted(valid_contours, key=lambda x: x[0])

# 在原始图像上绘制轮廓
for contour in valid_contours:
    x, y, w, h = contour
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 使用matplotlib显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Postal Code Detection")
plt.axis("off")
plt.savefig("zipcode_number_python.png", dpi=500,bbox="tight")
plt.show()
