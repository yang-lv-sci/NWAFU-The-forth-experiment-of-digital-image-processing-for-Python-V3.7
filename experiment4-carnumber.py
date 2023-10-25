# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:34:38 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = "car_number.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 只取图像的下半部分
height, width = gray.shape
roi = gray[int(height/2):, :]

# 针对蓝色车牌进行颜色检测
lower_blue = np.array([100, 40, 40])
upper_blue = np.array([140, 255, 255])
hsv = cv2.cvtColor(image[int(height/2):, :], cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 轮廓检测
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选和排序轮廓
valid_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if 1.7 < aspect_ratio < 1.8 and 15 < w < 300:  # 考虑车牌的长宽比和宽度
        valid_contours.append((x, y + int(height/2), w, h))

# 在原始图像上绘制轮廓
for contour in valid_contours:
    x, y, w, h = contour
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 使用matplotlib显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("License Plate Detection")
plt.axis("off")
plt.savefig("car_number_python.png", dpi=500,bbox="tight")
plt.show()

