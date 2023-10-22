# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:09:05 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('CHEN2-7.BMP', 0)

# 中值滤波以减少噪声
image = cv2.medianBlur(image, 5)

# 使用Otsu's阈值方法进行二值化
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用形态学操作进行细胞分离
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 找到轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 使用matplotlib显示图像
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(binary, cmap='gray')

# 面积阈值
area_threshold = 390  # 你可以根据实际情况调整这个值
flag=0
for idx, contour in enumerate(contours):
    # 面积
    area = cv2.contourArea(contour)
    
    # 如果面积小于阈值则跳过
    if area < area_threshold:
        flag+=1
        continue
    
    # 周长
    perimeter = cv2.arcLength(contour, True)
    
    # 圆形度
    circularity = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter**2)
    
    # 矩形度
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    rectangularity = 0 if rect_area == 0 else area / rect_area
    
    # 质心
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    
    # 方向
    mu20 = M["mu20"] / M["m00"] if M["m00"] != 0 else 0
    mu02 = M["mu02"] / M["m00"] if M["m00"] != 0 else 0
    mu11 = M["mu11"] / M["m00"] if M["m00"] != 0 else 0
    angle = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02)) if mu20 - mu02 != 0 else 0

    # 在图像上标注信息
    info = f"Cell {idx + 1-flag}:\nArea: {area:.2f}\nPerimeter: {perimeter:.2f}\nCircularity: {circularity:.2f}\nRectangularity: {rectangularity:.2f}\nCentroid: ({cx}, {cy})\nAngle: {np.degrees(angle):.2f}°"
    ax.text(cx, cy, info, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.title("Processed Image")
plt.savefig("cell_information.png",dpi=500,bbox="tight")
plt.show()
