import numpy as np
import os
from PIL import Image

# 定义输入和输出文件夹路径
input_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理'  # 替换为你的文件夹路径
output_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理_npy'  # 替换为输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有 .png 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 构造完整的文件路径
        image_path = os.path.join(input_folder, filename)

        # 读取图像并转换为 NumPy 数组
        image = Image.open(image_path)
        image_array = np.array(image)

        # 构造新的输出文件名（保持原文件名）
        npy_filename = filename.replace('.png', '.npy')
        npy_path = os.path.join(output_folder, npy_filename)

        # 保存为 .npy 文件
        np.save(npy_path, image_array)

print("All images have been converted to .npy files.")
