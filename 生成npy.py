import os
import numpy as np
from PIL import Image

# 输入和输出文件夹路径
input_folder = "C_处理后的推理图片——转化为灰度图_ab——自己推理"  # 替换为您的 PNG 文件夹路径
output_folder = "自己的npy_files"  # 保存 .npy 文件的文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的每个 PNG 图像文件
for filename in os.listdir(input_folder):
	if filename.endswith(".png"):
		image_path = os.path.join(input_folder, filename)
		
		# 读取灰度图像并转换为 numpy 数组
		image = Image.open(image_path)
		image_array = np.array(image)
		
		# 提取图片的原始文件名数字部分，并重新格式化文件名
		base_filename = os.path.splitext(filename)[0]  # 去掉扩展名
		base_number = base_filename.split("_")[1]  # 提取数字部分
		npy_filename = f"prediction_{base_number}.npy"  # 新文件名前缀
		
		# 构建保存路径
		output_path = os.path.join(output_folder, npy_filename)
		
		# 保存为 .npy 文件
		np.save(output_path, image_array)
		
		print(f"Processed and saved: {output_path}")
