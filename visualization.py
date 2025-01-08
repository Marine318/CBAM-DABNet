import os
from PIL import Image
import numpy as np

# 输入和输出文件夹路径
# input_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理'
input_folder = '真正国赛测试GT'
# output_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理-可视化_ab'
output_folder = '真正国赛测试GT_可视化'
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 定义颜色映射
color_map = {
    0: (0, 0, 0),  # 黑色
    1: (255, 0, 0),  # 红色
    2: (255, 255, 0),  # 黄色
    3: (0, 0, 255)  # 蓝色
}

# 遍历输入文件夹中的每张图片
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 打开灰度图像
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('L')

        # 将灰度图像转换为numpy数组
        img_array = np.array(img)

        # 创建彩色图像的空数组
        color_img_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

        # 根据灰度值赋予相应颜色
        for gray_value, color in color_map.items():
            color_img_array[img_array == gray_value] = color

        # 转换为PIL图像并保存
        color_img = Image.fromarray(color_img_array)
        color_img.save(os.path.join(output_folder, filename))

print("图片转换完成！")
