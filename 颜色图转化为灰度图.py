from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
input_folder = 'C_处理后的推理图片'
output_folder = 'C_处理后的推理图片——转化为灰度图'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 定义颜色到灰度的映射
color_to_gray = {
    (255, 0, 0): 1,  # 红色 -> 灰度值1
    (0, 255, 0): 2,  # 绿色 -> 灰度值2
    (0, 0, 255): 3  # 蓝色 -> 灰度值3
}


def process_image(image):
    # 将图片转换为 numpy 数组，并初始化一个全零的灰度图
    image_np = np.array(image)
    gray_image = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    # 遍历颜色映射，将特定颜色转换成对应灰度值
    for color, gray_value in color_to_gray.items():
        # 创建掩膜，将符合特定颜色的像素置为对应的灰度值
        mask = np.all(image_np[:, :, :3] == color, axis=-1)
        gray_image[mask] = gray_value

    return Image.fromarray(gray_image)


# 处理文件夹中的所有图片
first_image = None  # 用于保存第一张原图
first_gray_image = None  # 用于保存第一张灰度图

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # 读取图片
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # 保存第一张原图用于绘制颜色直方图
        if first_image is None:
            first_image = image

        # 处理图片并生成灰度图
        gray_image = process_image(image)

        # 保存第一张灰度图用于生成灰度直方图
        if first_gray_image is None:
            first_gray_image = gray_image

        # 保存处理后的图片
        output_path = os.path.join(output_folder, filename)
        gray_image.save(output_path)

print("所有图片处理完成！")

# 在处理前生成第一张源图片的颜色直方图
if first_image is not None:
    first_image_np = np.array(first_image)

    # 绘制RGB颜色分量的直方图
    plt.figure(figsize=(8, 6))
    plt.hist(first_image_np[:, :, 0].ravel(), bins=256, color='red', alpha=0.7, label='Red Channel')
    plt.hist(first_image_np[:, :, 1].ravel(), bins=256, color='green', alpha=0.7, label='Green Channel')
    plt.hist(first_image_np[:, :, 2].ravel(), bins=256, color='blue', alpha=0.7, label='Blue Channel')

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Count")
    plt.title("Color Histogram of the First Image")
    plt.legend(loc='upper right')
    plt.show()

# 生成第一张灰度图的灰度直方图
if first_gray_image is not None:
    gray_image_np = np.array(first_gray_image)

    # 绘制灰度图直方图
    plt.hist(gray_image_np.ravel(), bins=4, range=(0, 4), color='gray', alpha=0.7)
    plt.xlabel("Gray Level")
    plt.ylabel("Pixel Count")
    plt.title("Histogram of the First Grayscale Image")
    plt.xticks([0, 1, 2, 3], ["Background", "Red (1)", "Green (2)", "Blue (3)"])
    plt.show()
