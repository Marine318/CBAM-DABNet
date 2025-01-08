import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2


# 判断图像是否含有椒盐噪声的函数
def has_salt_and_pepper_noise(image, threshold=0.65):
    """
    使用中值滤波判断图像是否存在椒盐噪声。
    :param image: 输入图像
    :param threshold: 去噪前后图像差异的阈值，用于判断是否存在噪声
    :return: 布尔值，是否含有椒盐噪声
    """
    # 转为灰度图像（如果是彩色图像）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用中值滤波去噪
    filtered_image = cv2.medianBlur(image, 3)

    # 计算原图与去噪后的差异
    ssim_value, _ = ssim(image, filtered_image, full=True)

    # 如果图像差异较大，认为图像有椒盐噪声
    return ssim_value < threshold


# 自定义滤波函数
def custom_filter(image, kernel_size=3, threshold=30):
    """
    使用自定义滤波器处理椒盐噪声。
    :param image: 输入灰度图像
    :param kernel_size: 卷积核大小（必须是奇数）
    :param threshold: 中心像素与均值的差值阈值
    :return: 滤波后的图像
    """
    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                      cv2.BORDER_REFLECT)
    output_image = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 提取卷积核区域
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            region_mean = np.mean(region)
            region_median = np.median(region)

            # 判断中心像素与区域均值的差异
            if abs(image[i, j] - region_mean) > threshold:
                output_image[i, j] = region_median
            else:
                output_image[i, j] = image[i, j]

    return output_image


# 文件夹路径
# img_folder = 'Dataset C-供国赛参赛学生打榜用/供国赛参赛学生用/Img'  # 替换为原始图片文件夹路径
img_folder = 'Dataset B/DataB/Img'  # 替换为原始图片文件夹路径
output_folder = 'Dataset B/DataB/处理后的图像-25'  # 替换为处理后的图片保存文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有文件名
img_files = sorted(os.listdir(img_folder))

# 记录包含椒盐噪声的图片名称
images_with_noise = []
sum = 0

# 遍历文件夹中的所有图片
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)

    # 读取图像并转换为 NumPy 数组
    img = Image.open(img_path)
    img = np.array(img)

    # 判断图像是否有椒盐噪声
    if has_salt_and_pepper_noise(img):
        images_with_noise.append(img_file)
        print(f"图像 {img_file} 包含椒盐噪声")

        # 使用自定义滤波器处理图像
        filtered_img = custom_filter(img, kernel_size=3, threshold=25)

        # 将处理后的图像转换为 PIL 格式并保存到新文件夹
        filtered_img_pil = Image.fromarray(filtered_img)
        filtered_img_path = os.path.join(output_folder, img_file)
        filtered_img_pil.save(filtered_img_path)

        sum += 1

# 将带有椒盐噪声的图像名称写入到文件
with open('椒盐噪声图片.txt', 'w') as f:
    for img_name in images_with_noise:
        f.write(f"{img_name}\n")

print("检测完成，带有椒盐噪声的图像已处理并保存。")
print(f"共检测到 {sum} 张图像包含椒盐噪声，并已处理保存。")
