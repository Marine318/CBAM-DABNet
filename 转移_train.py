import os

# 源文件夹路径
source_folder = r'D:\pycharm\deeplabv3-plus-pytorch-main\新钢材数据\annotations\training'
# 目标文本文件路径
target_file = r'D:\pycharm\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\ImageSets\Segmentation\train.txt'

# 确保目标目录存在
os.makedirs(os.path.dirname(target_file), exist_ok=True)

# 获取源文件夹中的所有图片文件名
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# 写入到目标文本文件
with open(target_file, 'w', encoding='utf-8') as f:
    for image_file in image_files:
        # 写入文件名，不带路径
        f.write(os.path.splitext(image_file)[0] + '\n')  # 去掉扩展名

print("图片名称已复制到 train.txt 文件中！")