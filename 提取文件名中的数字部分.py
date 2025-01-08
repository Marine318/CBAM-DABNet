import os

# 定义包含 .npy 文件的文件夹路径
folder_path = 'C_处理后的推理图片——转化为灰度图_ab——自己推理'  # 替换为你的文件夹路径

# 遍历文件夹中的所有 .npy 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.npy' or '.png'):
        # 提取文件名中的数字部分
        number_part = ''.join(filter(str.isdigit, filename))

        # 构造新的文件名，前面加上 "c_"
        new_filename = f'prediction_{number_part}.npy'

        # 构造完整的旧文件路径和新文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)

print("All files have been renamed.")
