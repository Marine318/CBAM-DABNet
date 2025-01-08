import os
import shutil

def copy_process(source_folder,destination_folder,k):
	# 创建目标文件夹（如果不存在的话）
	os.makedirs(destination_folder, exist_ok=True)
	
	# 获取源文件夹中的所有图片文件
	image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
	
	# 复制并重命名图片
	for index, image_file in enumerate(image_files):
		# 生成新的文件名
		new_name = f"{index+k:05d}.png"  # 这里假设图片格式为jpg，可以根据实际情况修改
		# 复制文件
		shutil.copy(os.path.join(source_folder, image_file), os.path.join(destination_folder, new_name))
	
	

if __name__ == "__main__":
	
	# 源文件夹路径
	source_folder_test = r'D:\pycharm\deeplabv3-plus-pytorch-main\annotations\test'
	# 目标文件夹路径
	destination_folder_test = r'D:\pycharm\deeplabv3-plus-pytorch-main\新钢材数据\annotations\test'
	
	copy_process(source_folder_test, destination_folder_test, 0)
	
	# 源文件夹路径
	source_folder_val = r'D:\pycharm\deeplabv3-plus-pytorch-main\annotations\training'
	# 目标文件夹路径
	destination_folder_val = r'D:\pycharm\deeplabv3-plus-pytorch-main\新钢材数据\annotations\training'
	
	copy_process(source_folder_val, destination_folder_val,10000 )
	
	print("图片复制并重命名完成！")