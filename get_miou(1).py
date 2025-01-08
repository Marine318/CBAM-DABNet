import os

from PIL import Image
from tqdm import tqdm
import numpy as np
from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 4
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    name_classes = ["_background_", "Inclusions", "Patches","Scratches"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path = 'A测试集+C/jpg'
    miou_out_path = "生成npy文件"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # 获取图像文件夹路径
    image_folder = os.path.join(VOCdevkit_path)
    # 获取所有图像文件的文件名（假设图像格式为 .jpg）
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 检查并创建预测结果目录
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(image_folder, image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)

            # 将预测结果转换为NumPy数组
            image_array = np.array(image)
            # 保存为 .npy 文件
            np.save(os.path.join(pred_dir, image_id + ".npy"), image_array)
        print("Get predict result done.")

    # if miou_mode == 0 or miou_mode == 2:
    #     print("Get miou.")
    #     hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
    #     print("Get miou done.")
    #     show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)