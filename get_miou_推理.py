import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model import self_net

def preprocess_input(image):
    image /= 255.0
    return image

def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def get_miou_png(image):
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (200, 200))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        # if self.cuda:
        #     images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        net = self_net()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = r'last_epoch_weights_C_312.pth'
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.eval()
        print('{} model, and classes loaded.'.format(model_path))
        pr = net(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((200 - nh) // 2): int((200 - nh) // 2 + nh), \
             int((200 - nw) // 2): int((200 - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

    image = Image.fromarray(np.uint8(pr))
    return image

if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 1  # 设置为1，仅生成预测结果
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 4
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["_background_", "Inclusions", "Patches", "Scratches"]

    # 输入图片文件夹路径
    input_folder = 'Dataset C-供国赛参赛学生打榜用/供国赛参赛学生用/Img'  # 请修改为您自己的文件夹路径
    # 输出预测结果文件夹路径
    output_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理'  # 请修改为您想保存GT图像的文件夹路径

    # 获取文件夹中的所有jpg文件
    image_ids = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    if miou_mode == 0 or miou_mode == 1:
        # 只生成预测结果，不计算mIoU
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # print("Load model.")
        # deeplab = DeeplabV3()
        # print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(input_folder, image_id)
            image = Image.open(image_path)
            image = get_miou_png(image)

            # 保存为 PNG 图像
            output_path = os.path.join(output_folder, image_id.replace('.jpg', '.png'))
            image.save(output_path)  # 保存图像为 PNG 格式

        print("预测结果已生成并保存为 PNG 文件。")

    # 如果miou_mode是1（只生成GT结果），则无需进行mIoU计算，因此无需继续执行miou部分
    if miou_mode == 0 or miou_mode == 2:
        # 跳过mIoU计算部分
        pass

    print("GT生成完成！")
