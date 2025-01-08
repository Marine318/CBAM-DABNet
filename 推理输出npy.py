import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from model import self_net


def preprocess_input(image):
    image /= 255.0
    return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def get_miou_png(image):
    # 转换图像到RGB并获取原始尺寸
    image = cvtColor(image)
    orininal_h, orininal_w = image.size[1], image.size[0]

    # 调整图像大小并增加灰条
    image_data, nw, nh = resize_image(image, (200, 200))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        net = self_net()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = r'F:\人工智能算法精英大赛\CBAM+DABNet\last_epoch_weights_ab.pth'
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.eval()

        # 进行预测
        pr = net(images.to(device))[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # 去掉灰条部分
        pr = pr[int((200 - nh) // 2): int((200 - nh) // 2 + nh), int((200 - nw) // 2): int((200 - nw) // 2 + nw)]
        # 恢复到原始大小
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    return pr  # 返回numpy数组


if __name__ == "__main__":
    miou_mode = 1
    num_classes = 4
    name_classes = ["_background_", "Inclusions", "Patches", "Scratches"]

    # 输入与输出文件夹
    input_folder = 'Dataset C-供国赛参赛学生打榜用/供国赛参赛学生用/Img'  # 修改为实际的输入路径
    output_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理_npy'  # 修改为保存.npy的路径

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print("生成预测结果...")
        for image_id in tqdm(os.listdir(input_folder)):
            if image_id.endswith('.jpg'):
                image_path = os.path.join(input_folder, image_id)
                image = Image.open(image_path)
                prediction = get_miou_png(image)

                # 保存为.npy文件
                output_path = os.path.join(output_folder, image_id.replace('.jpg', '.npy'))
                np.save(output_path, prediction)
        print("预测结果已生成并保存为.npy文件。")

    print("处理完成！")
