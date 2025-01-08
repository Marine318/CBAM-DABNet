
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# Import necessary components from model.py
from model import self_net


def load_model(model_path, num_classes=4, cuda=True):
    model = self_net()
    device = torch.device('cuda' if cuda else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    return model


def perform_inference(model, image, input_shape, cuda=True):
    image = np.array(image)
    orininal_h, orininal_w = image.shape[:2]
    image_data = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data).float()
        if cuda:
            images = images.cuda()
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    return Image.fromarray(np.uint8(pr))


if __name__ == "__main__":
    model_path = r'F:\人工智能算法精英大赛\CBAM+DABNet\last_epoch_weights_ab.pth'   # Update with your model path
    input_folder = 'Dataset C-供国赛参赛学生打榜用/供国赛参赛学生用/Img'  # Update with your input folder path
    output_folder = 'C_处理后的推理图片——转化为灰度图_ab——自己推理'  # Update with your output folder path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = load_model(model_path)
    image_ids = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for image_id in tqdm(image_ids):
        image_path = os.path.join(input_folder, image_id)
        image = Image.open(image_path)
        result = perform_inference(model, image, input_shape=[200, 200])
        result.save(os.path.join(output_folder, image_id.replace('.jpg', '.png')))

    print("Inference complete!")