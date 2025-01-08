import numpy as np
import os

# 定义用于计算混淆矩阵的辅助函数
def fast_hist(label, pred, num_classes):
    """
    计算混淆矩阵。
    label: ground truth 扁平化数组。
    pred: prediction 扁平化数组。
    num_classes: 类别总数。
    """
    mask = (label >= 0) & (label < num_classes)
    hist = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def per_class_iu(hist):
    """计算每个类别的 IoU。"""
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    return intersection / np.maximum(union, 1)  # 避免除零错误

def per_class_PA_Recall(hist):
    """计算每个类别的 Recall。"""
    return np.diag(hist) / np.maximum(hist.sum(axis=1), 1)

def per_class_Precision(hist):
    """计算每个类别的 Precision。"""
    return np.diag(hist) / np.maximum(hist.sum(axis=0), 1)

def per_Accuracy(hist):
    """计算总体准确率。"""
    return np.diag(hist).sum() / hist.sum()

# 主函数：计算 mIoU
def compute_mIoU(gt_dir, pred_dir, npy_name_list, num_classes, name_classes=None):
    print('Num classes:', num_classes)
    # 初始化混淆矩阵
    hist = np.zeros((num_classes, num_classes))

    # 获取标签和预测文件路径
    gt_files = [os.path.join(gt_dir, x + ".npy") for x in npy_name_list]
    pred_files = [os.path.join(pred_dir, x + ".npy") for x in npy_name_list]

    # 遍历所有文件，计算每个图像的混淆矩阵
    for ind in range(len(gt_files)):
        # 加载 prediction 和 groundtruth
        pred = np.load(pred_files[ind])
        label = np.load(gt_files[ind])

        # 检查预测和标签形状是否匹配
        if pred.shape != label.shape:
            print(
                'Skipping: shape mismatch between gt and pred for files {}, {}'.format(
                    gt_files[ind], pred_files[ind]
                )
            )
            continue

        # 累加混淆矩阵
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # 每计算10张，输出当前的 mIoU、mPA 和 Accuracy
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIoU-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind,
                    len(gt_files),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )

    # 计算每个类别的 IoU、Recall 和 Precision
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    # 输出每个类别的 IoU、Recall 和 Precision
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 5)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 5))+ '; Precision-' + str(round(Precision[ind_class] * 100, 5)))

    # 输出总体的 mIoU、mPA 和 Accuracy
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 5)) +
          '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 5)) +
          '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 5)))

    return np.array(hist, int), IoUs, PA_Recall, Precision

# 使用示例
gt_dir = '标签的npy_files'
pred_dir = '自己的npy_files'
npy_name_list = [f[:-4] for f in os.listdir(gt_dir) if f.endswith('.npy')]  # 假设文件名不含扩展名
num_classes = 4  # 示例类别数，根据实际情况调整
name_classes = ['Class1', 'Class2', 'Class3', 'Class4']  # 类别名称

compute_mIoU(gt_dir, pred_dir, npy_name_list, num_classes, name_classes)

