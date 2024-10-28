from mindspore.dataset import transforms, vision
from .SHHA import SHHA


class DeNormalize(object):  # DeNormalize used to get original images
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root):
    transform = transforms.Compose([
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225], is_hwc=False),
    ])
    # 当处理的是HWC格式时，用Normalize对最后一个维度进行数据处理，然后用一次ToTensor()转换为CHW格式，但顺序可能不能随便颠倒
    # create the training dataset
    train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA(data_root, train=False, transform=transform)

    return train_set, val_set
