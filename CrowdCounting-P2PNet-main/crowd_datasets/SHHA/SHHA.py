import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
from PIL import Image
import cv2
import glob
import scipy.io as io


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.list"
        self.eval_list = "test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')  # 记录各个list文件名的列表
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}  #
        self.img_list = []  #
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # print(f'1. After load_data | img type: {type(img)}, img size: {img.size}'
        #       f', point type: {type(point)}, point shape: {point.shape}')
        # apply augmentation
        if self.transform is not None:
            img = self.transform(img)
            # print(f'2. After transform | img type: {type(img)}, img shape: {img.shape}')
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            scale = random.uniform(*scale_range)
            min_size = min(img.shape[1:])
            # scale the image and points
            if scale * min_size > 128:
                # img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), mode='bilinear', scale_factor=scale).squeeze(0)
                point *= scale
            # print(f'3.(train) After interpolate | img type: {type(img)}, img shape: {img.shape},'
            #       f' point type: {type(point)}, \npoint[0] type: {type(point[0])}, len of point: {len(point)} ')
        # random crop augmentation
        if self.train and self.patch:
            img, point = random_crop(img, point)
            # print(f'4.(patch) After random_crop | img type: {type(img)}, img shape: {img.shape}, img[0] type: {type(img[0])}'
            #       f' \npoint type: {type(point)}, point[0] type: {type(point[0])}, len of point: {len(point)} ')
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
            # print(f'5.(patch) After point[i] = torch.Tensor(point[i]) |'
            #       f' point type: {type(point)}, point[0] type: {type(point[0])}, len of point: {len(point)}')
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:  # flip一定要有patch为True，否则出错！
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]  # 此处128应为小图像的宽，因此此处有逻辑错误，
                # 因为flip操作程序上不一定要依赖patch为True，经验证，遇到patch为False而flip为True的情况时，若进行该操作，则程序会停止运行。
            # print(f'6.(flip) After img = torch.Tensor(img[:, :, :, ::-1] and point[i][:, 0] = 128 - point[i][:, 0] | '
            #       f'img type: {type(img)}, img shape: {img.shape}, \nimg[0] type: {type(img[0])} \npoint type: {type(point)}, '
            #       f'point[0] type: {type(point[0])}, len of point: {len(point)}')

        if not self.train:
            point = [point]
            # print(f'7.(load_data->not train) After point=[point] | img type: {type(img)}, point type: {type(point)}, '
            #       f'point[0] type: {type(point[0])}, len of point: {len(point)}')

        img = torch.Tensor(img)
        # print(f'8. After img = torch.Tensor(img) | img type: {type(img)}, img shape: {img.shape}'
        #       f', point type: {type(point)}, \npoint[0] type: {type(point[0])}, '
        #       f'len of point: {len(point)}')
        '''
        对于not patch?：
        target[i]['point']就是某个标注点的坐标，其中target[i]['image_id']的值都相同(同一张图)，target[i]['labels']的值为2(？)
        对于patch：
        target[i]['point']是某个crop的标注点集，target[i]['image_id']是对应crop的序号，target[i]['labels']为该crop的标注点数量
        '''
        # pack up related infos
        target = [{} for i in range(len(point))]  # 有几个标注点，就有几个字典
        for i, _ in enumerate(point):  # 把每个标注点的坐标和对应图像序号和标签(?)包装在一起。
            target[i]['point'] = torch.Tensor(point[i])
            # image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = int(img_path.split('\\')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
        # print(f'9.(target) | img type: {type(img)}, img shape: {img.shape}, img[0] type: {type(img[0])}'
        #       f' \npoint type: {type(target[0]["point"])}, point shape: {target[0]["point"].shape}')
        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)  # ?
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # PIL.Image.Image
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])  # 该图像的坐标列表

    return img, np.array(points)  # 返回PIL.Image.Image类型的图像和该图像ndarray类型的坐标数组


# random crop augumentation
def random_crop(img, den, num_patch=4):  # 一个图像默认切分为4个小图像
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])  # 4个[C, H, W]格式的图像
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]  # 这一步自动从tensor变为ndarray
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den


if __name__ == "__main__":
    # 测试代码
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    train_set = SHHA(r'..\..\data_root', train=True, transform=transform, patch=True, flip=True)
    val_set = SHHA(r'..\..\data_root', train=False, transform=transform)
    print("train set:")
    for img, target in train_set:
        print(img.shape, len(target[0]['point']))
    print("validate set:")
    for img, target in val_set:
        print(img.shape, len(target[0]['point']))