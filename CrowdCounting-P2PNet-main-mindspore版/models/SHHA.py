import os
import random
import mindspore
import mindspore.dataset as ds
import numpy as np
from mindspore.dataset import GeneratorDataset, transforms, vision
from PIL import Image
import cv2
import glob
import scipy.io as io


class SHHA:
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root  # 数据集文件路径
        self.train_lists = "train.list"  # 训练集和验证集的图片路径列表
        self.eval_list = "test.list"
        # there may exist multiple list files 如果存在多个list的处理，这里不用管
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}  # 图像数据地址和标注数据地址映射关系字典
        self.img_list = []  # 图像数据地址列表
        # loads the image/ground truth pairs 装载图片和真实点# loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):  # 遍历list文件名列表
            train_list = train_list.strip()  # 去除字符串两边的空格
            with open(os.path.join(self.root_path, train_list)) as fin:  # 打开list文件
                for line in fin:  # 遍历每一行
                    if len(line) < 2:  # 当行的元素小于2，即不同时有图像地址和标注数据地址时，跳过该行
                        continue
                    line = line.strip().split()  # line[0]为图片路径，line[1]为真实点的坐标文本路径
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())  # 把图像地址和对应的标注数据地址存储在img_map属性中
        self.img_list = sorted(list(self.img_map.keys()))  # 把图像地址提取到img_list列表中，并排序
        # number of samples
        self.nSamples = len(self.img_list)  # 样本的个数，即数据集长度
        self.transform = transform
        self.train = train  # 标记为训练集还是测试集
        self.patch = patch  # 是否将图像随机切成可重叠的4个128×128的小图像
        self.flip = flip  # 是否允许图像翻转

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'  # assert语句，确保index<=len(self)，否则直接报错，而不是等到运行时再报错
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # 返回用Image和cv2读取的图片RGB数组创建的PIL.Image.Image对象和该图片的头部标注点坐标ndarray数组
        img, point = load_data((img_path, gt_path), self.train)
        # apply augmentation 图片归一化
        img = np.asarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.train:  # 若是训练
            # data augmentation -> random scale 进行数据增强，随机选取一个规模
            scale_range = [0.7, 1.3]  # 随机因子的范围
            min_size = min(img[0].shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points 对图像和点进行等比例缩放
            if scale * min_size > 128:
                # 进行interpolate操作
                target_size = (int(img.shape[1] * scale), int(img.shape[2] * scale))
                img = mindspore.ops.interpolate(input=mindspore.Tensor(img).unsqueeze(0), size=target_size, mode="bilinear").squeeze(0)
                point *= scale
        # random crop augumentaiton 对图片进行裁剪
        if self.train and self.patch:
            img, point = random_crop(img.asnumpy(), point)
            for i, _ in enumerate(point):
                point[i] = mindspore.Tensor(point[i])
        # random flipping 有一半的概率进行水平翻转
        if random.random() > 0.5 and self.train and self.flip: # 图像y轴镜像翻转
            # random flip
            img = mindspore.Tensor(img[:, :, :, ::-1])
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0] # 128是小图像的宽
        img = mindspore.Tensor(img)
        if not self.train:
            point = [point]
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = mindspore.Tensor(point[i])
            image_id = int(img_path.split('\\')[-1].split('.')[0].split('_')[-1])  # 去掉目录地址和文件后缀名，提取图像的序号
            image_id = mindspore.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = mindspore.ops.ones([point[i].shape[0]]).long()
        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    '''
        cv2.imread读取图片，将图片转换为一个[height, width, channel]三维数组，HWC数组
        前两维表示图片的像素坐标，最后一位表示该像素的channel数据，
        channel的顺序是BGR不是RGB。
    '''
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将图片从BGR转换为RGB，然后再使用PIL的Image转为图片
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])
    return img, np.array(points)


def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])  # num_patch个图像切片数据，大小为half_h和half_w
    result_den = [] # 图像crop部分内的ground_truth数据
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.shape[1] - half_h)
        start_w = random.randint(0, img.shape[2] - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]  # 截取图像的某一矩形部分，CHW格式
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the coordinates
        record_den = den[idx] # 使用布尔数组索引选出crop中的点，这是record_den，为ndarray，可以使用[:, 0]索引
        # 将数值改为crop中的相对坐标
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        if record_den.shape[0] == 0:
            record_den = np.zeros((1, 2))

        result_den.append(record_den)

    return result_img, result_den


if __name__ == '__main__':
    transform = transforms.Compose(
        [vision.ToTensor(),
         vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False), ])
    train_set = SHHA(r'../data_root', train=True, transform=transform, patch=True, flip=True)
    for imgs, targets in train_set:
        print(imgs.shape, len(targets))
