import os
import random
import mindspore
import numpy as np
from mindspore.dataset import GeneratorDataset, transforms, vision
from PIL import Image
import cv2
# import glob
# import scipy.io as io


class SHHA:  # (Dataset), 继承Dataset，但mindspore不需要继承
    def __init__(self, data_root, train=False, transform=None, patch=False, flip=False):
        self.root_path = data_root  # 按项目要求组织后的数据集的地址
        self.train_lists = "train.list"  # shanghai_tech_part_a_train.list
        self.eval_list = "test.list"  # shanghai_tech_part_a_test.list
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')  # 分为train_list列表
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:  # 不是训练集就是测试集
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}  # 图像数据地址和标注数据地址映射关系字典
        self.img_list = []  # 图像数据地址列表
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):  # 遍历list文件名列表
            train_list = train_list.strip()  # 去除字符串两边的空格
            with open(os.path.join(self.root_path, train_list)) as fin:  # 打开list文件
                for line in fin:  # 遍历每一行
                    if len(line) < 2:  # 当行的元素小于2，即不同时有图像地址和标注数据地址时，跳过该行
                        continue
                    line = line.strip().split()  # 去除两边空格，再根据空格分割为图像地址和标注数据地址
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())  # 把图像地址和对应的标注数据地址存储在img_map属性中
        # sorted()可对所有可迭代对象进行排序操作，并返回新的list而不修改原本的list，但对于字符串元素好像没有效果
        self.img_list = sorted(list(self.img_map.keys()))  # 把图像地址提取到img_list列表中，并排序

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
        # print(f'1. After load_data | img type: {type(img)}, img size: {img.size}'
        #       f', point type: {type(point)}, point shape: {point.shape}')
        img = np.array(img)  # 转为ndarray才能进行后续的数据处理

        if self.transform is not None:
            img = self.transform(img)  # img是HWC数组
            # print(f'2. After transform | img type: {type(img)}, img shape: {img.shape}')

        if not self.train:  # 验证集
            point = [point]
            # print(f'7.(load_data->not train) After point=[point] | img type: {type(img)}, point type: {type(point)}, '
            #       f'point[0] type: {type(point[0])}, len of point: {len(point)}')

        if self.train:
            scale_range = [0.7, 1.3]
            scale = random.uniform(*scale_range)  # 产生0.7到1.3之间的随机数，均匀分布
            min_size = min(img.shape[1:])  # 求图像的宽和高的较小值
            if scale * min_size > 128:
                img = mindspore.ops.interpolate(input=mindspore.Tensor(img).unsqueeze(0), mode='bilinear',
                                                scale_factor=scale, recompute_scale_factor=True).squeeze(0)
                point *= scale
                img = mindspore.Tensor(img)
                # print(f'3.(train) After interpolate | img type: {type(img)}, img shape: {img.shape},'
                #       f' point type: {type(point)}, \npoint[0] type: {type(point[0])}, len of point: {len(point)} ')

        if self.train and self.patch:
            img, point = random_crop(img, point)  # 与源程序相同
            # print(f'4.(patch) After random_crop | img type: {type(img)}, img shape: {img.shape}, '
            #       f'img[0] type: {type(img[0])} '
            #       f'\npoint type: {type(point)}, point[0] type: {type(point[0])}, len of point: {len(point)} ')
            for i, _ in enumerate(point):
                if point[i].shape[0] != 0:  # 避免零维错误
                    point[i] = mindspore.Tensor(point[i])
                else:
                    point[i] = mindspore.Tensor([])
            # print(f'5.(patch) After point[i] = torch.Tensor(point[i]) |'
            #       f' point type: {type(point)}, point[0] type: {type(point[0])}, len of point: {len(point)}')

        if random.random() > 0.5 and self.train and self.flip:  # 图像y轴镜像翻转
            img = mindspore.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                if len(point[i]) != 0:
                    point[i][:, 0] = 128 - point[i][:, 0]  # 128是小图像的宽
            # print(f'6.(flip) After img = torch.Tensor(img[:, :, :, ::-1] and point[i][:, 0] = 128 - point[i][:, 0] | '
            #       f'img type: {type(img)}, img shape: {img.shape}, \nimg[0] type: {type(img[0])} \npoint type: {type(point)}, '
            #       f'point[0] type: {type(point[0])}, len of point: {len(point)}')

        img = mindspore.Tensor(img)
        # print(f'8. After img = torch.Tensor(img) | img type: {type(img)}, img shape: {img.shape}, img[0]: {type(img[0])}'
        #       f', point type: {type(point)}, \npoint[0] type: {type(point[0])}, '
        #       f'len of point: {len(point)}')

        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = mindspore.Tensor(point[i])
            image_id = int(img_path.split('\\')[-1].split('.')[0].split('_')[-1])  # 去掉目录地址和文件后缀名，提取图像的序号
            image_id = mindspore.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            if len(point[i]) != 0:  # 避免零维错误，零维即小图片没有截到头部
                target[i]['labels'] = mindspore.ops.ones([point[i].shape[0]]).long()  # long()指定数据类型为int64
            else:
                target[i]['labels'] = mindspore.Tensor([])
        # img为图像或多个图像，point为图像对应的标注点tensor，image_id为point对应图像的序号，labels为长度为point数量的全1tensor。
        # print(f'9.(target) | img type: {type(img)}, img shape: {img.shape}, img[0] type: {type(img[0])}'
        #       f' \npoint type: {type(target[0]["point"])}, point shape: {target[0]["point"].shape}')
        return img, target  # 返回tensor形式的图像数据(BCHW格式)，以及图像的标注数据、图像序号和labels


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    '''
    cv2.imread读取图片，将图片转换为一个[height, width, channel]三维数组，HWC数组
    前两维表示图片的像素坐标，最后一位表示该像素的channel数据，
    channel的顺序是BGR不是RGB。
    '''
    img = cv2.imread(img_path)
    # 使用cv2可以得到HWC的ndarray数组，但使用PIL只能得到HW数组
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode='RGB')  # 将图片从BGR转换为RGB，然后再使用PIL的Image转为图片
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, np.array(points)


def random_crop(img, den, num_patch=4):
    half_h = 128  # 图像crop的高
    half_w = 128  # 图像crop的宽
    # result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])  # num_patch个图像切片数据，大小为half_h和half_w
    result_img = mindspore.ops.zeros([num_patch, img.shape[0], half_h, half_w])  # num_patch个图像切片数据，大小为half_h和half_w
    result_den = []  # 图像crop部分内的ground_truth数据
    for i in range(num_patch):
        start_h = random.randint(0, img.shape[1] - half_h)
        start_w = random.randint(0, img.shape[2] - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # 切片操作很慢
        result_img[i] = img[:, start_h:end_h, start_w:end_w]  # 截取图像的某一矩形部分，CHW格式
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        record_den = den[idx]  # 使用布尔数组索引选出crop中的点，这是record_den，为ndarray，可以使用[:, 0]索引
        # 将数值改为crop中的相对坐标
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        #
        if record_den.shape[0] == 0:
            record_den = np.zeros((1, 2))
        #
        result_den.append(record_den)

    return result_img, result_den


if __name__ == "__main__":
    transform = transforms.Compose([
        vision.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        vision.ToTensor()
    ])
    # 当处理的是HWC格式时，用Normalize对最后一个维度进行数据处理，然后用一次ToTensor()转换为CHW格式

    train_set = SHHA(r'..\..\data_root', train=True, transform=transform, patch=True, flip=True)
    val_set = SHHA(r'..\..\data_root', train=False)
    for imgs, targets in train_set:
        print(imgs.shape, len(targets))
    for imgs, targets in val_set:
        print(imgs.shape, len(targets))
