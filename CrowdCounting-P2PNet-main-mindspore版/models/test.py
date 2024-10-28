import mindspore
from mindspore.dataset import transforms, vision
from SHHA import SHHA
import backbone
from p2pnet import P2PNet, SetCriterion_Crowd
from matcher import build_matcher_crowd
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./SHHA',
                        help='path where the dataset is')

    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./checkpoint',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser


num_classes = 1
weight_dict = {'loss_ce': 1, 'loss_points': 0.0002}  # args.point_loss_coef
losses = ['labels', 'points']
parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
transform = transforms.Compose(
    [vision.ToTensor(),
     vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False), ])
backbone = backbone.build_backbone()  # mindspore模型库中的vgg16类
model = P2PNet(backbone)
# 下面是这次作业我们需要重写的的类和函数
matcher = build_matcher_crowd(args)
criterion = SetCriterion_Crowd(num_classes,
                               matcher=matcher, weight_dict=weight_dict,
                               eos_coef=args.eos_coef, losses=losses)
train_set = SHHA(r'../data_root', train=True, transform=transform, patch=True, flip=True)
for imgs, targets in train_set:
    targets = [{k: v for k, v in t.items()} for t in targets]
    imgs = mindspore.Tensor(imgs, dtype=mindspore.float32)
    outputs = model(imgs)
    loss_dict = criterion(outputs, targets)
    print(1)
    assert len(loss_dict) == 2
