import argparse
import datetime
import random
import time
from pathlib import Path

import mindspore
from mindspore.dataset import transforms, vision
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
#from engine import *
from models.p2pnet import build
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./checkpoint/latest_epoch2.ckpt',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):
    print(args)
    # get the P2PNet
    model = build(args,training=False)
    # load trained model
    if args.weight_path is not None:
        param_dict = mindspore.load_checkpoint(args.weight_path)
        param_not_load, _ = mindspore.load_param_into_net(model, param_dict)

    # create the pre-processing transform
    transform = transforms.Compose(
        [vision.ToTensor(),
         vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False), ])

    # set your image path here
    img_path = "./vis/demo1.jpg"  # ./data_root/test/Images/IMG_9.jpg
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)
    samples = mindspore.Tensor(img)
    # run inference
    outputs = model(samples)
    outputs_scores = mindspore.ops.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    threshold = 0.01
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
