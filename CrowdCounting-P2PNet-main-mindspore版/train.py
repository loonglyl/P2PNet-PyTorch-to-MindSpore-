import argparse
import random
import time
import datetime
import sys
from tensorboardX import SummaryWriter  # from pathlib import Path
import math
from crowd_datasets import build_dataset  # from .crowd_datasets.SHHA.loading_data import loading_data
import os
from models.p2pnet import build
import numpy as np
import mindspore
from mindspore import nn
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    # 超参数配置
    parser.add_argument('--lr', default=1e-4, type=float)  # 1e-4
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)  # 8
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3, type=int)  # 3500
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16', type=str,  # vgg16_bn
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
    parser.add_argument('--data_root', default='./data_root',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./checkpoint',  # "ckpt"
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # ./checkpoint/latest_epoch4.ckpt
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = mindspore.Tensor(list(self.deque))
        return d.median()  # .item()

    @property
    def avg(self):
        d = mindspore.Tensor(list(self.deque), dtype=mindspore.float32)
        return d.mean()  # .item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, mindspore.Tensor):
                v = v.item(0)
                v = float(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ])

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    # device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed  # seed = args.seed + utils.get_rank()
    mindspore.set_seed(seed)  # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build(args, training=True)  # build_model
    # 统计网络需要求导的参数数量
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    n_parameters = sum(p.numel() for p in model.trainable_params() if p.requires_grad)
    print('number of params:', n_parameters)

    # use different optimization params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model.parameters_and_names() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.parameters_and_names() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]  # 字典列表
    lr_scheduler = nn.piecewise_constant_lr(milestone=[i * args.lr_drop for i in range(1, 100)]
                                            , learning_rates=[pow(0.9, i - 1) * args.lr for i in range(1, 100)])
    optimizer = nn.Adam(param_dicts, learning_rate=lr_scheduler)  # lr_scheduler作为optimizer的输入，list类型，optimizer的step在哪
    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root)
    if args.frozen_weights is not None:
        checkpoint = mindspore.load_checkpoint(args.frozen_weights)
        param_not_load, _ = mindspore.load_param_into_net(model, checkpoint)
        assert len(param_not_load) == 0  # 为空表示所有参数都加载成功
    if args.resume:  # 恢复训练
        checkpoint = mindspore.load_checkpoint(args.resume)
        param_not_load, _ = mindspore.load_param_into_net(model, checkpoint)
        assert len(param_not_load) == 0
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    def forward_fn(data, label):
        outputs = model(data)  # outputs为预测结果，logits
        loss_dict = criterion(outputs, label)  # 损失函数计算损失值
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce all losses
        loss_dict_reduced = loss_dict
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled  # .item()，loss  forward_fn
        return loss_value, outputs, loss_dict_reduced_scaled, loss_dict_reduced_unscaled

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _, loss_dict_reduced_scaled, loss_dict_reduced_unscaled), grads = grad_fn(data, label)
        if not math.isfinite(loss):  # 损失值无穷大时停止训练
            print("Loss is {}, stopping training".format(loss))
            # print(loss_dict_reduced)
            sys.exit(1)
        if args.clip_max_norm > 0:
            # mindspore.ops.clip_by_value(model.trainable_params(), args.clip_max_norm)
            mindspore.ops.clip_by_value(grads, args.clip_max_norm)
        optimizer(grads)
        return loss, loss_dict_reduced_scaled, loss_dict_reduced_unscaled

    def train_loop():
        size = len(train_set)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        for batch, (samples, targets) in enumerate(train_set):
            targets = [{k: v for k, v in t.items()} for t in targets]
            loss, loss_dict_reduced_scaled, loss_dict_reduced_unscaled = train_step(samples, targets)
            # update logger
            metric_logger.update(loss=loss, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(lr=optimizer.get_lr()[0])  # 记录学习率
            print(f"loss: {loss.asnumpy():>7f}  [{batch:>3d}/{size:>3d}]")
        # gather the stats from all processes
        # print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    print("Start training")
    start_time = time.time()
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    model.set_train()
    criterion.set_train()
    # training starts here, train_loop
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_loop()
        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.get_lr()[0], t2 - t1))  # param_groups[0]['lr']
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.get_lr()[0], t2 - t1))
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        # mindspore.save_checkpoint({'model': model}, checkpoint_latest_path)
        mindspore.save_checkpoint(model, checkpoint_latest_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
