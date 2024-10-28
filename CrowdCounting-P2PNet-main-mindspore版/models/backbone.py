# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import mindspore
import mindspore.ops as ops
from mindspore import nn
from . import vgg_


class BackboneBase_VGG(nn.Cell):
    def __init__(self, backbone: nn.Cell, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.layers.cells())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.SequentialCell(*features[:13])
                self.body2 = nn.SequentialCell(*features[13:23])
                self.body3 = nn.SequentialCell(*features[23:33])
                self.body4 = nn.SequentialCell(*features[33:43])
            else:
                self.body1 = nn.SequentialCell(*features[:9])
                self.body2 = nn.SequentialCell(*features[9:16])
                self.body3 = nn.SequentialCell(*features[16:23])
                self.body4 = nn.SequentialCell(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.SequentialCell(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.SequentialCell(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def construct(self, tensor_list):  # 传入数据时用到的函数，mindspore调用construct函数
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        # 实例化pytorch框架自带vgg16_bn或vgg16模型
        # if name == 'vgg16_bn':
        #     backbone = vgg_.vgg16_bn(pretrained=True)
        # elif name == 'vgg16':
        if name == 'vgg16':
            backbone = vgg_.vgg16()  # pretrained=True
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone():
    # backbone = Backbone_VGG(args.backbone, True)
    backbone = Backbone_VGG('vgg16', True)
    return backbone


if __name__ == '__main__':
    back = build_backbone()
    print(back)
    x = ops.standard_normal((4, 3, 128, 128))
    print(back(x)[0].shape, back(x)[1].shape, back(x)[2].shape)  # (4,1000)
