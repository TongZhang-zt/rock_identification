# from configs.common import *
from configs.backbones import *
from configs.necks import *
from configs.heads import *
from configs.common import BaseModule,Sequential

import torch.nn as nn
import torch

import functools
from inspect import getfullargspec
from collections import abc
import numpy as np

"""
### 定义函数build_model ###
根据传入的配置字典cfg，构建模型。如果传入的是一个列表，则构建Sequential容器；否则，构建单个模块
"""
def build_model(cfg):
    if isinstance(cfg, list):   # 如果cfg是列表，则构建Sequential容器
        modules = [
            eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:    # 如果cfg是字典，则构建单个模块
        return eval(cfg.pop("type"))(**cfg)

"""
### 定义类BuildNet ###
继承自BaseModule，表示网络的主要构建部分。通过配置信息构建主干、颈部和头部，提供了特征提取、模型前向传播等功能。
"""
class BuildNet(BaseModule):
    def __init__(self, cfg):    # 初始化模型各个部分
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")  # 定义颈部配置信息，若没有配置颈部，则为None
        self.head_cfg = cfg.get("head")  # 定义头部配置信息，若没有配置头部，则为None
        self.backbone = build_model(cfg.get("backbone"))    # 构建主干网络
        if self.neck_cfg is not None:    # 若配置了颈部，则构建颈部网络
            self.neck = build_model(cfg.get("neck"))

        if self.head_cfg is not None:    # 若配置了头部，则构建头部网络
            self.head = build_model(cfg.get("head"))

    #用于冻结模型中指定层的参数
    def freeze_layers(self, names):
        assert isinstance(names, tuple) # 确保传入的names是tuple类型
        for name in names:  # 遍历names，冻结相应层的参数
            layers = getattr(self, name)
            # layers.eval()
            for param in layers.parameters():    # 冻结参数
                param.requires_grad = False

    #用于从指定阶段提取特征
    def extract_feat(self, img, stage='neck'):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.

        Examples:
            1. Backbone output  # backbone 输出

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model    # ResNet18
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)    # Build the model
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model    # ViT-Base
            >>> model = build_classifier(cfg)
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')  # 确保stage参数合法

        x = self.backbone(img)  #将输入的图像输入到主干网络中，并得到输出

        if stage == 'backbone':  # 如果stage为backbone，则直接返回主干网络的输出
            return x

        if hasattr(self, 'neck') and self.neck is not None:  # 如果配置了颈部，则将主干网络的输出输入到颈部网络中
            x = self.neck(x)
        if stage == 'neck':
            return x

    # 定义模型的前向传播过程
    def forward(self, x, return_loss=True, train_statu=False, **kwargs):
        x = self.extract_feat(x)    # 特征提取

        if not train_statu:  # 训练状态下，将特征输入到头部网络中进行训练
            if return_loss:  # 如果是训练状态且需要返回损失，则将特征输入到头部网络中进行训练
                return self.forward_train(x, **kwargs)
            else:    # 如果是训练状态但不需要返回损失，则将特征输入到头部网络中进行测试
                return self.forward_test(x, **kwargs)
        else:    # 非训练状态下，将特征输入到头部网络中进行测试，同步返回测试结果和训练损失
            return self.forward_test(x), self.forward_train(x, **kwargs)
    # 模型训练阶段
    def forward_train(self, x, targets, **kwargs):

        losses = dict()  # 定义损失字典
        loss = self.head.forward_train(x, targets, **kwargs)    # 将特征输入到头部网络中进行训练，得到损失
        losses.update(loss)  # 将损失更新到损失字典中
        return losses
    #模型测试阶段
    def forward_test(self, x, **kwargs):

        out = self.head.simple_test(x, **kwargs)    # 将特征输入到头部网络中进行测试，得到测试结果
        return out

