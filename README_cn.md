# 教程

## 目录

- [efficientnet 模型介绍](./datas/explanation/efficientnet_model.md)
- [环境搭建](./datas/explanation/environment_setup.md)
- [数据集准备](./datas/explanation/data_preparation.md)
- [模型训练](./datas/explanation/model_training.md)

## 模型介绍
EfficientNet 是 Google 提出的一种新的模型架构，其目的是为了解决深度学习模型的效率问题。EfficientNet 架构的核心思想是利用多个不同大小的卷积核，并通过不同层的堆叠来提升模型的准确性和效率。EfficientNet 架构的特点是：

- 轻量级模型：EfficientNet 架构的模型大小只有 4.5MB，相比于 ResNet-50 等更深的模型，其轻量级程度更突出。
- 高效计算：EfficientNet 架构的计算复杂度与参数量成正比，这使得其在移动端、嵌入式设备等资源有限的场景下有着更高的实用价值。
- 高准确率：EfficientNet 架构的准确率在 ImageNet 数据集上超过了 ResNet-50 等更深的模型。

## 环境搭建
- 安装 Python 环境
- 安装 PyTorch 环境
- 安装相关库
- 详细细节请参考 [环境搭建](./datas/explanation/environment_setup.md)

## 数据集准备
- 下载数据集
- 数据预处理
- 数据加载
- 详细细节请参考 [数据集准备](./datas/explanation/data_preparation.md)

## 模型训练
- 定义模型
- 定义损失函数
- 定义优化器
- 训练模型
- 保存模型
- 详细细节请参考 [模型训练](./datas/explanation/model_training.md)


## 预训练权重下载
- [efficientnet-b0](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth)
- [efficientnet-b1](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth)
- [efficientnet-b2](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth)
- [efficientnet-b3](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth)
- [efficientnet-b4](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth)
- [efficientnet-b5](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32_in1k_20220119-e9814430.pth)
- [efficientnet-b6](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119-45b03310.pth)
- [efficientnet-b7](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth)
- [efficientnet-b8](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7.pth)