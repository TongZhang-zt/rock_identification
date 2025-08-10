## 修改模型参数

- 模型的配置文件在/rock_identification/models/efficientnet目录下
- 在model_cfg中修改num_classes为自己数据集的类别数
- 按照自己电脑的配置修改batch_size, num_workers, learning_rate等参数
- 如果有训练权重，则修改pretrained_weights参数为自己的权重路径
- 如果要冻结某些层，则修改freeze_layers参数为需要冻结的层名称
- 在optimizer_cfg中修改优化器参数，如学习率、权重衰减等

## 训练模型
- 确保数据集已经准备好，包括训练集、测试集
- 在/rock_identification/tools/目录下运行train.py脚本，命令如下：
```
python tools/train.py models/efficientnet/efficientnet_b4.py
```
- 训练完成后，模型权重保存在/rock_identification/logs/EfficientNet目录下
