# model settings

model_cfg = dict(
    backbone=dict(type='EfficientNet', arch='b4'),  # 骨干网络
    neck=dict(type='GlobalAveragePooling'),  # 全局池化
    head=dict(
        type='LinearClsHead',   # 分类头
        num_classes=11, # 类别数
        in_channels=1792,   # 输入通道数
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),    # 损失函数
        topk=(1, 5),        # 前k个结果
        ))

# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)    # 图像归一化配置
train_pipeline = [
    dict(type='LoadImageFromFile'),        # 加载图像
    dict(
        type='RandomResizedCrop',   # 随机裁剪
        size=380,                   # 裁剪尺寸
        efficientnet_style=True,    # 按照EfficientNet的裁剪方式
        interpolation='bicubic'),   # 裁剪方式
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),  # 随机翻转
    dict(type='Normalize', **img_norm_cfg),                          # 归一化
    dict(type='ImageToTensor', keys=['img']),                        # 图像转tensor
    dict(type='ToTensor', keys=['gt_label']),                        # 标签转tensor
    dict(type='Collect', keys=['img', 'gt_label'])                   # 收集数据
]

val_pipeline = [
    dict(type='LoadImageFromFile'),   # 加载图像
    dict(
        type='CenterCrop',    # 中心裁剪
        crop_size=380,     # 裁剪尺寸
        efficientnet_style=True,    # 按照EfficientNet的裁剪方式
        interpolation='bicubic'),   # 裁剪方式
    dict(type='Normalize', **img_norm_cfg),  # 归一化
    dict(type='ImageToTensor', keys=['img']),    # 图像转tensor
    dict(type='Collect', keys=['img'])    # 收集数据
]

# train
data_cfg = dict(
    batch_size = 4, # 批大小
    num_workers = 4,    # 加载数据线程数
    train = dict(
        pretrained_flag = True,    # 是否加载预训练模型
        pretrained_weights = 'C:/Users/zhangtong/Desktop/rock_identification/datas/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth',    # 预训练模型路径
        freeze_flag = True,        # 是否冻结骨干网络
        freeze_layers = ('backbone',),  # 冻结层
        epoches = 100,              # 训练轮数
    ),
    test=dict(
        ckpt = '',    # 测试模型路径
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'], # 评估指标
        metric_options = dict(
            topk = (1,5),    # 前k个结果
            thrs = None,     # 阈值
            average_mode='none'  # 平均方式
    )
    )
)

# batch 4
# lr = 0.1 *4 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',  # 优化器
    lr=0.1 * 4/256,   # 学习率
    momentum=0.9,    # 动量
    weight_decay=1e-4)  # 权重衰减

# learning
lr_config = dict(
    type='StepLrUpdater',    # 学习率更新器
    step=[30, 60, 90]    # 学习率更新步长
)

