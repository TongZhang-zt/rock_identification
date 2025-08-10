import os
import sys
# from typing import Sequence
sys.path.insert(0,os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')   # 定义解析器
    parser.add_argument('config', help='train config file path')    # 定义参数
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # 读取配置文件获取关键字段
    args = parse_args()  # 解析命令行参数
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)    # 读取配置文件
    print_info(model_cfg)    # 打印模型信息

    # 初始化
    meta = dict()    # 存储训练信息
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # 保存训练信息的文件夹名
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)  # 保存训练信息的文件夹路径
    meta['save_dir'] = save_dir  # 保存训练信息的文件夹路径

    # 设置随机数种子
    seed = init_random_seed(args.seed)  # 随机数种子
    set_random_seed(seed, deterministic=args.deterministic)  # 设置随机数种子
    meta['seed'] = seed  # 保存随机数种子

    # 读取训练&制作验证标签数据
    total_annotations = "datas/train.txt"    # 训练标签数据
    with open(total_annotations, encoding='utf-8') as f:
        total_datas = f.readlines()
    if args.split_validation:
        total_nums = len(total_datas)   # 训练数据总数
        # indices = list(range(total_nums))
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)    # 设置随机数生成器
            rng.shuffle(total_datas)    # 打乱训练数据
        val_nums = int(total_nums * args.ratio)  # 验证数据总数
        folds = list(range(int(1.0 / args.ratio)))  # 验证数据划分
        fold = random.choice(folds)  # 随机选择一个验证数据集
        val_start = val_nums * fold    # 验证数据开始索引
        val_end = val_nums * (fold + 1)  # 验证数据结束索引
        train_datas = total_datas[:val_start] + total_datas[val_end:]    # 训练数据
        val_datas = total_datas[val_start:val_end]    # 验证数据
    else:
        train_datas = total_datas.copy()    # 训练数据
        test_annotations = 'datas/test.txt'
        with open(test_annotations, encoding='utf-8') as f:
            val_datas = f.readlines()

    if args.device is not None:
        device = torch.device(args.device)  # 指定训练设备
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # 自动选择训练设备
    print('Initialize the weights.')
    model = BuildNet(model_cfg)  # 构建模型
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()    # 初始化模型权重
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))  # 冻结层名称
        print('Freeze layers : ' + freeze_layers)    # 打印冻结层名称
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))  # 冻结层

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)

    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    # 制作数据集->数据增强&预处理
    train_dataset = Mydataset(train_datas, train_pipeline)
    val_pipeline = copy.deepcopy(train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                              num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'),
                            num_workers=data_cfg.get('num_workers'), pin_memory=True,
                            drop_last=True, collate_fn=collate)

    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer=optimizer,    # 优化器
        train_loader=train_loader,  # 训练数据集
        val_loader=val_loader,  # 验证数据集
        iter=0,  # 训练迭代次数
        epoch=0,    # 训练轮数
        max_epochs=data_cfg.get('train').get('epoches'),     # 最大训练轮数
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),  # 最大训练迭代次数
        best_train_loss=float('INF'),    # 最佳训练损失
        best_val_acc=float(0),  # 最佳验证精度
        best_train_weight='',    # 最佳训练权重
        best_val_weight='',    # 最佳验证权重
        last_weight=''  # 上一次保存权重
    )
    meta['train_info'] = dict(train_loss=[],      # 训练损失
                              val_loss=[],      # 验证损失
                              train_acc=[],      # 训练精度
                              val_acc=[])      # 验证精度

    # 是否从中断处恢复训练
    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir)    # 创建保存训练信息的文件夹
        shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))  # 保存配置文件
        model = init_model(model, data_cfg, device=device, mode='train')    # 初始化模型

    # 初始化保存训练信息类
    train_history = History(meta['save_dir'])

    # 记录初始学习率
    lr_update_func.before_run(runner)

    # 训练
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)    # 更新学习率
        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), data_cfg.get('test'),
              meta)  # 训练
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)  # 验证

        train_history.after_epoch(meta)  # 保存训练信息


if __name__ == "__main__":
    main()
