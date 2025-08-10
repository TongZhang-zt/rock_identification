import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import mean
import os
import csv
#from torch.utils.tensorboard import SummaryWriter

class History():
    def __init__(self, dir):
        self.dir = dir  # 保存路径
        self.csv_dir = os.path.join(dir,'metrics_outputs.csv')  # 保存csv路径
        self.pic_dir = os.path.join(dir,'loss-acc.png')  # 保存loss-acc图片路径
        self.losses_epoch = []  # 保存每周期的训练损失
        self.acc_epoch = []  # 保存每周期的准确率
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Val Acc', 'Precision', 'Recall', 'F1 Score']]    #存储每个周期的各项指标，初始值为标头信息
        self.temp_data = []  # 临时存储每周期的各项指标
    #更新方法
    def update(self, data, mode):
        if mode == 'train':
            self.temp_data.append(data)
            self.losses_epoch.append(data)
        elif mode == 'test':
            self.temp_data.extend([data.get('accuracy_top-1'),mean(data.get('precision',0.0)),mean(data.get('recall',0.0)),mean(data.get('f1_score',0.0))])
            self.acc_epoch.append(data.get('accuracy_top-1'))
    #绘制loss-acc曲线
    def draw_loss_acc(self, loss, acc, save_path):
        total_epoch = range(1,len(loss)+1)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(total_epoch, loss, 'red', linewidth = 2, label='loss')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc')
        ax2.plot(total_epoch, acc, 'blue', linewidth = 2, label='acc')
        fig.legend()
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close("all")
        
    #周期结束后保存数据
    def after_epoch(self, meta):
        '''
        保存每周期的 'Train Loss', 'Val Acc', 'Precision', 'Recall', 'F1 Score'
        '''
        val_acc_epoch = []
        train_acc_epoch = []
        epoch_outputs = [['index', 'train_loss', 'train_acc', 'train_precision', 'train_recall', 'train_f1-score', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1-score']]
        with open(self.csv_dir, 'w', newline='') as f:
            writer          = csv.writer(f)
            for i in range(len(meta['train_info']['train_loss'])):
                temp_data = [i+1, meta['train_info']['train_loss'][i], meta['train_info']['train_acc'][i].get('accuracy_top-1'),mean(meta['train_info']['train_acc'][i].get('precision',0.0)),mean(meta['train_info']['train_acc'][i].get('recall',0.0)),mean(meta['train_info']['train_acc'][i].get('f1_score',0.0)), meta['train_info']['val_loss'][i], meta['train_info']['val_acc'][i].get('accuracy_top-1'),mean(meta['train_info']['val_acc'][i].get('precision',0.0)),mean(meta['train_info']['val_acc'][i].get('recall',0.0)),mean(meta['train_info']['val_acc'][i].get('f1_score',0.0))]
                val_acc_epoch.append(meta['train_info']['val_acc'][i].get('accuracy_top-1'))
                train_acc_epoch.append(meta['train_info']['train_acc'][i].get('accuracy_top-1'))
                epoch_outputs.append(temp_data)
            writer.writerows(epoch_outputs)

        '''
        绘制每周期Train|Val Loss-Accuracy
        '''
        train_loss_acc_pic = os.path.join(self.dir, 'train_loss-acc.png')
        self.draw_loss_acc(meta['train_info']['train_loss'], train_acc_epoch, train_loss_acc_pic)
        
        val_loss_acc_pic = os.path.join(self.dir, 'val_loss-acc.png')
        self.draw_loss_acc(meta['train_info']['val_loss'], val_acc_epoch, val_loss_acc_pic)
        
