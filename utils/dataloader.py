from random import shuffle
from PIL import Image
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose
"""
### 定义自己的Dataset类 ###
继承自 torch.utils.data.Dataset，这意味着 Mydataset 可以像其他 PyTorch 数据集一样被使用。
"""
class Mydataset(Dataset):
    """
    初始化，接受两个参数：
    - gt_labels: 包含图片路径和标签的列表
    - cfg: 配置文件，定义了图像预处理所需要的变换
    """
    def __init__(self, gt_labels, cfg):
        self.gt_labels   = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)   #初始化一个Compose对象，该对象将根据 cfg 中的配置对图像进行预处理。
        self.data_infos = self.load_annotations()    #加载图片路径和标签

    def __len__(self):  #返回数据集的长度
        
        return len(self.gt_labels)

    # def __getitem__(self, index):
    #     image_path = self.gt_labels[index].split(' ')[0].split()[0]
    #     image = Image.open(image_path)
    #     cfg = copy.deepcopy(self.cfg)
    #     image = self.preprocess(image,cfg)
    #     gt = int(self.gt_labels[index].split(' ')[1])
        
    #     return image, gt, image_path
    
    # def preprocess(self, image,cfg):
    #     if not (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
    #         image = image.convert('RGB')
    #     funcs = []

    #     for func in cfg:
    #         funcs.append(eval('transforms.'+func.pop('type'))(**func))
    #     image = transforms.Compose(funcs)(image)
    #     return image

    def __getitem__(self, index):   #返回数据集中第index个样本的图像、标签和路径。
        # 调用Compose对象对图像进行预处理。copy.deepcopy(self.data_infos[index]) 用于复制数据信息以避免在预处理过程中修改原始数据。
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        return results['img'], int(results['gt_label']), results['filename']
    #用于加载和解析图像路径和标签信息。
    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]
        
        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
# 将多个样本组合成一个批次
# 输入是一个批次的样本列表，每个样本包含图像、标签和路径
def collate(batches):
    images, gts, image_path = tuple(zip(*batches))  # 将图像、标签和路径分别压成三个独立的元组
    images = torch.stack(images, dim=0) #将多个图像张量堆叠在一起形成一个批次的图像张量
    gts = torch.as_tensor(gts)  #将标签列表转换为张量
    
    return images, gts, image_path
