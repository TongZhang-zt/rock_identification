import os
import sys

sys.path.insert(0, os.getcwd())  # 加入当前工作目录到系统路径
from utils.train_utils import get_info


def main():
    classes_path = '../datas/annotations.txt'
    datasets_path = 'C:/Users/zhangtong/Desktop/rock_identification/datasets'
    datasets = ["train", "test"]
    classes, indexs = get_info(classes_path)    # 获取类别和索引

    for dataset in datasets:
        txt_file = open(
            '../datas/' + dataset + '.txt',
            'w')
        datasets_path_ = os.path.join(datasets_path, dataset)   # 获取数据集路径
        classes_name = os.listdir(datasets_path_)   # 获取类别文件夹名称

        for name in classes_name:
            if name not in classes:
                continue
            cls_id = indexs[classes.index(name)]    # 获取类别索引
            images_path = os.path.join(datasets_path_, name)    # 获取类别文件夹路径
            images_name = os.listdir(images_path)   # 获取类别图片名称
            for photo_name in images_name:
                _, postfix = os.path.splitext(photo_name)    # 获取图片后缀
                if postfix not in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    continue
                txt_file.write('%s' % (os.path.join(images_path, photo_name)) + ' ' + str(cls_id))   # 写入数据集信息
                txt_file.write('\n')    # 换行
        txt_file.close()     # 关闭文件


if __name__ == "__main__":
    main()
