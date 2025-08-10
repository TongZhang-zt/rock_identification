import os
from shutil import copy, rmtree
import random
from tqdm import tqdm   #生成进度条，显示进度


def main():
    '''
    split_rate  : 测试集划分比例
    init_dataset: 未划分前的数据集路径
    new_dataset : 划分后的数据集路径

    '''

    def makedir(path):
        if os.path.exists(path):
            rmtree(path)    #删除文件夹及其内容，如果存在，则先删除，再创建，避免旧数据残留
        os.makedirs(path)

    split_rate = 0.2    #测试集划分比例，这里设置为0.2，即20%划分到测试集，可以根据实际情况调整
    init_dataset = '../rock_photos'    #未划分前的数据集路径,修改为自己的路径
    new_dataset = '../datasets'  #划分后的数据集路径，修改为自己的路径
    random.seed(0)  #设置随机数种子，保证每次划分结果相同

    classes_name = [name for name in os.listdir(init_dataset)]  #获取类别名称

    makedir(new_dataset)    #创建划分后的数据集路径
    training_set = os.path.join(new_dataset, "train")   #训练集路径
    test_set = os.path.join(new_dataset, "test")    #测试集路径
    makedir(training_set)   #创建训练集路径
    makedir(test_set)   #创建测试集路径

    for cla in classes_name:
        makedir(os.path.join(training_set, cla))    #创建训练集类别路径
        makedir(os.path.join(test_set, cla))    #创建测试集类别路径

    for cla in classes_name:
        class_path = os.path.join(init_dataset, cla)    #获取类别路径
        img_set = os.listdir(class_path)    #获取类别图片集
        num = len(img_set)  #获取类别图片数量
        test_set_index = random.sample(img_set, k=int(num * split_rate))    #随机选取测试集图片
        with tqdm(total=num, desc=f'Class : ' + cla, mininterval=0.3) as pbar:
            for _, img in enumerate(img_set):    #遍历类别图片集
                if img in test_set_index:    #如果图片在测试集图片集中，则复制到测试集路径
                    init_img = os.path.join(class_path, img)
                    new_img = os.path.join(test_set, cla)
                    copy(init_img, new_img)
                else:    #如果图片不在测试集图片集中，则复制到训练集路径
                    init_img = os.path.join(class_path, img)
                    new_img = os.path.join(training_set, cla)
                    copy(init_img, new_img)
                pbar.update(1)   #更新进度条
        print()


if __name__ == '__main__':
    main()