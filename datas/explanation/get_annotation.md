# get_annotation.py文件解释
### 文件主要功能
- 该文件主要用来获取训练集和测试集数据集图片的路径信息，并将其保存到txt文件中。
- 该文件主要调用了`utils/train_utils.py`中的`get_info`函数。

### 文件代码
#### 头文件导入
- os用于获取数据集图片的路径信息。
- sys用于将当前工作目录加入系统路径。
- train_utils用于获取数据集图片的路径信息。
```python
import os
import sys
sys.path.insert(0, os.getcwd())  # 加入当前工作目录到系统路径
from utils.train_utils import get_info
```
#### 获取数据集图片的路径信息
- 调用`get_info`函数，获取数据集图片的路径信息。get_info函数在utils/train_utils.py文件中。
- 将获取到的图片路径信息保存到txt文件中。
```python
def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name,index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
        
    return names,indexs
```
classes_path为数据集类别信息文件路径。其中annotations.txt文件内容格式为：
```
class_name index
```
其中class_name为类别名称，index为类别索引。
#### 调用get_info函数获取数据集图片的路径信息
- 调用`get_info`函数，获取数据集图片的路径信息。
- 将获取到的图片路径信息保存到txt文件中。
- datasets为数据集名称列表，datasets_path为数据集路径。
- classes_path为数据集类别信息文件路径。
- os.path.splitext用于获取文件后缀名。
```python
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
                _, postfix = os.path.splitext(photo_name) 
                if postfix not in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    continue
                txt_file.write('%s' % (os.path.join(images_path, photo_name)) + ' ' + str(cls_id)) 
                txt_file.write('\n')  
        txt_file.close()   
```
最终在[../datas](/rock_identification/datas)目录下生成相应数据集的txt文件，文件内容格式为：
```
image_path cls_id
```
其中image_path为图片路径，cls_id为类别索引。在这里会生成test.txt和train.txt文件。


