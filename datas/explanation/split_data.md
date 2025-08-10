# split_data.py文件解释
### 文件主要功能
- 该文件用于将数据集划分为训练集和测试集。
- 训练集用于训练模型，测试集用于评估模型的性能。
- 该文件使用os创建数据集的目录，使用random模块随机划分数据集。

### 文件代码
#### 头文件导入：
```python 
import os
from shutil import copy, rmtree
import random
from tqdm import tqdm
```
#### 创建数据集目录：
如果目录存在，则删除目录，创建目录。防止原始数据存在影响。
- os.path.exists(path)用于判断目录是否存在。
- rmtree(path)用于删除目录。
- os.makedirs(path)用于创建目录。
```python
def makedir(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)
```
#### 划分数据集：
- split_rate用于设置划分比例，这里设置为0.2。
- init_dataset用于设置原始数据集的路径。原始图片就放在这个目录下。
- new_dataset用于设置划分后数据集的路径。划分后的数据集在这个目录下。
- random.seed(0)用于固定随机数种子，保证每次划分数据集的结果相同。
- classes_name用于获取原始数据集的类别名称。
```python
split_rate = 0.2
init_dataset = '../rock_photos' #根据自己的路径修改
new_dataset = '../datasets' #根据自己的路径修改
random.seed(0)

classes_name = [name for name in os.listdir(init_dataset)]

makedir(new_dataset)
training_set = os.path.join(new_dataset, "train")
test_set = os.path.join(new_dataset, "test")
makedir(training_set)
makedir(test_set)
```
这里以岩石数据为例，原始数据集的路径为`../rock_photos`，划分后数据集的路径为`../datasets`。
原始数据集目录如下：

```
rock_photos/
├── class1
│   ├── rock1.jpg
│   ├── rock2.jpg
│   ├── rock3.jpg
│   ├── rock4.jpg
│   └── rock5.jpg
├── class2
│   ├── rock1.jpg
│   ├── rock2.jpg
│   ├── rock3.jpg
│   ├── rock4.jpg
│   └── rock5.jpg
├── class3
│   ├── rock1.jpg
│   ├── rock2.jpg
│   ├── rock3.jpg
│   ├── rock4.jpg
│   └── rock5.jpg
└── class4
    ├── rock1.jpg
    ├── rock2.jpg
    ├── rock3.jpg
    ├── rock4.jpg
    └── rock5.jpg
```

划分后数据集目录如下：

```
datasets/
├── train
│   ├── class1
│   ├── class2
│   ├── class3
│   └── class4
└── test
    ├── class1
    ├── class2
    ├── class3
    └── class4
```

#### 复制图片到训练集和测试集：
- 遍历每个类别，获取该类别的图片列表。
- 随机选取测试集图片的索引。
- 遍历每个图片，如果图片的索引在测试集索引列表中，则复制到测试集目录下，否则复制到训练集目录下。
- 使用tqdm模块显示进度条。
- os.listdir(path)用于获取目录下的文件列表。
- os.path.join(path, file)用于拼接目录和文件名。
- random.sample(population, k)用于随机选取k个元素。
- copy(src, dst)用于复制文件。
- pbar.update(1)用于更新进度条。
```python
for cla in classes_name:
    class_path = os.path.join(init_dataset, cla) 
    img_set = os.listdir(class_path) 
    num = len(img_set)
    test_set_index = random.sample(img_set, k=int(num * split_rate))
    with tqdm(total=num, desc=f'Class : ' + cla, mininterval=0.3) as pbar:
        for _, img in enumerate(img_set): 
            if img in test_set_index:
                init_img = os.path.join(class_path, img)
                new_img = os.path.join(test_set, cla)
                copy(init_img, new_img)
            else: 
                init_img = os.path.join(class_path, img)
                new_img = os.path.join(training_set, cla)
                copy(init_img, new_img)
            pbar.update(1) 
    print()
```
运行这个文件最终在`../datasets`目录下会生成两个文件夹，分别是`train`和`test`。`train`文件夹下包含训练集，`test`文件夹下包含测试集。
