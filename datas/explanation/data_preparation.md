## 数据集准备

本示例以岩石数据集为例，目录结构如下：
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
在/rock_identification/datas目录下创建annotations.txt文件，内容如下：
```
class1 0
class2 1
class3 2
class4 3
```

## 数据集划分
将数据集划分为训练集和测试集

打开文件/rock_identification/tools/split_data.py文件

```
init_dataset = '../rock_photos'    #未划分前的数据集路径,修改为自己的路径
new_dataset = '../datasets'  #划分后的数据集路径，修改为自己的路径
```
运行文件后得到划分后的数据集格式如下：
```
datasets/
├── train/
│   ├── class1
|   |   ├── rock1.jpg
|   |   ├── rock2.jpg
|   |   ├── rock3.jpg
|   |   ├── rock4.jpg
|   |   └── rock5.jpg
│   ├── class2
|   |   ├── rock1.jpg
|   |   ├── rock2.jpg
|   |   ├── rock3.jpg
|   |   ├── rock4.jpg
|   |   └── rock5.jpg
│   ├── class3
|   |   ├── rock1.jpg
|   |   ├── rock2.jpg
|   |   ├── rock3.jpg
|   |   ├── rock4.jpg
|   |   └── rock5.jpg
│   └── class4
|       ├── rock1.jpg
|       ├── rock2.jpg
|       ├── rock3.jpg
|       ├── rock4.jpg
|       └── rock5.jpg
└── test/
    ├── class1
    |   ├── rock1.jpg
    |   ├── rock2.jpg
    |   ├── rock3.jpg
    |   ├── rock4.jpg
    |   └── rock5.jpg
    ├── class2
    |   ├── rock1.jpg
    |   ├── rock2.jpg
    |   ├── rock3.jpg
    |   ├── rock4.jpg
    |   └── rock5.jpg
    ├── class3
    |   ├── rock1.jpg
    |   ├── rock2.jpg
    |   ├── rock3.jpg
    |   ├── rock4.jpg
    |   └── rock5.jpg
    └── class4
        ├── rock1.jpg
        ├── rock2.jpg
        ├── rock3.jpg
        ├── rock4.jpg
        └── rock5.jpg
```

## 数据集信息文件制作

确保划分后数据集在/rock_identification/datasets目录下，打开文件/rock_identification/tools/get_annotation.py文件
```
datasets_path = '/rock_identification/datasets' #修改为自己的路径,建议为绝对路径，否则容易出错
```
运行文件后得到数据集信息文件在/rock_identification/datas目录下，文件名为test.txt和train.txt，内容如下：
```
/rock_photos/class1/rock1.jpg 0
/rock_photos/class1/rock2.jpg 0
/rock_photos/class1/rock3.jpg 0
/rock_photos/class1/rock4.jpg 0
/rock_photos/class1/rock5.jpg 0
/rock_photos/class2/rock1.jpg 1
/rock_photos/class2/rock2.jpg 1
/rock_photos/class2/rock3.jpg 1
/rock_photos/class2/rock4.jpg 1
/rock_photos/class2/rock5.jpg 1
/rock_photos/class3/rock1.jpg 2
/rock_photos/class3/rock2.jpg 2
/rock_photos/class3/rock3.jpg 2
/rock_photos/class3/rock4.jpg 2
/rock_photos/class3/rock5.jpg 2
/rock_photos/class4/rock1.jpg 3
/rock_photos/class4/rock2.jpg 3
/rock_photos/class4/rock3.jpg 3
/rock_photos/class4/rock4.jpg 3
/rock_photos/class4/rock5.jpg 3
```

