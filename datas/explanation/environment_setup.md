## Anaconda 安装

Anaconda 是 Python 语言的一个开源发行版本，包含了众多科学计算、数据处理、机器学习等常用库。Anaconda 是一个基于 Python 的开源数据科学环境，可以用于数据分析、科学计算、机器学习等领域。

Anaconda 官网：https://www.anaconda.com/

Anaconda 下载：https://www.anaconda.com/download/success

下载后，根据系统类型选择安装包进行安装即可。

常见命令：

- 查看 conda 版本：`conda --version`
- 查看 conda 环境：`conda env list`
- 创建 conda 环境：`conda create -n env_name python=3.7`
- 激活 conda 环境：`conda activate env_name`
- 退出 conda 环境：`conda deactivate`
- 删除 conda 环境：`conda remove -n env_name --all`

具体安装过程可以参考教程：[Window安装Anaconda](https://mp.csdn.net/mp_blog/creation/editor/149695296)

## pytorch 安装
pytorch 是一个基于 Python 的开源深度学习库，可以用于构建和训练神经网络。

pytorch 官网：https://pytorch.org/

pytorch 安装：

```
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

安装完成后，可以测试是否安装成功：

```
import torch
print(torch.__version__)
```

如果输出版本号，则说明安装成功。   

安装完成后再运行
```
pip install -r requirements.txt
```

即可安装项目所需的依赖库。
