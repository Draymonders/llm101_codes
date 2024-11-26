有两种方式创建本章代码的运行环境：

1. 运行下面命令创建conda环境
```
conda env create -f environment.yml
```

environment.yml文件中列出了所有需要的包，运行这个命令会自动安装所有需要的包。有可能比较慢，请耐心等待。或者使用第二种方式：

2. 运行下面命令创建conda环境

```
conda create -n llm101 python=3.11

```
然后执行代码过程中，看看报错信息，根据报错信息安装缺失的包。