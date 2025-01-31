# 第一章什么是语言模型

## 环境配置

1. 运行下面命令创建conda环境
```
conda env create -f environment.yml
```

environment.yml文件中列出了所有需要的包，运行这个命令会自动安装所有需要的包。有可能比较慢，请耐心等待。或者使用第二种方式：

2. 运行下面命令创建conda环境

```
conda create -n llm101 python=3.11
pip install requests scikit-learn numpy==1.26.4 seaborn torch transformers umap-learn
```
然后执行代码过程中，看看报错信息，根据报错信息安装缺失的包。


## 教程

### 一、词向量可视化

进入[Embedding Projector网站](https://projector.tensorflow.org/)，使用`Word2Vec 10K`并用降维的方式，查看向量的分布，相近向量表示了：在同一个上下文语义环境下，出现的位置相近。

### 二、 计算句子向量的余弦相似度

句子向量相近通常有如下含义

1. 语义相似性
2. 上下文关系
3. 功能相似性

实际应用

- 文本相似度匹配
- 智能问答系统
- 文本聚类
- 文本推荐系统
- 语义搜索引擎

### 三、中文文本分类

通过bert+MSE进行多分类，对头条新闻标题进行分类