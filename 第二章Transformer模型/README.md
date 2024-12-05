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

RNN Encoder-Decoder with Attention 代码在`rnn_encoder_decoder_attn`目录下。

The Annotated Transformer 的链接在 [点击](https://nlp.seas.harvard.edu/annotated-transformer/)

基于OpenNMT-py训练Transformer模型做英德翻译任务的示例代码 [点击](https://github.com/OpenNMT/OpenNMT-py/tree/master/docs/source/examples/wmt17)

如果要改成中英翻译，建议的数据集 [en-zh](https://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/)