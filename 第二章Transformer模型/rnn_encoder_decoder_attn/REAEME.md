# RNN Encoder-Decoder with Attention

训练RNN Encoder-Decoder with Attention模型做中文翻译成英文，基于单词(word)构建词表。

![](https://pytorch.org/tutorials/_static/img/seq-seq-images/seq2seq.png)


## Encoder

Encoder结构, 单向GRU：

![](https://pytorch.org/tutorials/_static/img/seq-seq-images/encoder-network.png)

## Decoder without Attention

without Attention的Decoder结构：

![](https://pytorch.org/tutorials/_static/img/seq-seq-images/decoder-network.png)

输入给decoder的第一个token是start-of-string `<SOS>`，用Encoder最后一个时刻的隐状态向量初始化decoder的隐状态向量。

## Decoder with Attention

with Attention的Decoder结构，参考课程中讲过的 [Neural Machine Translation by Jointly Learning to
Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)，这种注意力计算方式也被称为加性注意力(additive attention)，具体实现并不是完整的复现论文，比如没有修改GRU Decoder的结构。


![](https://pytorch.org/tutorials/_static/img/seq-seq-images/attention-decoder-network.png)

模型训练完成后，可以输入中文，得到英文。

## 训练数据

中英翻译数据链接：https://www.manythings.org/anki/cmn-eng.zip

下载cmn-eng.zip后解压缩，cmn.txt就是中英双语数据。每一行对应一条数据，数据格式：

``` {.sh}
Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
```

代码参考：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


## 训练

运行以下命令训练模型，会保存模型和两个语言的词表
```
python seq2seq_translation_tutorial.py
```

## 测试1

运行以下命令测试模型，会输出输入的中文和模型翻译的英文
```
python seq2seq_translation_tutorial.py --evaluate
```


## 测试2【推荐】

运行以下命令启动web应用，会输出输入的中文和模型翻译的英文
```
streamlit run app.py
```
