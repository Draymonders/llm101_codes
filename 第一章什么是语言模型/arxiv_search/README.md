# 基于arXiv论文数据 + SiliconFlow API + faiss + streamlit 构建论文搜索引擎demo


数据下载页面：https://www.kaggle.com/datasets/Cornell-University/arxiv

解压缩得到文件 arxiv-metadata-oai-snapshot.json

然后运行 1.arxiv_eda.ipynb 进行数据探索性分析和数据过滤，我只保留了cs下面4个领域的数据，得到arxiv_cs-metadata.json

然后运行 2.preprocess_paper.ipynb 调用SiliconFlow API，得到62000个paper的title + abstract 的embedding，并保存到processed_data/embeddings.pkl 对embedding进行faiss索引，保存到 processed_data/embeddings.pkl

最后运行 main.py 进行搜索:
```
streamlit run main.py
```

Note:
* 由于数据比较大，我没有上传到GitHub