import json
import time
import requests
import numpy as np
import joblib
import faiss
import streamlit as st

# Silicon Flow API配置
SILICON_FLOW_API_KEY = "Bearer sk-fosjotncdjmdyrtscxlfxufqlacaslflxwkrpjtpzbrfzhdk"
MODEL = "BAAI/bge-m3"  # 查看模型信息 https://cloud.siliconflow.cn/models

def get_embedding(texts, max_retries=2, sleep_time=2):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": MODEL,  # 配置LLM embedding model
        "input": [texts],
        "encoding_format": "float"
    }
    headers = {
        "Authorization": SILICON_FLOW_API_KEY,  # 添加siliconflow API 密钥
        "Content-Type": "application/json"
    }
    for attempt in range(max_retries):
        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code == 200:
            response_embeds = json.loads(response.text)["data"]
            embeddings = response_embeds[0]["embedding"]
            return np.array(embeddings).reshape(1, -1).astype(np.float32)  # (1, 1024)
        elif response.status_code == 429:
            st.toast("Error code 429, try again", icon="⚠️")
            time.sleep(sleep_time)
            continue
        else:
            return None
    return None

def main():
    st.title("Paper Searching Engine")
    
    # 加载预处理好的数据
    @st.cache_resource
    def load_data():
        # 加载论文向量数据
        abs_embeddings, titles, ids, abs = joblib.load("processed_data/embeddings.pkl")
        papers = []
        for i in range(len(abs_embeddings)):
            papers.append({
                "title": titles[i],
                "id": ids[i],
                "abstract": abs[i]
            })
        # 加载Faiss索引
        index = faiss.read_index("processed_data/faiss_index.bin")
        
        return papers, index
    
    papers, index = load_data()
    st.toast("Data loaded successfully!")  # 推荐使用，会在右上角显示几秒后自动消失

    # 用户输入
    query = st.text_input("Input the content of the paper you want to search：")
    top_k = st.slider("Display the number of most similar papers", 1, 10, 5)
    
    if query:
        # 获取查询文本的embedding
        query_embedding = get_embedding(query)
        st.toast("Query embedding generated successfully!")  # 推荐使用，会在右上角显示几秒后自动消失
        
        # 搜索最相似的文档
        distances, indices = index.search(
            query_embedding.reshape(1, -1).astype("float32"), 
            top_k
        )
        
        # 显示结果
        st.subheader("Search Resulsts：")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            st.markdown(f"### {i+1}. 相似度得分: {1 - dist:.4f}")
            st.markdown(f"**Title：** {papers[idx]['title']} [论文链接](https://arxiv.org/abs/{papers[idx]['id']})")
            # st.markdown("[论文链接](https://arxiv.org/abs/1234.5678)")
            st.markdown(f"**Abstract：** {papers[idx]['abstract']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
