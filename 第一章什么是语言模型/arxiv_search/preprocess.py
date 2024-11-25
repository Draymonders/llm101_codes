import os
import json
import numpy as np
import faiss
import requests
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple

# Silicon Flow API configuration
SILICON_FLOW_API_KEY = "Bearer sk-fosjotncdjmdyrtscxlfxufqlacaslflxwkrpjtpzbrfzhdk"
SILICON_FLOW_MODEL = "BAAI/bge-m3"  # "BAAI/bge-large-en-v1.5"


def get_embedding(text: str) -> np.ndarray:
    """Call Silicon Flow API to get text embedding vector"""
    url = "https://api.siliconflow.cn/v1/embeddings"

    payload = {
        "model": model,  # Configure LLM embedding model
        "input": texts,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": api_key,  # Add siliconflow API key, https://cloud.siliconflow.cn/account/ak
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 200:
        return np.array(response.json()['embedding'])
    else:
        raise Exception(f"API call failed: {response.text}")

def process_papers(json_file):
    """Process paper data and generate embeddings for abstracts"""
    with open(json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Get embeddings for all abstracts
    embeddings = []
    for paper in tqdm(papers, desc="Generating embeddings"):
        embedding = get_embedding(paper['abstract'])
        embeddings.append(embedding)
    
    return papers, embeddings

def build_and_save_index(embeddings: List[np.ndarray], save_path: str) -> None:
    """Build and save Faiss index"""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)
    
    # Save index
    faiss.write_index(index, save_path)

def main():
    input_json = "arxiv-metadata-oai-snapshot.json"  # arxiv metadata json file
    output_dir = "processed_data"  # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process data
    papers, embeddings = process_papers(input_json)
    
    # Save processed data
    print("Saving processed data...")
    with open(f"{output_dir}/papers.pkl", 'wb') as f:
        pickle.dump(papers, f)
    
    # Build and save index
    print("Building and saving Faiss index...")
    build_and_save_index(embeddings, f"{output_dir}/faiss.index")
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main() 