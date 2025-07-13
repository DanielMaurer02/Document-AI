import requests
from typing import List
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', None)


class AlibabaDashScopeEmbeddings(Embeddings):
    def __init__(self, model_name:str = "text-embedding-v3",base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"):
        if DASHSCOPE_API_KEY is None:
            raise ValueError("API key must be provided for Alibaba Embeddings.")
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        max_batch_size = 10
        all_embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i+max_batch_size]

            payload = {
                "model": self.model_name,
                "input": batch,
                "dimensions": 1024,
                "encoding_format":"float"

            }
            response = requests.post(f"{self.base_url}/embeddings", json=payload, headers=self.headers)
            if response.status_code > 299:
                logging.error("Error: %s %s", response.status_code, response.text)
                response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item['embedding'] for item in data['data']])
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]