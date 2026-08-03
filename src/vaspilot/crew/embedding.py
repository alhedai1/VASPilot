import requests
from crewai.rag.core.types import Documents, Embeddings
from crewai.rag.embeddings.providers.custom.embedding_callable import CustomEmbeddingFunction
from typing import cast
from urllib.parse import urljoin

class LocalAPIEmbedder(CustomEmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "local_api_embedder"
    
    def __init__(self, url: str = "http://172.16.8.24:8003/v1/", 
                 model_id: str = "BAAI/bge-m3",
                 api_key: str = "EMPTY"):
        self.url = urljoin(url, "embeddings")
        self.model_id = model_id
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def __call__(self, input: Documents) -> Embeddings:
        """将文本列表发送到远程 embedding API 进行处理"""
        payload = {
            "input": input,
            "model": self.model_id
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=300  # 设置超时时间（根据实际情况调整）
        )

        # 处理 API 响应
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.text}")

        # 根据实际API返回结构提取结果
        results = response.json()
        
        # 假设返回的数据结构：
        # {
        #   "embeddings": [[...], [...], ...]
        # }
        # 如果实际结构不同需要在此修改
        embeddings = results.get("data")
        
        sorted_embeddings = sorted(
                embeddings, key=lambda e: e["index"]  # type: ignore
            )

        return cast(
                Embeddings, [result["embedding"] for result in sorted_embeddings]
            )
