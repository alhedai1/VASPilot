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
        """Send the list of texts to the remote embedding API for processing"""
        payload = {
            "input": input,
            "model": self.model_id
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=300  # timeout in seconds (adjust as needed)
        )

        # Handle the API response
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")

        # Extract results based on the actual API response structure
        results = response.json()

        # Assumed response structure:
        # {
        #   "embeddings": [[...], [...], ...]
        # }
        # Update this if the actual structure differs
        embeddings = results.get("data")
        
        sorted_embeddings = sorted(
                embeddings, key=lambda e: e["index"]  # type: ignore
            )

        return cast(
                Embeddings, [result["embedding"] for result in sorted_embeddings]
            )
