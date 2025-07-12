from enum import Enum
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", None)


class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    ALIBABA = "alibaba"

class Embedding_Service:
    def __init__(self, embedding_provider: EmbeddingProvider = EmbeddingProvider.ALIBABA):
        self.embedding_provider = embedding_provider

    def get_embeddings(self) -> Embeddings:
        """Create and return an embedding service instance based on the provider."""
        if self.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct" if EMBEDDING_MODEL_NAME is None else EMBEDDING_MODEL_NAME)
        elif self.embedding_provider == EmbeddingProvider.ALIBABA:
            from embedding.alibaba import AlibabaDashScopeEmbeddings
            return AlibabaDashScopeEmbeddings(model_name="text-embedding-v3" if EMBEDDING_MODEL_NAME is None else EMBEDDING_MODEL_NAME)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")