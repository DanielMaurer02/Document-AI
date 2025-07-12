from chromadb import Documents,Embeddings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction


class MultilingualE5LargeInstructFunction(EmbeddingFunction):
    def __call__(self, input:Documents) -> Embeddings:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large-instruct"
        )
        embeddings = sentence_transformer_ef(input)
        return embeddings
    
    def __init__(self):
        super().__init__()


