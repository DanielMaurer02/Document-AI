import os

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from .embedding.embeddings import Embedding_Service, EmbeddingProvider
from .add_documents import add_documents_to_chromadb
from .query_llm import invoke_query
import time
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
EMBEDDING_SERVICE = os.getenv('EMBEDDING_SERVICE', 'alibaba')

class DocumentAI:
    """A Document AI class for managing vector databases and document retrieval.
    
    This class provides functionality to connect to ChromaDB, manage document collections,
    add documents, and perform queries using vector similarity search.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "rag"):
        start_time = time.time()
        self.persistent_client = chromadb.HttpClient(host, port,settings=Settings(anonymized_telemetry=False))
        self.vectorstore, self.embeddings = self.__get_vectorstore(collection_name)
        logging.info(f"DocumentAI initialized in {time.time() - start_time:.2f} seconds")

    def __get_vectorstore(self, collection_name: str) -> tuple[Chroma, Embeddings]:
        """Create and configure a ChromaDB vectorstore with embeddings.
        
        Args:
            collection_name (str): The name of the collection to create or connect to.
            
        Returns:
            tuple[Chroma, HuggingFaceEmbeddings]: A tuple containing the vectorstore instance
                and the embeddings model.
        """
        provider_service = EmbeddingProvider(EMBEDDING_SERVICE.lower())
        embedding_service = Embedding_Service(provider_service)
        embeddings = embedding_service.get_embeddings()
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=self.persistent_client,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return vectorstore, embeddings

    def delete_collection(self, collection_name: str = "rag"):
        """Delete a collection from ChromaDB.
        
        Args:
            collection_name (str): The name of the collection to delete. Defaults to "rag".
        """
        self.persistent_client.delete_collection(collection_name)

    #TODO: Recognize duplicates, in a first step by hashing the content and in a second step by setting a threshold for similarity
    def add_documents(self, file_path: str | list[str]) -> None:
        """Add documents to the vectorstore.
        
        Args:
            file_path (str | list[str]): Path to a single document or list of paths to multiple documents.
        """
        start_time = time.time()
        add_documents_to_chromadb(file_path, self.vectorstore)
        logging.info(f"Documents added in {time.time() - start_time:.2f} seconds")


    # TODO: Save answers from llm for session
    def query(self, query: str) -> str:
        """Execute a query against the vectorstore and return the result.
        
        Args:
            query (str): The query string to search for in the documents.
            
        Returns:
            str: The response from the LLM based on the retrieved documents.
        """
        start_time = time.time()
        result = invoke_query(query, self.vectorstore, self.embeddings)
        logging.info(f"Query execution time: {time.time() - start_time:.2f} seconds")
        return result


#file_path = [r"c:\Users\danie\Downloads\20240930 DHBW Zeugnis Daniel Maurer .pdf",r"c:\Users\danie\Downloads\MV blanko 1.OG rechts.pdf"]
#host = os.getenv("CHROMA_HOST", "localhost")
#doc_ai = DocumentAI(host=host)
#doc_ai.delete_collection("rag")  # Clear the collection before adding new documents
#doc_ai.add_documents(file_path)
#while True:
#    query = input("Enter your query (or 'exit' to quit): ")
#    if query.lower() == 'exit':
#        break
#    result = doc_ai.query(query)
#    print(f"Query Result: {result}")