import  os
from getpass import getpass

from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from add_documents import add_documents_to_chromadb
from query_llm import invoke_query
import time

load_dotenv()

class DocumentAI:
    """A Document AI class for managing vector databases and document retrieval.
    
    This class provides functionality to connect to ChromaDB, manage document collections,
    add documents, and perform queries using vector similarity search.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "rag"):
        self.persistent_client = chromadb.HttpClient(host, port)
        self.vectorstore, self.embeddings = self.__get_vectorstore(collection_name)

    def __get_vectorstore(self, collection_name: str) -> tuple[Chroma, HuggingFaceEmbeddings]:
        """Create and configure a ChromaDB vectorstore with embeddings.
        
        Args:
            collection_name (str): The name of the collection to create or connect to.
            
        Returns:
            tuple[Chroma, HuggingFaceEmbeddings]: A tuple containing the vectorstore instance
                and the embeddings model.
        """
        HF_EMBED_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
        embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)

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

    def add_documents(self, file_path: str | list[str]) -> None:
        """Add documents to the vectorstore.
        
        Args:
            file_path (str | list[str]): Path to a single document or list of paths to multiple documents.
        """
        add_documents_to_chromadb(file_path, self.vectorstore)

    def query(self, query: str) -> str:
        """Execute a query against the vectorstore and return the result.
        
        Args:
            query (str): The query string to search for in the documents.
            
        Returns:
            str: The response from the LLM based on the retrieved documents.
        """
        start_time = time.time()
        result = invoke_query(query, self.vectorstore, self.embeddings)
        print(f"Query execution time: {time.time() - start_time:.2f} seconds")
        return result


file_path = ["/Users/danie/Downloads/Rechnung-Herman-Miller-Chairgo.pdf","/Users/danie/Downloads/20240930 DHBW Zeugnis Daniel Maurer .pdf"]
doc_ai = DocumentAI()
#doc_ai.add_documents(file_path)
query = "Wie schwer darf ich maximal für meinen Bürostuhl werden?"

result = doc_ai.query(query)
print(f"Query Result: {result}")