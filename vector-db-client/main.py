import  os

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
        start_time = time.time()
        self.persistent_client = chromadb.HttpClient(host, port)
        self.vectorstore, self.embeddings = self.__get_vectorstore(collection_name)
        print(f"DocumentAI initialized in {time.time() - start_time:.2f} seconds")

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
        start_time = time.time()
        add_documents_to_chromadb(file_path, self.vectorstore)
        print(f"Documents added in {time.time() - start_time:.2f} seconds")

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


file_path = [r"c:\Users\danie\Downloads\MV blanko 1.OG rechts.pdf",r"c:\Users\danie\Downloads\23-10-21-ETV_DB_Systel_-_unterzeichnet.pdf"]
host = os.getenv("CHROMA_HOST", "localhost")
doc_ai = DocumentAI(host=host)
#doc_ai.add_documents(file_path)
query = "Kannst du in meinem Gewerkschaftsvertrag der EVG nachsehen,  wie viel Geld ich erhalte, wenn ich von Tarifgruppe 4 in Tarifgruppe 5 aufsteige? Ich arbeite 100% also nur mit den 30 Urlaubstagen."

result = doc_ai.query(query)
print(f"Query Result: {result}")