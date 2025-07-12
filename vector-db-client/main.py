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
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "rag"):
        self.persistent_client = chromadb.HttpClient(host, port)
        self.vectorstore, self.embeddings = self.__get_vectorstore(collection_name)

    def __get_vectorstore(self, collection_name: str) -> tuple[Chroma, HuggingFaceEmbeddings]:
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
        self.persistent_client.delete_collection(collection_name)

    def add_documents(self, file_path: str | list[str]):
        add_documents_to_chromadb(file_path, self.vectorstore)

    def query(self, query: str) -> str:
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