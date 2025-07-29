import os
from typing import Iterator
from unittest import result

from dotenv import load_dotenv
import chromadb

from chromadb.config import Settings
from langchain_chroma import Chroma

from services.ai_service.embedding.embeddings import Embedding_Service, EmbeddingProvider
from services.ai_service.add_documents import add_documents_to_chromadb
from services.ai_service.query_llm import invoke_query, invoke_query_stream
from services.ai_service.utils.hash_file import blake2b_file
import time
import logging
import concurrent.futures


logging.basicConfig(level=logging.INFO)

load_dotenv()
EMBEDDING_SERVICE = os.getenv("EMBEDDING_SERVICE", "alibaba")


class DocumentAI:
    """A Document AI class for managing vector databases and document retrieval.

    This class provides functionality to connect to ChromaDB, manage document collections,
    add documents, and perform queries using vector similarity search.
    """

    def __init__(self, collection_name: str = "rag"):
        start_time = time.time()
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", 8000))
        self.persistent_client = chromadb.HttpClient(
            host, port, settings=Settings(anonymized_telemetry=False)
        )
        self.vectorstore = self.__get_vectorstore(collection_name)
        logging.info(
            f"DocumentAI initialized in {time.time() - start_time:.2f} seconds"
        )

    def __get_vectorstore(self, collection_name: str) -> Chroma:
        """Create and configure a ChromaDB vectorstore with embeddings.

        Args:
            collection_name (str): The name of the collection to create or connect to.

        Returns:
            Chroma: A vectorstore instance
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
        return vectorstore

    def delete_collection(self, collection_name: str = "rag"):
        """Delete a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to delete. Defaults to "rag".
        """
        self.persistent_client.delete_collection(collection_name)

    # TODO: Add similarity-based duplicate detection as a second step (hash-based detection implemented)
    def add_documents(
        self, file_path: str | list[str], force_readd: bool = False
    ) -> None:
        """Add documents to the vectorstore with duplicate detection, supporting parallel processing for multiple files.

        Args:
            file_path (str | list[str]): Path to a single document or list of paths to multiple documents.
            force_readd (bool): If True, will re-add documents even if they already exist. Defaults to False.
        """

        start_time = time.time()

        if isinstance(file_path, str):
            file_paths = [file_path]
        else:
            file_paths = file_path

        if force_readd:
            self._remove_duplicate_documents(file_paths)

        def process_file(fp):
            add_documents_to_chromadb(fp, self.vectorstore)

        if len(file_paths) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(process_file, file_paths))
        else:
            process_file(file_paths[0])

        logging.info(f"Documents processed in {time.time() - start_time:.2f} seconds")

    def _remove_duplicate_documents(self, file_path: str | list[str]) -> None:
        """Remove documents from vectorstore that have the same hash as the provided files.

        Args:
            file_path (str | list[str]): Path to a single document or list of paths to multiple documents.
        """

        if isinstance(file_path, str):
            file_paths = [file_path]
        else:
            file_paths = file_path

        for fp in file_paths:
            file_hash = blake2b_file(fp)
            try:
                # Find documents with this hash
                results = self.vectorstore.get(where={"file_hash": file_hash})
                if results["ids"]:
                    # Delete existing documents with this hash
                    self.vectorstore.delete(ids=results["ids"])
                    logging.info(
                        f"Removed {len(results['ids'])} existing chunks for file: {fp}"
                    )
            except Exception as e:
                logging.warning(f"Error removing duplicate document {fp}: {e}")

    def query(self, query: str) -> str:
        """Execute a query against the vectorstore and return the result.

        Args:
            query (str): The query string to search for in the documents.

        Returns:
            str: The response from the LLM based on the retrieved documents.
        """
        start_time = time.time()
        result = invoke_query(query, self.vectorstore)
        logging.info(f"Query execution time: {time.time() - start_time:.2f} seconds")
        return result

    def query_stream(self, query: str) -> Iterator[str]:
        """Execute a query against the vectorstore and return the result as a stream.

        Args:
            query (str): The query string to search for in the documents.

        Yields:
            str: Streaming chunks of the response from the LLM based on the retrieved documents.
        """
        start_time = time.time()
        try:
            yield from invoke_query_stream(query, self.vectorstore)
        finally:
            logging.info(
                f"Query streaming execution time: {time.time() - start_time:.2f} seconds"
            )
