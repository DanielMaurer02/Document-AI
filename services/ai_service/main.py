import os
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

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

    def reset_collection(self, collection_name: str = "rag"):
        """Reset a collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection to reset. Defaults to "rag".
        """
        self.persistent_client.delete_collection(collection_name)
        self.vectorstore = self.__get_vectorstore(collection_name)
        logging.info(f"Collection '{collection_name}' reset successfully.")

    def _split_files_into_batches(self, file_path: dict[str, str], num_workers: int = 3) -> list[dict[str, str]]:
        """Split the file dictionary into batches for parallel processing.

        Args:
            file_path (dict[str, str]): Dictionary of files to process
            num_workers (int): Number of worker batches to create

        Returns:
            list[dict[str, str]]: List of file dictionaries, one per worker
        """
        items = list(file_path.items())
        batch_size = math.ceil(len(items) / num_workers)
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = dict(items[i:i + batch_size])
            if batch:  # Only add non-empty batches
                batches.append(batch)
        
        return batches

    def _process_batch(self, batch: dict[str, str]) -> tuple[bool, str]:
        """Process a batch of files in a single worker.

        Args:
            batch (dict[str, str]): Dictionary of files to process

        Returns:
            tuple[bool, str]: Success status and message
        """
        try:
            add_documents_to_chromadb(batch, self.vectorstore)
            return True, f"Successfully processed {len(batch)} files"
        except Exception as e:
            return False, f"Error processing batch: {str(e)}"


    # TODO: Add similarity-based duplicate detection as a second step (hash-based detection implemented)
    def add_documents(
        self, file_path: dict[str, str], force_readd: bool = False, num_workers: int = 3
    ) -> None:
        """Add documents to the vectorstore with duplicate detection, supporting parallel processing for multiple files.

        Args:
            file_path (dict[str, str]): Dictionary where key is the file path and value is the file URL on the server.
            force_readd (bool): If True, will re-add documents even if they already exist. Defaults to False.
        """

        start_time = time.time()

    
        if force_readd:
            self._remove_duplicate_documents(list(file_path.keys()))

        # Split files into batches for parallel processing
        batches = self._split_files_into_batches(file_path, num_workers)


        if len(batches) == 1:
            # If only one batch, process directly without threading overhead
            success, message = self._process_batch(batches[0])
            if success:
                logging.info(message)
            else:
                logging.error(message)
        else:
            # Process batches in parallel
            successful_batches = 0
            failed_batches = 0
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all batches to the executor
                future_to_batch = {
                    executor.submit(self._process_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Process completed futures
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        success, message = future.result()
                        if success:
                            successful_batches += 1
                            logging.info(f"Worker {batch_index + 1}: {message}")
                        else:
                            failed_batches += 1
                            logging.error(f"Worker {batch_index + 1}: {message}")
                    except Exception as e:
                        failed_batches += 1
                        logging.error(f"Worker {batch_index + 1}: Unexpected error: {str(e)}")

            # Log final results
            total_files = len(file_path)
            logging.info(
                f"Parallel processing completed: {successful_batches} successful batches, "
                f"{failed_batches} failed batches, "
                f"{total_files} total files processed in {time.time() - start_time:.2f} seconds"
            )

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
