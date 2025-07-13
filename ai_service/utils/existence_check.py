from langchain_chroma import Chroma
from .hash_file import blake2b_file
import logging

logging.basicConfig(level=logging.INFO)


def check_document_exists(file_path: str, vectorstore: Chroma) -> bool:
    """Check if a document already exists in the vectorstore by its hash.

    Args:
        file_path (str): Path to the file to check
        vectorstore (Chroma): ChromaDB vectorstore instance

    Returns:
        bool: True if document already exists, False otherwise
    """
    file_hash = blake2b_file(file_path)

    # Query ChromaDB for any documents with this file hash
    try:
        results = vectorstore.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0
    except Exception as e:
        logging.warning(f"Error checking for duplicate document: {e}")
        return False
