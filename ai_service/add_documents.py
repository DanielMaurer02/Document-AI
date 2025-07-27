from langchain_chroma import Chroma
from ai_service.document_conversion.convert_documents import convert_documents
from ai_service.utils.hash_file import blake2b_file
from ai_service.utils.existence_check import check_document_exists
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def add_documents_to_chromadb( #TODO: Progress indicator
    file_path: str | list[str], vectorstore: Chroma
) -> None:
    """Add documents to ChromaDB with metadata including file paths and duplicate detection.

    Args:
        file_path (str | list[str]): Path to a single file or list of paths to multiple files
            to convert and add to the vectorstore.
        vectorstore (Chroma): ChromaDB vectorstore instance where documents will be added.
    """
    if isinstance(file_path, str):
        file_path_array = [file_path]
    else:
        file_path_array = file_path

    # Filter out duplicates
    file_paths = []
    skipped_count = 0

    for fp in file_path_array:
        if check_document_exists(fp, vectorstore):
            logging.info(f"Skipping duplicate document: {Path(fp).name}")
            skipped_count += 1
        else:
            file_paths.append(fp)

    if not file_paths:
        logging.info(
            f"All {len(file_path_array)} documents already exist in the vectorstore"
        )
        return

    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} duplicate documents")

    # Convert only new documents
    documents = convert_documents(file_paths)

    # Calculate hashes for metadata
    file_hashes = {fp: blake2b_file(fp) for fp in file_paths}

    # Extract texts and metadata from Document objects
    texts = [doc.page_content for doc in documents]
    metadatas = []

    for doc in documents:
        metadata = doc.metadata.copy()
        # Add file hash to metadata for future duplicate detection
        source_file = metadata.get("source", "")
        if source_file in file_hashes:
            metadata["file_hash"] = file_hashes[source_file]
        metadatas.append(metadata)

    # Add documents with metadata to the vectorstore
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
    )

    logging.info(
        f"Added {len(documents)} document chunks from {len(file_paths)} new files to the vectorstore"
    )

    logging.info("Temporary files removed after processing")