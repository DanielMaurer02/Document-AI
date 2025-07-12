from langchain_chroma import Chroma
from document_conversion.convert_documents import convert_documents



def add_documents_to_chromadb(file_path: str | list[str], vectorstore: Chroma) -> None:
    """Add documents to ChromaDB with metadata including file paths.
    
    Args:
        file_path (str | list[str]): Path to a single file or list of paths to multiple files
            to convert and add to the vectorstore.
        vectorstore (Chroma): ChromaDB vectorstore instance where documents will be added.
    """
    documents = convert_documents(file_path)
    
    # Extract texts and metadata from Document objects
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Add documents with metadata to the vectorstore
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
    )
    
    print(f"Added {len(documents)} document chunks with metadata to the vectorstore")
