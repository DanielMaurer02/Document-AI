from langchain_chroma import Chroma
from document_conversion.main import convert_document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



def add_documents_to_chromadb(file_path: str | list[str], vectorstore: Chroma):
    """
    Add documents to ChromaDB with metadata including file paths.
    
    Args:
        file_path: Path to the file(s) to convert and add
        vectorstore: ChromaDB vectorstore instance
    
    Returns:
        Chroma: Updated vectorstore
    """
    documents = convert_document(file_path)
    
    # Extract texts and metadata from Document objects
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Add documents with metadata to the vectorstore
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
    )
    
    print(f"Added {len(documents)} document chunks with metadata to the vectorstore")
    return vectorstore
