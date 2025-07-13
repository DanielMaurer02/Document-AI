from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document as LCDocument
from typing import Optional

def split_markdown_text(doc_text: str, metadata: Optional[dict] = None) -> list[LCDocument]:
    """Split markdown text into chunks and create LangChain Document objects.
    
    Args:
        doc_text (str): The markdown text to be split into chunks.
        metadata (Optional[dict]): Optional metadata dictionary to be included
            with each document chunk. Defaults to None.
            
    Returns:
        list[LCDocument]: A list of LangChain Document objects, each containing
            a text chunk and associated metadata including chunk index and ID.
    """
    mD_splitter = MarkdownTextSplitter()
    md_split = mD_splitter.split_text(doc_text)
    
    documents = []
    for i, chunk in enumerate(md_split):
        print(f"Chunk {i+1}:")
        print(f"Total Number of Characters: {len(chunk)}")
        print(f"Total Number of Words: {len(chunk.split(' '))}")
        print(f"Total Number of Tokens: {len(chunk.split()) * (4/3)}")
        print("------------------------------------------------")
        
        # Create metadata for each chunk
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_id"] = f"{metadata.get('file_path', 'unknown')}_chunk_{i}" if metadata else f"chunk_{i}"
        
        documents.append(LCDocument(page_content=chunk, metadata=chunk_metadata))
    
    return documents
