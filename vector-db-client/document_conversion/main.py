from document_conversion.docling_pdf_loader import DoclingDocLoader
from document_conversion.document_splitter import split_markdown_text

def convert_document(file_path: str | list[str]) -> list:
    """Convert document(s) to markdown chunks with metadata including file path.
    
    Args:
        file_path (str | list[str]): Path to a single document or list of paths to multiple documents.
        
    Returns:
        list: List of LangChain Document objects with content and metadata containing
            the markdown chunks from the processed documents.
    """
    loader = DoclingDocLoader(file_path=file_path)
    
    docs = loader.load()

    md_split = []
    
    for doc in docs:
        print(doc)
        # Extract metadata from the first document (includes file path info)
        doc_metadata = doc.metadata
        
        # Split the document and preserve metadata
        md_split.extend(split_markdown_text(doc.page_content, doc_metadata))

    return md_split
    
