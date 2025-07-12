from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter

class DoclingDocLoader(BaseLoader):
    """A document loader using the Docling library.

    This loader converts documents to markdown format using Docling's
    document converter and creates LangChain Document objects with metadata.
    """

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        """Lazily load documents and convert them to LangChain Document objects.

        Yields:
            LCDocument: LangChain Document objects containing the markdown content
                of the documents along with metadata including source file path.
        """
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            metadata = {
                "source": source,
                "file_path": source,
            }
            yield LCDocument(page_content=text, metadata=metadata)