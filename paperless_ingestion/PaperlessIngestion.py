import requests
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from time import sleep

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




class PaperlessIngestion:

    def __init__(self):
        """Initialize the Paperless Ingestion class."""
        # Track downloaded files for cleanup
        self.downloaded_files: List[str] = []
        self.failures: List[str] = []
        self.paperless_api_url = os.getenv("PAPERLESS_API_URL", "")
        if not self.paperless_api_url:
            logger.error("PAPERLESS_API_URL environment variable is not set.")
            raise ValueError("PAPERLESS_API_URL environment variable is required.")
        self.paperless_token = os.getenv("PAPERLESS_API_TOKEN", "")
        if not self.paperless_token:
            logger.error("PAPERLESS_API_TOKEN environment variable is not set.")
            raise ValueError("PAPERLESS_API_TOKEN environment variable is required.")

        self.temp_dir = os.path.join(os.getcwd(), "paperless_temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {self.temp_dir}")


    def cleanup_downloaded_files(self):
        """Clean up all downloaded temporary files."""
        for file_path in self.downloaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except OSError as e:
                logger.error(f"Error removing file {file_path}: {e}")
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
            except OSError as e:
                logger.error(f"Error removing temporary directory {self.temp_dir}: {e}")
        self.downloaded_files.clear()


    def get_all_documents(self) -> List[Dict[Any, Any]]:
        """Fetch all documents from the Paperless API."""
        headers = {"Authorization": f"Token {self.paperless_token}"}
        docs = []
        url = self.paperless_api_url

        logger.info("Fetching documents from Paperless API...")
        while url:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()  # Raise an exception for bad status codes
            data = resp.json()
            docs.extend(data["results"])
            url = data["next"]
        
        logger.info(f"Retrieved {len(docs)} documents from API")
        return docs

    def _download_document(self, document_id: int, filename: str, temp_dir: str) -> None:
        """Download a document temporarily to local storage."""
        headers = {"Authorization": f"Token {self.paperless_token}"}
        download_url = f"{self.paperless_api_url}/{document_id}/download/"
        logger.info(f"Downloading document {document_id} from {download_url}")
        try:
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Create safe filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            if not safe_filename:
                safe_filename = f"document_{document_id}"
            
            # Ensure we have a file extension
            if not Path(safe_filename).suffix:
                safe_filename += ".pdf"  # Default to PDF
            
            temp_file_path = os.path.join(temp_dir, safe_filename)
            
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self.downloaded_files.append(temp_file_path)
            logger.info(f"Downloaded: {safe_filename}")
            
        except requests.RequestException as e:
            logger.error(f"Error downloading document {document_id}: {e}")
            self.failures.append(download_url)

    def download_all_documents(self) -> List[str] | None:
        """Download all documents to a temporary directory."""
        docs = self.get_all_documents()
        
        for document in docs:
            document_id = document['id']
            title = document['title']
            
            logger.info(f"Processing document {document_id}: {title}")
            
            # Download document temporarily
            self._download_document(document_id, title, self.temp_dir)

        logger.info(f"Successfully downloaded {len(self.downloaded_files)} documents")
        if self.failures:
            logger.warning(f"Failed to download {len(self.failures)} documents: {self.failures}")
        return self.downloaded_files

    def download_specific_document(self, document_id: int, title: str) -> List[str] | None:
        """Download a specific document to a temporary directory."""
        self._download_document(document_id, title, self.temp_dir)
        return self.downloaded_files

