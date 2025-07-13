# Document AI

A powerful document AI system that combines ChromaDB vector database with advanced language models to provide intelligent document retrieval and question-answering capabilities. The system uses Docling for document conversion, Alibaba embeddings for semantic search, and Qwen LLMs for generating responses.

## Features

- **Document Processing**: Supports PDF documents using Docling for high-quality text extraction and conversion to markdown
- **Vector Search**: Utilizes ChromaDB for efficient similarity search with multilingual embeddings
- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with LLM responses for accurate, context-aware answers
- **Web Interface**: Integrated Open WebUI for easy interaction with the document AI system
- **API Backend**: FastAPI-based OpenAI-compatible API for seamless integration
- **Document Compression**: Uses LLM-based reranking for improved retrieval quality
- **Source Attribution**: Provides source file paths and chunk information for transparency
- **Multilingual Support**: Uses `text-embedding-v3` embeddings for multiple languages

## Architecture

The system consists of the following services:
- **ChromaDB Server**: Vector database for document embeddings
- **Document AI API**: FastAPI backend providing OpenAI-compatible API (pre-built Docker image)
- **Open WebUI**: Modern web interface for chatting with your documents

## Prerequisites

- Docker and Docker Compose
- API key for Alibaba DashScope (default configuration)

## Quick Start

The entire system can now be started with a single command! No manual environment configuration needed.

### Start the Complete System

#### Step 1: Setup Configuration
First, copy the user configuration file to create your docker-compose.yml:

```bash
cp docker-compose.user.yml docker-compose.yml
```

#### Step 2: Start Services
```bash
docker-compose up -d
```

This will start:
- ChromaDB server on port 8000
- Document AI API on port 8008
- Open WebUI on port 3000

#### Step 3: Configure WebUI Connection
1. Open http://localhost:3000 in your browser
2. In the Open WebUI interface, go to **Settings** (gear icon)
3. Navigate to **Connections** or **External Connections**
4. Add a new OpenAI API connection with:
   - **API Base URL**: `http://localhost:8008`
   - **API Key**: `not-required` (or leave empty)
5. Save the configuration

### Access the Application

Once all services are running and configured, you can:
- **Chat with your documents**: Use the web interface at http://localhost:3000
- **API endpoint**: The OpenAI-compatible API is available at http://localhost:8008

### Configure API Keys

The system is pre-configured to use Alibaba DashScope. You need to update the API key in the docker-compose.user.yml file before copying it:

```yaml
environment:
  - EMBEDDING_SERVICE=alibaba
  - LLM_SERVICE=qwen
  - DASHSCOPE_API_KEY=your_api_key_here  # Replace with your actual API key
  - EMBEDDING_MODEL_NAME=text-embedding-v3
  - LLM_MODEL_NAME=qwen3-32b
  - DOMAIN=http://localhost:3000
```

**To get your API key:**
1. Sign up at [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Replace `KEY_DASHSCOPE` in docker-compose.user.yml with your actual key
5. Copy the file to docker-compose.yml as shown in the setup steps above


### Alternative Configuration Options

If you want to use different LLM or embedding providers, you can modify the environment variables in the docker-compose.yml file:

#### Available Service Providers

**Embedding Services:**
- `alibaba` (default) - Uses Alibaba's text-embedding-v3
- `huggingface` - Uses HuggingFace models

**LLM Services:**
- `qwen` (default) - Alibaba's Qwen models
- `groq` - Groq's fast inference

#### Example Alternative Configurations

**Configuration 1: HuggingFace + Groq**
```yaml
environment:
  - EMBEDDING_SERVICE=huggingface
  - LLM_SERVICE=groq
  - EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large-instruct
  - LLM_MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
  - HUGGINGFACE_API_KEY=your_huggingface_token_here
  - GROQ_API_KEY=your_groq_api_key_here
  - DOMAIN=http://localhost:3000
```

**Available Models:**
- **Groq LLM Models**: `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Alibaba LLM Models**: `qwen3-32b`
- **HuggingFace Embedding Models**: `intfloat/multilingual-e5-large-instruct`
- **Alibaba Embedding Models**: `text-embedding-v3`

## Usage

### Using the Web Interface

1. Open http://localhost:3000 in your browser
2. You'll see the Open WebUI interface
3. Start chatting with your documents immediately
4. The system will search through your document collection and provide context-aware responses

### Adding Documents

There are several ways to add documents to the system:

#### Duplicate Detection

The system now automatically detects duplicate documents using BLAKE2b file hashing before processing and embedding. This prevents:
- Wasting computational resources on re-processing the same files
- Creating duplicate embeddings that could skew search results
- Unnecessary storage usage

**How it works:**
1. Before processing any document, the system calculates a BLAKE2b hash of the file
2. It checks ChromaDB metadata to see if any document with that hash already exists
3. If found, the document is skipped with a log message
4. If not found, the document is processed and the hash is stored in the metadata for future checks

**Force Re-adding:** If you need to re-process a document (e.g., after changing embedding models), you can use the `force_readd=True` parameter.

#### Method 1: Using the Python API (for developers)

The main interface is the `DocumentAI` class in `ai_service/main.py`:

```python
from ai_service.main import DocumentAI

# Initialize the Document AI system
doc_ai = DocumentAI(host="server")  # "server" is the ChromaDB service name

# Add documents to the vector database (with automatic duplicate detection)
file_paths = [
    "/path/to/your/document1.pdf",
    "/path/to/your/document2.pdf"
]
doc_ai.add_documents(file_paths)

# Force re-add documents even if they already exist
doc_ai.add_documents(file_paths, force_readd=True)

# Query the documents
query = "What information can you find about...?"
result = doc_ai.query(query)
print(f"Query Result: {result}")
```

#### Method 2: Direct File Upload

You can copy PDF files directly into the running container and process them:

```bash
# Copy documents to the container
docker cp /path/to/your/document.pdf document-ai-document-ai-api-1:/app/documents/

# Execute a Python script to add documents
docker exec -it document-ai-document-ai-api-1 python -c "
from ai_service.main import DocumentAI
doc_ai = DocumentAI(host='server')
doc_ai.add_documents(['/app/documents/document.pdf'])
"
```

### DocumentAI Class Methods

#### `__init__(host="localhost", port=8008, collection_name="rag")`
Initialize the DocumentAI instance with ChromaDB connection parameters.

#### `add_documents(file_path: str | list[str], force_readd: bool = False)`
Add one or more PDF documents to the vector database with automatic duplicate detection. Documents are:
1. Checked for duplicates using BLAKE2b file hashing
2. Skipped if already processed (unless `force_readd=True`)
3. Converted to markdown using Docling
4. Split into semantic chunks
5. Embedded using multilingual embeddings
6. Stored in ChromaDB with metadata including file hash


#### `query(query: str) -> str`
Execute a query against the document collection:
1. Performs similarity search to find relevant document chunks
2. Uses LLM-based reranking for improved relevance
3. Generates a response using the LLM with retrieved context
4. Returns formatted answer with source attribution

#### `delete_collection(collection_name="rag")`
Delete a collection from ChromaDB (useful for cleanup or reset).

## API Endpoints

The FastAPI backend provides an OpenAI-compatible API with automatic interactive documentation:

- **Interactive API Documentation**: Available at http://localhost:8008/docs
- **OpenAPI Schema**: Available at http://localhost:8008/openapi.json

### Chat Completions
```bash
curl -X POST http://localhost:8008/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the main topics in the documents?"}
    ]
  }'
```

### List Models
```bash
curl http://localhost:8008/models
```

## Configuration

### Customizing the System

You can customize various aspects by modifying the docker-compose.yml file:

#### Port Configuration
```yaml
ports:
  - "3000:8080"  # Change 3000 to your preferred port for Web UI
  - "8008:8008"  # Change first 8008 to your preferred API port
  - "8000:8000"  # Change first 8000 to your preferred ChromaDB port
```

#### Environment Variables
The Document AI API supports the following environment variables:

**Core Configuration:**
- `EMBEDDING_SERVICE`: Embedding service provider (`alibaba`, `huggingface`)
- `LLM_SERVICE`: Language model service (`qwen`, `groq`)
- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use
- `LLM_MODEL_NAME`: Name of the LLM model to use
- `DOMAIN`: Domain URL for the application (e.g., `http://localhost:3000`)

**Database Configuration:**
- `CHROMA_HOST`: ChromaDB server hostname (default: `chroma`)
- `CHROMA_PORT`: ChromaDB server port (default: `8008`)

**API Keys:**
- `DASHSCOPE_API_KEY`: API key for Alibaba DashScope services
- `HUGGINGFACE_API_KEY`: API key for HuggingFace services
- `GROQ_API_KEY`: API key for Groq services

#### Model Configuration
The system uses pre-built Docker images for easy deployment:

```yaml
# Document AI API service configuration
document-ai-api:
  image: danitherex/document-ai-api:latest  # Pre-built Docker image
  environment:
    - EMBEDDING_MODEL_NAME=text-embedding-v3
    - LLM_MODEL_NAME=qwen3-32b
    # Change these to use different models
```

#### ChromaDB Configuration
The vector database uses cosine similarity for embeddings and is automatically configured for optimal performance.

### Custom Prompt Templates

You can customize the prompt template by modifying `ai_service/query_llm.py`. The current prompt template is optimized for document question-answering with source attribution.

## Project Structure

```
document-ai/
├── README.md
├── docker-compose.yml          # Complete system orchestration
├── docker-compose.user.yml     # User configuration template
├── Dockerfile                  # Document AI API container build (for development)
├── app.py                     # FastAPI backend (OpenAI-compatible API)
├── pyproject.toml             # Python dependencies
├── ai_service/
│   ├── main.py                # Main DocumentAI class
│   ├── add_documents.py       # Document addition logic
│   ├── query_llm.py          # Query processing and LLM interaction
│   ├── document_conversion/
│   │   ├── docling_pdf_loader.py  # PDF loading with Docling
│   │   ├── document_splitter.py   # Text chunking
│   │   └── convert_documents.py   # Document conversion pipeline
│   ├── embedding/
│   │   ├── alibaba.py         # Alibaba embedding service
│   │   └── embeddings.py      # Embedding abstraction
│   ├── llm/
│   │   └── model.py          # LLM configuration
│   └── utils/
│       └── thinking_animation.py
```

**Note**: The system now uses a pre-built Docker image (`danitherex/document-ai-api:latest`) for the API service, making deployment faster and more reliable. The Dockerfile is included for development purposes if you need to build custom versions.

## Logs

View real-time logs for all services:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f document-ai-api
docker-compose logs -f chroma  # ChromaDB
docker-compose logs -f openwebui
```

## Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   docker-compose down
   docker-compose up -d --build --force-recreate
   ```

2. **API Key Issues**:
   - Verify your API key is correctly set in docker-compose.yml
   - Check that the API key has necessary permissions
   - Restart the services after changing the key

3. **Memory Issues with Large Documents**:
   - Monitor container resources: `docker stats`
   - Consider processing documents in smaller batches

4. **Connection Issues**:
   - Check if all ports are available (3000, 8000, 8008)
   - Verify no other services are using these ports

### Health Checks

Check service status:
```bash
# Check all services
docker-compose ps

# Check specific service health
curl http://localhost:8008/models  # API health
curl http://localhost:8008/api/v1/heartbeat  # ChromaDB health
```

### Performance Optimization

For better performance with large document collections:
1. Increase ChromaDB memory allocation in docker-compose.yml
2. Adjust retrieval parameters in the code
3. Consider using SSD storage for ChromaDB data persistence

## Development

### Local Development Setup

The system now uses pre-built Docker images for production deployment. For development:

1. Clone the repository
2. For quick testing, use the pre-built image (default configuration)
3. For custom development, modify the code and rebuild:
   ```bash
   # Build custom image locally
   docker-compose build
   docker-compose up -d
   ```

#### Running the FastAPI Development Server Locally

For local development and testing of the API:

```bash
# Install dependencies
uv sync

# Install pre-commit hooks for code quality
uv run pre-commit install

# Run the FastAPI development server with auto-reload
uv run app.py
```

The FastAPI server will start with:
- Automatic reload on code changes
- Interactive API documentation at http://localhost:8008/docs
- OpenAPI schema at http://localhost:8008/openapi.json

### Building Custom Images

If you need to modify the API service, you can build a custom image:

```bash
# Build the image locally
docker build -t custom-document-ai-api .

# Update docker-compose.yml to use your custom image
# Change: image: danitherex/document-ai-api:latest
# To: image: custom-document-ai-api
```

### Adding New Features

The modular architecture makes it easy to:
- Add new document types (modify `document_conversion/`)
- Integrate new LLM providers (modify `llm/`)
- Add new embedding services (modify `embedding/`)
- Customize the API (modify `app.py`)

## License

This project is open source. Feel free to contribute and improve the system!