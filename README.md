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
- **Paperless Integration**: Seamless integration with Paperless-ngx document management system
- **Parallel Processing**: Multi-threaded document processing for improved performance with multiple files
- **Automated Collection Management**: API endpoints for managing document collections and triggering bulk operations

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

### Deployment Options

The system supports two deployment modes:

1. **Remote API Mode (Default)**: Uses cloud-based APIs for LLM inference (Qwen, Groq)
2. **Local GPU Mode**: Runs a local GGUF model using GPU acceleration via llama-cpp-python

### Start the Complete System

#### Option 1: Remote API Mode (Recommended for most users)

**Step 1: Setup Configuration**
First, copy the user configuration file to create your docker-compose.yml:

```bash
cp docker-compose.user.yml docker-compose.yml
```

**Step 2: Configure API Keys**
Edit the docker-compose.yml file and replace the placeholder API keys with your actual keys:

```yaml
environment:
  - DASHSCOPE_API_KEY=your_actual_dashscope_key_here  # Replace with actual key
  - HUGGINGFACE_API_KEY=your_actual_huggingface_key_here  # Replace with actual key
  - PAPERLESS_API_URL=your_paperless_url_here  # Optional: for Paperless integration
  - PAPERLESS_API_TOKEN=your_paperless_token_here  # Optional: for Paperless integration
```

**Step 3: Start Remote Services**
```bash
docker-compose --profile remote up -d
```

#### Option 2: Local GPU Mode (For users with NVIDIA GPUs)

This mode runs a local Qwen 2.5 7B model using GPU acceleration. Requires an NVIDIA GPU with CUDA support.

**Prerequisites:**
- NVIDIA GPU with at least 8GB VRAM (recommended for Qwen 2.5 7B Q4 model)
- NVIDIA Docker runtime installed
- CUDA 12.5 compatible GPU driver

**Step 1: Setup Configuration**
```bash
cp docker-compose.user.yml docker-compose.yml
```

**Step 2: Configure HuggingFace API Key and Optional Paperless Integration**
Edit docker-compose.yml and set your HuggingFace API key (needed for embeddings) and optionally configure Paperless integration:

```yaml
environment:
  - HUGGINGFACE_API_KEY=your_actual_huggingface_key_here  # Replace with actual key
  - PAPERLESS_API_URL=your_paperless_url_here  # Optional: for Paperless integration
  - PAPERLESS_API_TOKEN=your_paperless_token_here  # Optional: for Paperless integration
```

**Step 3: Adjust CUDA Version (if needed)**
Check your CUDA version:
```bash
nvidia-smi
```

If you have a different CUDA version than 12.5, edit the `Dockerfile.local` and change:
```dockerfile
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS builder # Change to your CUDA version
# and
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04 # Change to your CUDA version
```

**Step 4: Start Local GPU Services**
```bash
docker-compose --profile local-gpu up -d
```

**Note**: The first run will download the Qwen 2.5 7B GGUF model (~4GB), which may take some time depending on your internet connection.

#### Services Overview

Both modes will start:
- **ChromaDB server** on port 8000
- **Document AI API** on port 8008 (using remote APIs or local GPU model)
- **Open WebUI** on port 3000

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

#### For Remote API Mode
The system supports multiple cloud-based API providers. Update the required keys in docker-compose.yml:

**Alibaba DashScope + HuggingFace Configuration:**
```yaml
environment:
  - EMBEDDING_SERVICE=alibaba
  - LLM_SERVICE=qwen_remote
  - DASHSCOPE_API_KEY=your_dashscope_key_here  # Replace with your actual API key
  - EMBEDDING_MODEL_NAME=text-embedding-v3
  - LLM_MODEL_NAME=qwen3-32b
  - DOMAIN=http://localhost:3000
```

**HuggingFace + Qwen Configuration:**
```yaml
environment:
  - EMBEDDING_SERVICE=huggingface
  - LLM_SERVICE=qwen_remote
  - DASHSCOPE_API_KEY=your_dashscope_key_here
  - HUGGINGFACE_API_KEY=your_huggingface_key_here
  - EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
  - LLM_MODEL_NAME=qwen3-32b
  - DOMAIN=http://localhost:3000
```

#### For Local GPU Mode
Only requires HuggingFace API key for embeddings:
```yaml
environment:
  - EMBEDDING_SERVICE=huggingface
  - LLM_SERVICE=qwen_local  # Uses local GGUF model
  - HUGGINGFACE_API_KEY=your_huggingface_key_here
  - EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
  - DOMAIN=http://localhost:3000
```

**To get your API keys:**
- **Alibaba DashScope**: Sign up at [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
- **HuggingFace**: Get your token at [HuggingFace Tokens](https://huggingface.co/settings/tokens)
- **Groq**: Get your API key at [Groq Console](https://console.groq.com)
- **Paperless-ngx**: Configure your Paperless instance URL and generate an API token from your Paperless settings


### Alternative Configuration Options

If you want to use different LLM or embedding providers, you can modify the environment variables in the docker-compose.yml file:

#### Available Service Providers

**Embedding Services:**
- `alibaba` - Uses Alibaba's text-embedding-v3
- `huggingface` - Uses HuggingFace models

**LLM Services:**
- `qwen_remote` - Alibaba's Qwen models (cloud API)
- `qwen_local` - Local Qwen GGUF model with GPU acceleration
- `groq` - Groq's fast inference

#### Example Alternative Configurations

**Configuration 1: HuggingFace + Groq (Remote)**
```yaml
environment:
  - EMBEDDING_SERVICE=huggingface
  - LLM_SERVICE=groq
  - EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
  - LLM_MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
  - HUGGINGFACE_API_KEY=your_huggingface_token_here
  - GROQ_API_KEY=your_groq_api_key_here
  - DOMAIN=http://localhost:3000
```

**Configuration 2: Local GPU with HuggingFace Embeddings**
```yaml
environment:
  - EMBEDDING_SERVICE=huggingface
  - LLM_SERVICE=qwen_local
  - EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
  - HUGGINGFACE_API_KEY=your_huggingface_token_here
  - DOMAIN=http://localhost:3000
```

**Available Models:**
- **Remote LLM Models**: 
  - Groq: `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct`
  - Alibaba: `qwen3-32b`
- **Local LLM Models**: 
  - `Qwen/Qwen2.5-7B-Instruct-GGUF_Q4` (7B parameters, Q4 quantization)
- **Embedding Models**: 
  - HuggingFace: `Qwen/Qwen3-Embedding-0.6B`
  - Alibaba: `text-embedding-v3`

#### GPU Requirements for Local Mode

**Minimum Requirements:**
- NVIDIA GPU with CUDA support
- 8GB+ VRAM (for Qwen 2.5 7B Q4 model)
- CUDA 12.5 compatible drivers (or adjust Dockerfile.local for your version)

**Performance Notes:**
- Q4 quantization provides good quality with reduced memory usage
- Inference speed depends on GPU memory bandwidth
- First startup downloads ~4GB GGUF model file

## Usage

### Using the Web Interface

1. Open http://localhost:3000 in your browser
2. You'll see the Open WebUI interface
3. Start chatting with your documents immediately
4. The system will search through your document collection and provide context-aware responses

### Adding Documents

There are several ways to add documents to the system:

#### Method 1: Paperless-ngx Integration (Recommended for automated workflows)

The system now includes seamless integration with Paperless-ngx document management systems. This allows you to automatically sync and process documents from your existing Paperless instance.

**Setup Paperless Integration:**
1. Configure your Paperless instance URL and API token in docker-compose.yml:
   ```yaml
   environment:
     - PAPERLESS_API_URL=https://your-paperless-instance.com/api/documents/
     - PAPERLESS_API_TOKEN=your_paperless_api_token_here
   ```

2. Use the API endpoint to trigger a full sync:
   ```bash
   curl -X POST http://localhost:8008/paperless/full_load
   ```

**Features:**
- **Automatic Download**: Downloads all documents from your Paperless instance to a temporary directory
- **Parallel Processing**: Processes multiple documents concurrently for improved performance
- **Automatic Cleanup**: Removes temporary files after processing
- **Background Processing**: Runs in a separate thread to avoid blocking the API
- **Error Handling**: Comprehensive logging and error tracking for failed downloads

#### Method 2: Using the Python API (for developers)

The main interface is the `DocumentAI` class in `ai_service/main.py`:

```python
from ai_service.main import DocumentAI

# Initialize the Document AI system
doc_ai = DocumentAI(collection_name="rag")  # ChromaDB connection automatically configured

# Add documents to the vector database (with automatic duplicate detection and parallel processing)
file_paths = [
    "/path/to/your/document1.pdf",
    "/path/to/your/document2.pdf",
    "/path/to/your/document3.pdf"
]
doc_ai.add_documents(file_paths)  # Multiple files processed in parallel automatically

# Force re-add documents even if they already exist
doc_ai.add_documents(file_paths, force_readd=True)

# Query the documents
query = "What information can you find about...?"
result = doc_ai.query(query)
print(f"Query Result: {result}")
```

#### Method 3: Direct File Upload

You can copy PDF files directly into the running container and process them:

```bash
# Copy documents to the container
docker cp /path/to/your/document.pdf document-ai-document-ai-api-1:/app/documents/

# Execute a Python script to add documents
docker exec -it document-ai-document-ai-api-1 python -c "
from ai_service.main import DocumentAI
doc_ai = DocumentAI(collection_name='rag')
doc_ai.add_documents(['/app/documents/document.pdf'])
"
```

#### Duplicate Detection

The system automatically detects duplicate documents using BLAKE2b file hashing before processing and embedding. This prevents:
- Wasting computational resources on re-processing the same files
- Creating duplicate embeddings that could skew search results
- Unnecessary storage usage

**How it works:**
1. Before processing any document, the system calculates a BLAKE2b hash of the file
2. It checks ChromaDB metadata to see if any document with that hash already exists
3. If found, the document is skipped with a log message
4. If not found, the document is processed and the hash is stored in the metadata for future checks

**Force Re-adding:** If you need to re-process a document (e.g., after changing embedding models), you can use the `force_readd=True` parameter.

### DocumentAI Class Methods

#### `__init__(collection_name="rag")`
Initialize the DocumentAI instance with ChromaDB connection parameters. The connection to ChromaDB is automatically configured using environment variables (`CHROMA_HOST` and `CHROMA_PORT`).

#### `add_documents(file_path: str | list[str], force_readd: bool = False)`
Add one or more PDF documents to the vector database with automatic duplicate detection and parallel processing. Features:
1. **Automatic Duplicate Detection**: Uses BLAKE2b file hashing to skip already processed documents
2. **Parallel Processing**: When multiple files are provided, they are processed concurrently using ThreadPoolExecutor (max 2 workers)
3. **Document Processing Pipeline**: Documents are converted to markdown using Docling, split into semantic chunks, embedded using multilingual embeddings, and stored in ChromaDB with metadata including file hash
4. **Force Re-adding**: Use `force_readd=True` to reprocess documents even if they already exist

#### `query(query: str) -> str`
Execute a query against the document collection:
1. Performs similarity search to find relevant document chunks
2. Uses LLM-based reranking for improved relevance
3. Generates a response using the LLM with retrieved context
4. Returns formatted answer with source attribution

#### `query_stream(query: str) -> Iterator[str]`
Execute a query against the document collection and return the result as a streaming iterator:
1. Same retrieval process as `query()` method
2. Returns streaming chunks for real-time response display
3. Useful for web interfaces and chat applications

#### `delete_collection(collection_name="rag")`
Delete a collection from ChromaDB (useful for cleanup or reset).

## API Endpoints

The FastAPI backend provides an OpenAI-compatible API with automatic interactive documentation:

- **Interactive API Documentation**: Available at http://localhost:8008/docs
- **OpenAPI Schema**: Available at http://localhost:8008/openapi.json

### OpenAI-Compatible Endpoints

#### Chat Completions
```bash
curl -X POST http://localhost:8008/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the main topics in the documents?"}
    ]
  }'
```

#### List Models
```bash
curl http://localhost:8008/models
```

### Document Management Endpoints

#### Paperless Integration - Full Load
Triggers a background download and processing of all documents from your Paperless-ngx instance:
```bash
curl -X POST http://localhost:8008/paperless/full_load
```

**Response:**
```json
{
  "status": "started",
  "message": "Full load started in background"
}
```

**Features:**
- Downloads all documents from configured Paperless instance
- Processes documents in parallel for improved performance
- Runs in background thread to avoid blocking API
- Automatically cleans up temporary files after processing
- Comprehensive logging for monitoring progress

#### Collection Management - Drop Collection
Drops the entire ChromaDB collection (useful for reset or cleanup):
```bash
curl -X POST http://localhost:8008/chroma/drop_collection
```

**Response:**
```json
{
  "status": "success",
  "message": "ChromaDB collection dropped"
}
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

#### Docker Profiles

The system uses Docker Compose profiles to manage different deployment modes:

**Available Profiles:**
- `remote`: Remote API mode using cloud-based LLM services
- `local-gpu`: Local GPU mode using GGUF models with CUDA acceleration

**Starting specific profiles:**
```bash
# Remote mode only
docker-compose --profile remote up -d

# Local GPU mode only  
docker-compose --profile local-gpu up -d

# Both modes (not recommended - conflicts on port 8008)
docker-compose --profile remote --profile local-gpu up -d
```

**Switching between modes:**
```bash
# Stop current services
docker-compose down

# Start different profile
docker-compose --profile local-gpu up -d
```

#### Environment Variables
The Document AI API supports the following environment variables:

**Core Configuration:**
- `EMBEDDING_SERVICE`: Embedding service provider (`alibaba`, `huggingface`)
- `LLM_SERVICE`: Language model service (`qwen_remote`, `qwen_local`, `groq`)
- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use
- `LLM_MODEL_NAME`: Name of the LLM model to use
- `DOMAIN`: Domain URL for the application (e.g., `http://localhost:3000`)

**Database Configuration:**
- `CHROMA_HOST`: ChromaDB server hostname (default: `chroma`)
- `CHROMA_PORT`: ChromaDB server port (default: `8000`)

**API Keys:**
- `DASHSCOPE_API_KEY`: API key for Alibaba DashScope services
- `HUGGINGFACE_API_KEY`: API key for HuggingFace services
- `GROQ_API_KEY`: API key for Groq services

**Paperless Integration:**
- `PAPERLESS_API_URL`: URL to your Paperless-ngx API endpoint (e.g., `https://paperless.example.com/api/documents/`)
- `PAPERLESS_API_TOKEN`: API token for authenticating with Paperless-ngx

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
├── Dockerfile.local            # Local GPU-enabled container build
├── app.py                     # FastAPI backend (OpenAI-compatible API)
├── pyproject.toml             # Python dependencies
├── models.py                  # Pydantic models for API requests
├── ai_service/
│   ├── main.py                # Main DocumentAI class with parallel processing
│   ├── add_documents.py       # Document addition logic with duplicate detection
│   ├── query_llm.py          # Query processing and LLM interaction
│   ├── document_conversion/
│   │   ├── docling_pdf_loader.py  # PDF loading with Docling
│   │   ├── document_splitter.py   # Text chunking
│   │   └── convert_documents.py   # Document conversion pipeline
│   ├── embedding/
│   │   ├── alibaba.py         # Alibaba embedding service
│   │   └── embeddings.py      # Embedding abstraction
│   ├── llm/
│   │   └── model.py          # LLM configuration and model providers
│   └── utils/
│       ├── hash_file.py       # BLAKE2b file hashing for duplicate detection
│       ├── existence_check.py # Document existence validation
│       ├── constants.py       # System constants and configuration
│       ├── query.py          # Query utilities
│       └── thinking_animation.py
├── paperless_ingestion/
│   └── PaperlessIngestion.py  # Paperless-ngx integration and document download
└── scripts/
    └── download_model.py      # Model download utilities
```

**Note**: The system now uses a pre-built Docker image (`danitherex/document-ai-api:latest`) for the API service, making deployment faster and more reliable. The Dockerfile is included for development purposes if you need to build custom versions.

## Logs

View real-time logs for all services:
```bash
# All services
docker-compose logs -f

# Specific services
docker-compose logs -f document-ai-api          # Remote API mode
docker-compose logs -f document-ai-api-local    # Local GPU mode
docker-compose logs -f chroma                   # ChromaDB
docker-compose logs -f openwebui                # Web interface
```

## Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   docker-compose down
   docker-compose --profile remote up -d --build --force-recreate
   # or for local GPU
   docker-compose --profile local-gpu up -d --build --force-recreate
   ```

2. **API Key Issues**:
   - Verify your API key is correctly set in docker-compose.yml
   - Check that the API key has necessary permissions
   - Restart the services after changing the key: `docker-compose restart document-ai-api`

3. **GPU/CUDA Issues (Local GPU Mode)**:
   - **NVIDIA Docker runtime not found**:
     ```bash
     # Install NVIDIA Docker runtime
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```
   
   - **CUDA version mismatch**:
     ```bash
     # Check your CUDA version
     nvidia-smi
     
     # Edit Dockerfile.local to match your CUDA version
     # Change both lines with nvidia/cuda:12.5.0 to your version
     ```
   
   - **Insufficient GPU memory**:
     - The Q4 model requires ~8GB VRAM
     - Check GPU memory usage: `nvidia-smi`
     - Close other GPU-intensive applications

4. **Model Download Issues (Local GPU Mode)**:
   - **Slow download**: The GGUF model is ~4GB, ensure stable internet
   - **Download fails**: Check HuggingFace API key and permissions
   - **Manual download**: 
     ```bash
     # Download manually and place in models volume
     docker volume inspect document-ai_models
     ```

5. **Memory Issues with Large Documents**:
   - Monitor container resources: `docker stats`
   - Consider processing documents in smaller batches
   - Increase Docker memory limits if needed

6. **Connection Issues**:
   - Check if all ports are available (3000, 8000, 8008)
   - Verify no other services are using these ports
   - Ensure profile is correctly specified: `--profile remote` or `--profile local-gpu`

7. **Paperless Integration Issues**:
   - **Authentication errors**: Verify `PAPERLESS_API_TOKEN` is correctly set and has read permissions
   - **Connection timeout**: Check `PAPERLESS_API_URL` is accessible from the container
   - **SSL certificate issues**: For self-signed certificates, you may need to disable SSL verification
   - **Rate limiting**: Large Paperless instances may rate-limit API requests
   - **Network connectivity**: Ensure the container can reach your Paperless instance
   
   **Debug Paperless connection**:
   ```bash
   # Test API connectivity from container
   docker exec -it document-ai-document-ai-api-1 curl -H "Authorization: Token YOUR_TOKEN" YOUR_PAPERLESS_URL
   
   # Check Paperless integration logs
   docker-compose logs -f document-ai-api | grep -i paperless
   ```

8. **Parallel Processing Issues**:
   - **Memory pressure**: Multiple concurrent document processing may consume significant memory
   - **Thread contention**: Adjust `max_workers` in `ai_service/main.py` if experiencing performance issues
   - **File system limits**: Ensure sufficient disk space for temporary files during processing

### Health Checks

Check service status:
```bash
# Check all services
docker-compose ps

# Check specific service health
curl http://localhost:8008/models  # API health
curl http://localhost:8008/api/v1/heartbeat  # ChromaDB health

# Check GPU usage (local GPU mode)
nvidia-smi
docker exec document-ai-document-ai-api-local-1 nvidia-smi  # GPU inside container
```

### Performance Optimization

**General Performance:**
1. **Parallel Document Processing**: The system automatically uses parallel processing for multiple documents (max 2 workers by default)
2. **Batch Processing**: When adding large numbers of documents, they are processed concurrently for improved throughput
3. **Duplicate Detection**: BLAKE2b hashing prevents reprocessing of existing documents, saving computational resources
4. **Paperless Integration**: Background processing ensures API responsiveness during bulk document operations

**For Remote API Mode:**
For better performance with large document collections:
1. Increase ChromaDB memory allocation in docker-compose.yml
2. Adjust retrieval parameters in the code
3. Consider using SSD storage for ChromaDB data persistence

**For Local GPU Mode:**
1. **GPU Memory Optimization**:
   - Monitor GPU memory usage: `nvidia-smi`
   - Close other GPU applications before starting
   - Consider using smaller quantized models if memory constrained

2. **Model Performance**:
   - Qwen 2.5 7B Q4: ~8GB VRAM, good quality/speed balance
   - Inference speed depends on GPU memory bandwidth
   - RTX 3080/4080+ recommended for optimal performance

3. **CPU Configuration**:
   - The container automatically uses all available CPU cores
   - Ensure adequate CPU cooling for sustained workloads

4. **Storage Performance**:
   - Place model files on fast storage (SSD/NVMe)
   - Consider persistent volumes for model caching

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

If you need to modify the API service, you can build custom images:

**For Remote API Mode:**
```bash
# Build the standard image locally
docker build -t custom-document-ai-api .

# Update docker-compose.yml to use your custom image
# Change: image: danitherex/document-ai-api:latest
# To: image: custom-document-ai-api
```

**For Local GPU Mode:**
```bash
# Build the GPU-enabled image locally
docker build -f Dockerfile.local -t custom-document-ai-api-local .

# Update docker-compose.yml to use your custom image
# Change: build: context: . / dockerfile: Dockerfile.local
# To: image: custom-document-ai-api-local
```

**CUDA Version Considerations:**
The `Dockerfile.local` uses CUDA 12.5 by default. If you have a different CUDA version:

1. Check your CUDA version: `nvidia-smi`
2. Edit `Dockerfile.local` and update both base images:
   ```dockerfile
   FROM nvidia/cuda:YOUR_VERSION-devel-ubuntu22.04 AS builder
   # and
   FROM nvidia/cuda:YOUR_VERSION-runtime-ubuntu22.04
   ```
3. Rebuild the image

### Adding New Features

The modular architecture makes it easy to:
- Add new document types (modify `document_conversion/`)
- Integrate new LLM providers (modify `llm/`)
- Add new embedding services (modify `embedding/`)
- Customize the API (modify `app.py`)

## License

This project is open source. Feel free to contribute and improve the system!