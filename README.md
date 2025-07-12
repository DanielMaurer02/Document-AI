# Document AI

A powerful document AI system that combines ChromaDB vector database with advanced language models to provide intelligent document retrieval and question-answering capabilities. The system uses Docling for document conversion, HuggingFace embeddings for semantic search, and Groq LLMs for generating responses.

## Features

- **Document Processing**: Supports PDF documents using Docling for high-quality text extraction and conversion to markdown
- **Vector Search**: Utilizes ChromaDB for efficient similarity search with multilingual embeddings
- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with LLM responses for accurate, context-aware answers
- **Document Compression**: Uses LLM-based reranking for improved retrieval quality
- **Source Attribution**: Provides source file paths and chunk information for transparency
- **Multilingual Support**: Uses `intfloat/multilingual-e5-large-instruct` embeddings for multiple languages

## Architecture

- **vector-db-server**: ChromaDB server with observability features (OpenTelemetry, Zipkin)
- **vector-db-client**: Python client for document processing and querying

## Prerequisites

- Docker and Docker Compose
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended for fast Python dependency management)
- API keys for Groq and HuggingFace

## Quick Start

### 1. Start the Vector Database Server

Navigate to the `vector-db-server` directory and start ChromaDB:

```bash
cd vector-db-server
docker-compose up -d
```

This will start:
- ChromaDB server on port 8000
- Zipkin for tracing on port 9411
- OpenTelemetry collector for observability

### 2. Configure Environment Variables

Create a `.env` file in the `vector-db-client` folder with the following variables:

```env
HUGGINGFACE_API_KEY=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
TOKENIZERS_PARALLELISM=false
```

**Note about HuggingFace API Key**: You don't need a paid HuggingFace account. The key is only used to download the embedding model (`intfloat/multilingual-e5-large-instruct`). You can get a free token from [HuggingFace](https://huggingface.co/settings/tokens).

**Available GROQ Models**: You can change the `GROQ_MODEL` to use different models available on Groq, such as:
- `meta-llama/llama-4-scout-17b-16e-instruct` (default)
- `meta-llama/llama-4-maverick-17b-128e-instruct`
- Other Groq-supported models

### 3. Install Dependencies

Navigate to the `vector-db-client` directory and install dependencies:

```bash
cd vector-db-client
uv sync
```

## Usage

### Using the DocumentAI Class

The main interface is the `DocumentAI` class in `main.py`:

```python
from main import DocumentAI

# Initialize the Document AI system
doc_ai = DocumentAI()

# Add documents to the vector database
file_paths = [
    "/path/to/your/document1.pdf",
    "/path/to/your/document2.pdf"
]
doc_ai.add_documents(file_paths)

# Query the documents
query = "What information can you find about...?"
result = doc_ai.query(query)
print(f"Query Result: {result}")
```

### DocumentAI Class Methods

#### `__init__(host="localhost", port=8000, collection_name="rag")`
Initialize the DocumentAI instance with ChromaDB connection parameters.

#### `add_documents(file_path: str | list[str])`
Add one or more PDF documents to the vector database. Documents are:
1. Converted to markdown using Docling
2. Split into semantic chunks
3. Embedded using multilingual embeddings
4. Stored in ChromaDB with metadata

#### `query(query: str) -> str`
Execute a query against the document collection:
1. Performs similarity search to find relevant document chunks
2. Uses LLM-based reranking for improved relevance
3. Generates a response using the Groq LLM with retrieved context
4. Returns formatted answer with source attribution

#### `delete_collection(collection_name="rag")`
Delete a collection from ChromaDB (useful for cleanup or reset).

### Example Usage

#### Example Code

```python
# Complete example
from main import DocumentAI

# Initialize
doc_ai = DocumentAI()

# Add documents
documents = [
    "/Users/documents/manual.pdf",
    "/Users/documents/report.pdf"
]
doc_ai.add_documents(documents)

# Query with context
query = "What are the safety requirements mentioned in the documents?"
result = doc_ai.query(query)
print(result)
```

#### Running the Example

Since `uv` automatically creates and manages a virtual environment, you can run the main script directly:

```bash
cd vector-db-client
uv run main.py
```

This command will:
1. Activate the virtual environment created by `uv sync`
2. Execute the script with all dependencies available
3. No need to manually activate/deactivate the virtual environment

## Customizing the Prompt

You can customize the prompt template used for queries by modifying the `invoke_query` function in `vector-db-client/query_llm.py`. The current prompt template is defined around line 58:

```python
prompt = PromptTemplate.from_template(
    "Context information is below. Each piece of context includes source information in brackets.\n"
    "---------------------\n{context}\n---------------------\n"
    "Given the context information and not prior knowledge, answer the query. "
    "Only include relevant information from the context."
    f"The current Date and Time is {current_date}. You don't need to state where it was found.\n"
    "If relevant, please include the source file path(s) where the information was found and place that information at the bottom.\n"
    "Use markdown formatting for the answer.\n"
    "Directly answer the question without using a heading like Answer\n"
    "Query: {question}\nAnswer:\n"
)
```

### Prompt Customization Options

You can modify:
- **System instructions**: Change how the model should behave
- **Context formatting**: Adjust how retrieved documents are presented
- **Output format**: Modify the response structure (markdown, plain text, etc.)
- **Source attribution**: Change how sources are cited
- **Language instructions**: Add specific language requirements

### Example Custom Prompt

```python
prompt = PromptTemplate.from_template(
    "You are a helpful assistant analyzing technical documents.\n"
    "Context from documents:\n{context}\n"
    "Based on the above context, provide a detailed answer to: {question}\n"
    "Requirements:\n"
    "- Use bullet points for key information\n"
    "- Include confidence level (High/Medium/Low)\n"
    "- Cite sources at the end\n"
    "Answer:"
)
```

## Configuration

### ChromaDB Configuration

The vector database uses cosine similarity for embeddings and can be configured in the `__get_vectorstore` method:

```python
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    client=self.persistent_client,
    collection_metadata={"hnsw:space": "cosine"},  # Can be changed to "l2" or "ip"
)
```

### Retrieval Configuration

Adjust retrieval parameters in `query_llm.py`:

```python
# Number of documents to retrieve
retriver = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Increase for more context
)
```

## Project Structure

```
document-ai/
├── README.md
├── vector-db-server/
│   ├── docker-compose.yml          # ChromaDB server setup
│   └── otel-collector-config.yaml  # Observability configuration
└── vector-db-client/
    ├── main.py                     # Main DocumentAI class
    ├── add_documents.py            # Document addition logic
    ├── query_llm.py               # Query processing and LLM interaction
    ├── .env                       # Environment variables (create this)
    ├── document_conversion/
    │   ├── docling_pdf_loader.py  # PDF loading with Docling
    │   ├── document_splitter.py   # Text chunking
    │   └── main.py               # Document conversion pipeline
    └── llm/
        └── main.py               # LLM configuration (Groq)
```

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**: Ensure the vector-db-server is running (`docker-compose up -d`)

2. **API Key Issues**: 
   - Verify your `.env` file is in the `vector-db-client` directory
   - Check that API keys are valid and have necessary permissions

3. **Memory Issues with Large Documents**: 
   - Reduce chunk size in document splitter
   - Process documents in smaller batches

4. **Slow Query Performance**:
   - Reduce the number of retrieved documents (`k` parameter)
   - Check ChromaDB server resources

### Monitoring

Access observability tools:
- **Zipkin UI**: http://localhost:9411 (distributed tracing)