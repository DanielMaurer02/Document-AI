import json
import logging
import os
import time
import uuid
from typing import Iterable, List
from models import ChatCompletionRequest, ApiKey, ApiKeyResponse
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, Security, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from services.ai_service.main import DocumentAI
from services.paperless_ingestion.PaperlessIngestion import PaperlessIngestion
from services.security_service.main import get_api_key, create_api_key, delete_key, get_db_session
from services.db_service.main import Database

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

DOMAIN: str = os.getenv("DOMAIN", "http://localhost:3000")
MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3-32b")
MODEL_ID: str = f"{MODEL_NAME}-rag"




app = FastAPI(title="OpenAI‑compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DOMAIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and create initial API key on startup
db = Database()

def init_initial_api_key():
    """Create an initial API key if no keys exist."""
    with Session(db._engine) as session:
        # Check if any API keys exist
        statement = select(ApiKey)
        existing_keys = session.exec(statement).first()
        
        if not existing_keys:
            # Create initial API key
            initial_key = str(uuid.uuid4())
            create_api_key("Initial Key", initial_key, session, is_initial=True)
            
            print("=" * 80)
            print("🔑 INITIAL API KEY CREATED")
            print("=" * 80)
            print(f"API Key: {initial_key}")
            print("This key will be automatically deleted when you create your first custom API key.")
            print("Please save this key and use it to authenticate your first requests.")
            print("=" * 80)
            
            logging.info("Created initial API key: %s", initial_key)

# Initialize initial key on startup
init_initial_api_key()




doc_ai = DocumentAI()
paperless_ingest = PaperlessIngestion()


def _format_sse_chunk(
    chunk_id: str,
    created: int,
    *,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    """Return a formatted SSE chat chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "rag-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _chat_stream_generator(prompt: str) -> Iterable[str]:
    """Stream chat completion chunks as Server‑Sent Events (SSE)."""
    chunk_id = str(uuid.uuid4())
    created_time = int(time.time())

    # Initial delta containing role
    yield _format_sse_chunk(
        chunk_id, created_time, delta={"role": "assistant", "content": ""}
    )

    # Stream content pieces
    for token in doc_ai.query_stream(prompt):
        if token:  # Skip empty chunks
            yield _format_sse_chunk(chunk_id, created_time, delta={"content": token})

    # Final "done" chunk
    yield _format_sse_chunk(chunk_id, created_time, delta={}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# TODO: Save answers from llm for session (are being returned in message object)
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Security(get_api_key)):
    """OpenAI‑compatible chat endpoint."""
    logging.info("Received chat completion request: %s", request.as_dict())

    latest_message = request.messages[-1].content

    if request.stream:
        return StreamingResponse(
            _chat_stream_generator(latest_message),
            media_type="text/event-stream; charset=utf-8",
        )

    ai_response = doc_ai.query(latest_message)
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "rag-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ai_response},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/engines", include_in_schema=False)
@app.get("/models")
async def list_models(api_key: str = Security(get_api_key)):
    """Return the single RAG model supported by this server."""
    return {
        "data": [
            {
                "object": "engine",
                "id": MODEL_ID,
                "ready": True,
                "owner": "Alibaba",
                "permissions": None,
                "created": None,
            }
        ]
    }
@app.post("/paperless/full_load")
def full_load_paperless(api_key: str = Security(get_api_key)):
    """Trigger background download and processing of all Paperless documents."""
    def background_task():
        logging.info("Starting full load of Paperless documents")
        downloaded_files = paperless_ingest.download_all_documents()
        if downloaded_files:
            logging.info(f"Successfully downloaded {len(downloaded_files)} documents")
            for downloaded_file in downloaded_files:
                if doc_ai.vectorstore:
                    doc_ai.add_documents([downloaded_file])
            paperless_ingest.cleanup_downloaded_files()
            logging.info("All documents processed with AI service")
        else:
            logging.warning("No documents were downloaded or an error occurred")

    threading.Thread(target=background_task, daemon=True).start()
    return {"status": "started", "message": "Full load started in background"}

#TODO: Webhook endpoint for paperless

@app.delete("/chroma/drop_collection")
def drop_collection(api_key: str = Security(get_api_key)):
    """Drop the ChromaDB collection."""
    logging.info("Dropping ChromaDB collection")
    doc_ai.delete_collection()
    logging.info("ChromaDB collection dropped successfully")
    return {"status": "success", "message": "ChromaDB collection dropped"}

# API Key Management Endpoints
@app.get("/api-keys", response_model=List[ApiKeyResponse])
def list_api_keys(
    api_key: str = Security(get_api_key),
    session: Session = Depends(get_db_session)
):
    """List all API keys."""
    statement = select(ApiKey)
    api_keys = session.exec(statement).all()
    
    return [
        ApiKeyResponse(
            id=key.id,
            name=key.name,
            key=key.key,
            is_active=key.is_active,
            is_initial=key.is_initial,
            created_at=key.created_at.isoformat(),
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None
        )
        for key in api_keys
    ]

@app.post("/api-keys", response_model=ApiKeyResponse)
def create_new_api_key(
    api_key_name: str,
    api_key: str = Security(get_api_key),
    session: Session = Depends(get_db_session)
):
    """Create a new API key."""
    try:
        # Generate a new API key
        generated_key = str(uuid.uuid4())
        
        # Delete any initial keys when creating the first custom key
        deleted_count = delete_key(session, is_initial=True)
        if deleted_count > 0:
            logging.info(f"Deleted {deleted_count} initial API key(s)")
        
        # Create the new API key
        new_key = create_api_key(api_key_name, generated_key, session, is_initial=False)

        return ApiKeyResponse(
            id=new_key.id,
            name=new_key.name,
            key=new_key.key,
            is_active=new_key.is_active,
            is_initial=new_key.is_initial,
            created_at=new_key.created_at.isoformat(),
            last_used_at=new_key.last_used_at.isoformat() if new_key.last_used_at else None
        )
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail="API key already exists")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api-keys/{key}")
def delete_api_key(
    key: str,
    api_key: str = Security(get_api_key),
    session: Session = Depends(get_db_session)
):
    """Deactivate an API key."""
    success = delete_key(session, key)
    if success:
        return {"status": "success", "message": "API key deactivated"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")             

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
