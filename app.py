import json
import logging
import os
import time
import uuid
from typing import Iterable
from models import ChatCompletionRequest

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ai_service.main import DocumentAI

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


doc_ai = DocumentAI()


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


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
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
async def list_models():
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
