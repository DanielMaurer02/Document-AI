from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from sqlmodel import SQLModel, Field as SQLField
from datetime import datetime, timezone


class ApiKey(SQLModel, table=True):
    """API Key model for database storage."""
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    key: str = SQLField(unique=True, index=True)
    name: str = SQLField(description="Human readable name for the API key")
    is_active: bool = SQLField(default=True)
    is_initial: bool = SQLField(default=False, description="Whether this is the initial auto-generated key")
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = SQLField(default=None)


class ApiKeyResponse(BaseModel):
    id: int | None
    name: str
    key: str
    is_active: bool
    is_initial: bool
    created_at: str
    last_used_at: str | None


class PaperlessWebhookPayload(BaseModel):
    """Webhook payload from Paperless NGX workflows."""
    
    document_url: str = ""
    filename: str = ""
    original_filename: str = ""



class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request payload for POST /chat/completions."""

    messages: List[ChatMessage]
    stream: bool = Field(
        False, description="If true, the response will be streamed using SSE."
    )

    @model_validator(mode="after")
    def validate_messages_not_empty(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("`messages` must contain at least one item.")
        return self

    def as_dict(self) -> dict:
        """Return a dict suitable for logging (avoids pydantic objects)."""
        return {
            "messages": [msg.model_dump() for msg in self.messages],
            "stream": self.stream,
        }
