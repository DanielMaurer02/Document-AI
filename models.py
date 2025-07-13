from pydantic import BaseModel, Field, model_validator
from typing import List


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
