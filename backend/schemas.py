"""Pydantic schemas used to validate requests and shape responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import settings


class ConversationCreate(BaseModel):
    title: str = Field("New Chat", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Title must not be blank.")
        return value


class ConversationRename(ConversationCreate):
    pass


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


class ChatRequest(BaseModel):
    conversation_id: int | None = Field(None, description="Existing conversation, or null to start a new one.")
    message: str = Field(..., min_length=1, max_length=settings.max_message_length)
    model: str | None = Field(None, max_length=100)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=16, le=8192)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ChatResponse(BaseModel):
    conversation_id: int
    model: str
    user_message: MessageOut
    assistant_message: MessageOut


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    ollama_running: bool
    default_model: str
    default_model_available: bool
