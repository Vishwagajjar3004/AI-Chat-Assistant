"""All HTTP endpoints: health, models, chat and conversation management."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import schemas
from ..config import settings
from ..database import get_db
from ..models import Conversation, Message, utcnow
from ..services import ollama_service

router = APIRouter()

DEFAULT_TITLE = "New Chat"


def _get_conversation_or_404(db: Session, conversation_id: int) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return conversation


def _make_title(text: str) -> str:
    first_line = text.strip().splitlines()[0]
    return first_line if len(first_line) <= 40 else first_line[:40].rstrip() + "…"


def _model_installed(wanted: str, installed: list[str]) -> bool:
    """'llama3.2' matches 'llama3.2' and 'llama3.2:latest'; 'llama3.2:1b' must match exactly."""
    if wanted in installed:
        return True
    return ":" not in wanted and f"{wanted}:latest" in installed


# ---------------------------------------------------------------- health & models
@router.get("/health", response_model=schemas.HealthResponse)
async def health() -> schemas.HealthResponse:
    running = await ollama_service.is_running()
    available = False
    if running:
        try:
            names = [m["name"] for m in await ollama_service.list_models()]
            available = _model_installed(settings.default_model, names)
        except ollama_service.OllamaError:
            available = False
    return schemas.HealthResponse(
        status="ok",
        ollama_running=running,
        default_model=settings.default_model,
        default_model_available=available,
    )


@router.get("/models", response_model=schemas.ModelsResponse)
async def models() -> schemas.ModelsResponse:
    try:
        installed = await ollama_service.list_models()
    except ollama_service.OllamaError as exc:
        raise HTTPException(exc.status_code, exc.user_message) from exc
    return schemas.ModelsResponse(default_model=settings.default_model, models=installed)


# ---------------------------------------------------------------- chat
@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(request: schemas.ChatRequest, db: Session = Depends(get_db)) -> schemas.ChatResponse:
    if request.conversation_id is None:
        conversation = Conversation(title=DEFAULT_TITLE)
        db.add(conversation)
        db.flush()
    else:
        conversation = _get_conversation_or_404(db, request.conversation_id)

    # Last N saved messages give the model its context.
    recent = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id.desc())
        .limit(settings.max_history_messages)
    ).all()
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

    model = request.model or settings.default_model
    ollama_messages = [
        {"role": "system", "content": settings.system_prompt},
        *history,
        {"role": "user", "content": request.message},
    ]

    try:
        reply = await ollama_service.chat(
            ollama_messages, model=model, temperature=request.temperature, max_tokens=request.max_tokens
        )
    except ollama_service.OllamaError as exc:
        db.rollback()  # nothing is saved (and a brand-new conversation is discarded) on failure
        raise HTTPException(exc.status_code, exc.user_message) from exc

    user_msg = Message(conversation_id=conversation.id, role="user", content=request.message)
    assistant_msg = Message(conversation_id=conversation.id, role="assistant", content=reply)
    db.add_all([user_msg, assistant_msg])
    if conversation.title == DEFAULT_TITLE:
        conversation.title = _make_title(request.message)
    conversation.updated_at = utcnow()
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return schemas.ChatResponse(
        conversation_id=conversation.id,
        model=model,
        user_message=user_msg,
        assistant_message=assistant_msg,
    )


# ---------------------------------------------------------------- conversations
@router.post("/conversations", response_model=schemas.ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: schemas.ConversationCreate, db: Session = Depends(get_db)) -> Conversation:
    conversation = Conversation(title=payload.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/conversations", response_model=list[schemas.ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[Conversation]:
    return list(db.scalars(select(Conversation).order_by(Conversation.updated_at.desc(), Conversation.id.desc())))


@router.get("/conversations/{conversation_id}", response_model=schemas.ConversationDetail)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)) -> Conversation:
    return _get_conversation_or_404(db, conversation_id)


@router.patch("/conversations/{conversation_id}", response_model=schemas.ConversationOut)
def rename_conversation(
    conversation_id: int, payload: schemas.ConversationRename, db: Session = Depends(get_db)
) -> Conversation:
    conversation = _get_conversation_or_404(db, conversation_id)
    conversation.title = payload.title
    db.commit()
    db.refresh(conversation)
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)) -> Response:
    conversation = _get_conversation_or_404(db, conversation_id)
    db.delete(conversation)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/conversations/{conversation_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_conversation(conversation_id: int, db: Session = Depends(get_db)) -> Response:
    """Remove every message but keep the conversation itself."""
    conversation = _get_conversation_or_404(db, conversation_id)
    conversation.messages.clear()
    conversation.updated_at = utcnow()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
