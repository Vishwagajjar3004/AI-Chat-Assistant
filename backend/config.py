"""Application settings, loaded from environment variables (and an optional .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


class Settings:
    """Simple settings holder. Every value can be overridden with an environment variable."""

    def __init__(self) -> None:
        # Ollama
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.default_model: str = os.getenv("DEFAULT_MODEL", "llama3.2")
        self.ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))

        # Chat behaviour
        self.max_message_length: int = int(os.getenv("MAX_MESSAGE_LENGTH", "8000"))
        self.max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful, friendly AI assistant. Answer clearly and accurately. "
            "When you write code, use fenced code blocks and explain it simply.",
        )

        # Database (local SQLite file by default)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.database_url: str = os.getenv("DATABASE_URL") or f"sqlite:///{(DATA_DIR / 'chat.db').as_posix()}"

        # Server (bound to localhost by default - never expose publicly)
        self.backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
        self.backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))


settings = Settings()
