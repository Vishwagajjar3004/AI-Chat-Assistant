"""Talks to the local Ollama server over HTTP.

Every failure is converted into an OllamaError subclass that carries a short,
friendly message and an HTTP status code. Raw exceptions and stack traces are
only written to the server log - they never reach the chat UI.
"""

import logging

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    status_code = 500
    user_message = "Something went wrong while talking to the AI model. Please try again."

    def __init__(self, user_message: str | None = None) -> None:
        self.user_message = user_message or self.user_message
        super().__init__(self.user_message)


class OllamaUnavailableError(OllamaError):
    status_code = 503
    user_message = "Ollama is not running. Start it with `ollama serve` and try again."


class ModelNotFoundError(OllamaError):
    status_code = 404

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' is not installed. Run `ollama pull {model}` first.")


class OllamaTimeoutError(OllamaError):
    status_code = 504
    user_message = "The model took too long to respond. Try a shorter message or a smaller model."


class EmptyResponseError(OllamaError):
    status_code = 502
    user_message = "The model returned an empty response. Please try again."


def _client(timeout: float | None = None) -> httpx.AsyncClient:
    """Create an HTTP client for Ollama (tests replace this with a mock transport)."""
    return httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout or settings.ollama_timeout)


async def is_running() -> bool:
    """Return True if the Ollama server answers."""
    try:
        async with _client(timeout=3) as client:
            response = await client.get("/api/tags")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def list_models() -> list[dict]:
    """Return the locally installed models as [{name, size, modified_at}, ...]."""
    try:
        async with _client(timeout=10) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        logger.warning("Ollama timed out while listing models: %s", exc)
        raise OllamaTimeoutError() from exc
    except httpx.HTTPError as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        raise OllamaUnavailableError() from exc
    except ValueError as exc:
        logger.warning("Ollama returned invalid JSON for the model list: %s", exc)
        raise OllamaError() from exc

    return [
        {"name": m.get("name", ""), "size": m.get("size"), "modified_at": m.get("modified_at")}
        for m in data.get("models", [])
        if m.get("name")
    ]


async def chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Send the conversation to Ollama and return the assistant's reply text."""
    options: dict = {"temperature": temperature}
    if max_tokens:
        options["num_predict"] = max_tokens
    payload = {"model": model, "messages": messages, "stream": False, "options": options}

    try:
        async with _client() as client:
            response = await client.post("/api/chat", json=payload)
    except httpx.TimeoutException as exc:
        logger.warning("Ollama chat timed out: %s", exc)
        raise OllamaTimeoutError() from exc
    except httpx.HTTPError as exc:
        logger.warning("Could not reach Ollama: %s", exc)
        raise OllamaUnavailableError() from exc

    if response.status_code == 404:
        raise ModelNotFoundError(model)
    if response.status_code != 200:
        logger.error("Ollama returned HTTP %s: %s", response.status_code, response.text[:300])
        raise OllamaError()

    try:
        content = response.json().get("message", {}).get("content", "")
    except ValueError as exc:
        logger.error("Ollama returned invalid JSON: %s", exc)
        raise OllamaError() from exc

    content = (content or "").strip()
    if not content:
        raise EmptyResponseError()
    return content
