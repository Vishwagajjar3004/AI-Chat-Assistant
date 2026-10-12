"""API tests. Ollama is mocked, so these run without any model installed."""

import asyncio

import httpx
import pytest

from backend.services import ollama_service


@pytest.fixture()
def fake_ollama(monkeypatch):
    """Replace the real Ollama call with a fake that records what it was sent."""
    calls = []

    async def fake_chat(messages, model, temperature=0.7, max_tokens=None):
        calls.append({"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens})
        return f"Echo: {messages[-1]['content']}"

    monkeypatch.setattr(ollama_service, "chat", fake_chat)
    return calls


# ------------------------------------------------------------------ health & models
def test_health_when_ollama_is_down(client, monkeypatch):
    async def down():
        return False

    monkeypatch.setattr(ollama_service, "is_running", down)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama_running"] is False
    assert body["default_model_available"] is False


def test_health_when_model_installed(client, monkeypatch):
    async def up():
        return True

    async def models():
        return [{"name": "llama3.2:latest", "size": 1, "modified_at": None}]

    monkeypatch.setattr(ollama_service, "is_running", up)
    monkeypatch.setattr(ollama_service, "list_models", models)
    body = client.get("/health").json()
    assert body["ollama_running"] is True
    assert body["default_model_available"] is True


def test_models_lists_installed(client, monkeypatch):
    async def models():
        return [{"name": "llama3.2:latest", "size": 2_000_000_000, "modified_at": "2026-01-01"}]

    monkeypatch.setattr(ollama_service, "list_models", models)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "llama3.2:latest"


def test_models_returns_503_when_ollama_down(client, monkeypatch):
    async def models():
        raise ollama_service.OllamaUnavailableError()

    monkeypatch.setattr(ollama_service, "list_models", models)
    response = client.get("/models")
    assert response.status_code == 503
    assert "Ollama is not running" in response.json()["detail"]


# ------------------------------------------------------------------ chat
def test_chat_creates_conversation_and_saves_messages(client, fake_ollama):
    response = client.post("/chat", json={"message": "Hello there"})
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["content"] == "Echo: Hello there"
    assert body["user_message"]["role"] == "user"

    detail = client.get(f"/conversations/{body['conversation_id']}").json()
    assert detail["title"] == "Hello there"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_chat_sends_history_to_the_model(client, fake_ollama):
    first = client.post("/chat", json={"message": "My name is Vishwa"}).json()
    client.post("/chat", json={"conversation_id": first["conversation_id"], "message": "What is my name?"})

    sent = fake_ollama[1]["messages"]
    assert sent[0]["role"] == "system"
    assert [m["content"] for m in sent[1:]] == [
        "My name is Vishwa",
        "Echo: My name is Vishwa",
        "What is my name?",
    ]


def test_chat_passes_model_and_settings(client, fake_ollama):
    client.post("/chat", json={"message": "hi", "model": "mistral", "temperature": 1.2, "max_tokens": 256})
    assert fake_ollama[0]["model"] == "mistral"
    assert fake_ollama[0]["temperature"] == 1.2
    assert fake_ollama[0]["max_tokens"] == 256


@pytest.mark.parametrize(
    "error, status",
    [
        (ollama_service.OllamaUnavailableError(), 503),
        (ollama_service.ModelNotFoundError("nope"), 404),
        (ollama_service.OllamaTimeoutError(), 504),
        (ollama_service.EmptyResponseError(), 502),
    ],
)
def test_chat_errors_are_friendly_and_nothing_is_saved(client, monkeypatch, error, status):
    async def failing_chat(*args, **kwargs):
        raise error

    monkeypatch.setattr(ollama_service, "chat", failing_chat)
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == status
    assert "Traceback" not in response.text
    assert response.json()["detail"] == error.user_message
    assert client.get("/conversations").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 9000},
        {"message": 123},
        {"message": "hi", "temperature": 5},
        {"message": "hi", "max_tokens": 1},
        {},
    ],
)
def test_chat_rejects_invalid_input(client, fake_ollama, payload):
    assert client.post("/chat", json=payload).status_code == 422
    assert fake_ollama == []


def test_chat_unknown_conversation_returns_404(client, fake_ollama):
    response = client.post("/chat", json={"conversation_id": 999, "message": "hi"})
    assert response.status_code == 404


# ------------------------------------------------------------------ conversations
def test_conversation_crud(client):
    created = client.post("/conversations", json={"title": "Ideas"})
    assert created.status_code == 201
    cid = created.json()["id"]

    assert [c["id"] for c in client.get("/conversations").json()] == [cid]

    renamed = client.patch(f"/conversations/{cid}", json={"title": "Better ideas"})
    assert renamed.json()["title"] == "Better ideas"

    assert client.delete(f"/conversations/{cid}").status_code == 204
    assert client.get(f"/conversations/{cid}").status_code == 404
    assert client.delete(f"/conversations/{cid}").status_code == 404


def test_create_conversation_defaults_title(client):
    assert client.post("/conversations", json={}).json()["title"] == "New Chat"


def test_rename_rejects_blank_title(client):
    cid = client.post("/conversations", json={"title": "A"}).json()["id"]
    assert client.patch(f"/conversations/{cid}", json={"title": "   "}).status_code == 422


def test_clear_conversation_keeps_it_but_removes_messages(client, fake_ollama):
    cid = client.post("/chat", json={"message": "hello"}).json()["conversation_id"]
    assert client.delete(f"/conversations/{cid}/messages").status_code == 204

    detail = client.get(f"/conversations/{cid}").json()
    assert detail["messages"] == []
    assert detail["title"] == "hello"


def test_most_recently_used_conversation_is_listed_first(client, fake_ollama):
    first = client.post("/chat", json={"message": "first"}).json()["conversation_id"]
    second = client.post("/chat", json={"message": "second"}).json()["conversation_id"]
    client.post("/chat", json={"conversation_id": first, "message": "again"})

    assert [c["id"] for c in client.get("/conversations").json()] == [first, second]


# ------------------------------------------------------------------ ollama_service error mapping
def _mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        ollama_service, "_client", lambda timeout=None: httpx.AsyncClient(transport=transport, base_url="http://test")
    )


def test_service_returns_reply(monkeypatch):
    _mock_client(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": " Hi! "}}))
    assert asyncio.run(ollama_service.chat([{"role": "user", "content": "yo"}], "llama3.2")) == "Hi!"


def test_service_maps_connection_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    _mock_client(monkeypatch, handler)
    with pytest.raises(ollama_service.OllamaUnavailableError):
        asyncio.run(ollama_service.chat([], "llama3.2"))


def test_service_maps_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    _mock_client(monkeypatch, handler)
    with pytest.raises(ollama_service.OllamaTimeoutError):
        asyncio.run(ollama_service.chat([], "llama3.2"))


def test_service_maps_missing_model(monkeypatch):
    _mock_client(monkeypatch, lambda request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(ollama_service.ModelNotFoundError) as info:
        asyncio.run(ollama_service.chat([], "ghost"))
    assert "ollama pull ghost" in info.value.user_message


def test_service_maps_empty_response(monkeypatch):
    _mock_client(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "  "}}))
    with pytest.raises(ollama_service.EmptyResponseError):
        asyncio.run(ollama_service.chat([], "llama3.2"))


def test_service_hides_server_errors(monkeypatch):
    _mock_client(monkeypatch, lambda request: httpx.Response(500, text="Traceback: secret stuff"))
    with pytest.raises(ollama_service.OllamaError) as info:
        asyncio.run(ollama_service.chat([], "llama3.2"))
    assert "secret" not in info.value.user_message
