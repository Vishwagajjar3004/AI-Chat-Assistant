# AI Chat Assistant (100% local)

A ChatGPT-style assistant that runs entirely on your computer.

- **AI engine:** [Ollama](https://ollama.com) with `llama3.2` (no API keys, no cloud, no paid services)
- **Backend:** FastAPI + SQLAlchemy + SQLite
- **Frontend:** Streamlit (dark theme)
- **Storage:** all conversations are saved locally in `data/chat.db`

## Features

- Chat, explain topics, write/debug Python, summarize and translate text
- Context-aware conversations (history is sent to the model)
- New / rename / delete / clear conversations, loaded from SQLite
- Model picker, temperature and response-length controls, Ollama status
- Friendly errors when Ollama is stopped, the model is missing, or a request times out

## Project structure

```
ai_chat_assistant/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── backend/
│   ├── main.py             # FastAPI app
│   ├── config.py           # settings from environment variables
│   ├── database.py         # engine, sessions
│   ├── models.py           # Conversation, Message tables
│   ├── schemas.py          # Pydantic validation
│   ├── routers/chat.py     # all API endpoints
│   └── services/ollama_service.py
├── frontend/chat_ui.py     # chat interface
├── data/                   # SQLite database lives here
└── tests/
```

## Setup

### 1. Install Ollama and download the model

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve            # skip if it already runs as a service
ollama pull llama3.2
```

**Windows**

1. Download and install Ollama from https://ollama.com/download/windows (it starts automatically).
2. In PowerShell:

```powershell
ollama pull llama3.2
```

### 2. Create a virtual environment and install dependencies

**Linux**

```bash
cd ai_chat_assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
cd ai_chat_assistant
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

> If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. Start the backend (terminal 1)

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 4. Start the frontend (terminal 2, same venv activated)

```bash
streamlit run app.py
```

Open http://localhost:8501

### 5. Verify it works

1. Check the backend: open http://127.0.0.1:8000/health — it should show `"ollama_running": true`.
2. In the app, type "Hello, who are you?" and press Enter — you should get a reply.
3. Ask a follow-up like "What did I just ask you?" — the answer proves context is kept.
4. Stop both servers (Ctrl+C), start them again, and click your conversation in the sidebar —
   the messages are still there, loaded from SQLite.

API docs are available at http://127.0.0.1:8000/docs.

## Run the tests

The tests do not need Ollama (it is mocked) and use an in-memory database.

```bash
python -m pytest -v
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | App + Ollama status |
| GET | `/models` | Installed Ollama models |
| POST | `/chat` | Send a message, get a reply (saves both) |
| POST | `/conversations` | Create a conversation |
| GET | `/conversations` | List conversations |
| GET | `/conversations/{id}` | Conversation with messages |
| PATCH | `/conversations/{id}` | Rename |
| DELETE | `/conversations/{id}` | Delete |
| DELETE | `/conversations/{id}/messages` | Clear messages |

## Security notes

- Backend and frontend bind to `127.0.0.1` only; do not expose Ollama or this app to the internet.
- Input is validated (length, types, temperature range).
- AI-generated code is only displayed, never executed.
- Errors shown in the UI are friendly messages; details go only to the server log.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Ollama is not running" | Run `ollama serve` (Linux) or start the Ollama app (Windows) |
| "Model 'llama3.2' is not installed" | Run `ollama pull llama3.2` |
| "Cannot reach the backend" | Start `uvicorn backend.main:app --host 127.0.0.1 --port 8000` from the project root |
| Slow first reply | The model is loading into memory; later replies are faster |
