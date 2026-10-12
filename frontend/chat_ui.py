"""Streamlit chat interface. Talks to the FastAPI backend over HTTP."""

import os
from datetime import datetime, timezone

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "8000"))
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120")) + 15

CSS = """
<style>
.block-container { max-width: 880px; padding-top: 2rem; }
[data-testid="stChatMessage"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    background: rgba(255,255,255,0.03);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(99,102,241,0.14);
    border-color: rgba(99,102,241,0.35);
}
[data-testid="stSidebar"] .stButton > button { text-align: left; justify-content: flex-start; }
.welcome { text-align: center; margin-top: 18vh; }
.welcome h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
.welcome p { opacity: 0.7; font-size: 1.05rem; }
@media (max-width: 640px) {
    .block-container { padding: 1rem 0.6rem; }
    .welcome { margin-top: 8vh; }
    .welcome h1 { font-size: 1.6rem; }
}
</style>
"""


class ApiError(Exception):
    """A friendly error message from (or about) the backend."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api(method: str, path: str, **kwargs):
    """Call the backend and return parsed JSON (or None for 204). Raises ApiError with a readable message."""
    try:
        response = httpx.request(method, f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    except httpx.ConnectError:
        raise ApiError(f"Cannot reach the backend at {BACKEND_URL}. Is it running?") from None
    except httpx.TimeoutException:
        raise ApiError("The request timed out. Please try again.") from None
    except httpx.HTTPError:
        raise ApiError("A network error occurred while contacting the backend.") from None

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, str):
            message = detail
        elif response.status_code == 422:
            message = "Your input was not valid. Please check it and try again."
        else:
            message = "Something went wrong. Please try again."
        raise ApiError(message, response.status_code)

    return None if response.status_code == 204 else response.json()


def fmt_time(iso: str) -> str:
    """Backend timestamps are naive UTC; show them in the user's local time."""
    try:
        return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).astimezone().strftime("%d %b, %H:%M")
    except ValueError:
        return ""


def fmt_size(size: int | None) -> str:
    return f"{size / 1024**3:.1f} GB" if size else ""


def init_state() -> None:
    st.session_state.setdefault("conversation_id", None)


# ------------------------------------------------------------------ sidebar
def render_sidebar(health: dict) -> None:
    with st.sidebar:
        st.markdown("## 💬 AI Chat Assistant")
        if st.button("➕ New Chat", type="primary", width="stretch"):
            st.session_state.conversation_id = None
            st.rerun()

        st.divider()
        st.caption("CONVERSATIONS")
        try:
            conversations = api("GET", "/conversations")
        except ApiError as exc:
            st.error(str(exc))
            conversations = []

        if not conversations:
            st.caption("No saved conversations yet.")
        for conv in conversations:
            active = conv["id"] == st.session_state.conversation_id
            open_col, delete_col = st.columns([5, 1])
            label = ("● " if active else "") + conv["title"]
            if open_col.button(label, key=f"open_{conv['id']}", width="stretch"):
                st.session_state.conversation_id = conv["id"]
                st.rerun()
            if delete_col.button("🗑", key=f"delete_{conv['id']}", help="Delete this conversation"):
                try:
                    api("DELETE", f"/conversations/{conv['id']}")
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    if active:
                        st.session_state.conversation_id = None
                    st.rerun()

        render_current_chat_tools()
        render_settings(health)


def render_current_chat_tools() -> None:
    conversation_id = st.session_state.conversation_id
    if conversation_id is None:
        return
    st.divider()
    st.caption("CURRENT CHAT")
    with st.expander("Rename / Clear"):
        new_title = st.text_input("Title", key=f"rename_{conversation_id}", max_chars=200, label_visibility="collapsed")
        if st.button("Save title", key=f"save_title_{conversation_id}", width="stretch"):
            if not new_title.strip():
                st.warning("Please type a title first.")
            else:
                try:
                    api("PATCH", f"/conversations/{conversation_id}", json={"title": new_title})
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
        if st.button("🧹 Clear Chat", key=f"clear_{conversation_id}", width="stretch"):
            try:
                api("DELETE", f"/conversations/{conversation_id}/messages")
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def render_settings(health: dict) -> None:
    st.divider()
    st.caption("AI SETTINGS")

    if health["ollama_running"]:
        st.success("Ollama is running", icon="✅")
    else:
        st.error("Ollama is not running. Start it with `ollama serve`.", icon="⚠️")

    model_names: list[str] = []
    installed: list[dict] = []
    if health["ollama_running"]:
        try:
            installed = api("GET", "/models")["models"]
            model_names = [m["name"] for m in installed]
        except ApiError as exc:
            st.warning(str(exc))

    default = health["default_model"]
    if model_names:
        index = next((i for i, n in enumerate(model_names) if n == default or n == f"{default}:latest"), 0)
        st.selectbox("Model", model_names, index=index, key="model_select")
    else:
        st.session_state["model_select"] = default
        st.text_input("Model", value=default, disabled=True)
        if health["ollama_running"]:
            st.warning(f"No models installed. Run `ollama pull {default}`.")

    st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="temperature", help="Higher = more creative, lower = more focused.")
    st.slider("Max response length (tokens)", 64, 4096, 1024, 64, key="max_tokens")

    if installed:
        with st.expander(f"Installed models ({len(installed)})"):
            for m in installed:
                st.markdown(f"✅ **{m['name']}** {fmt_size(m.get('size'))}")
    if health["ollama_running"] and not health["default_model_available"]:
        st.warning(f"Default model `{default}` is not installed. Run `ollama pull {default}`.")


# ------------------------------------------------------------------ main area
def render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome">
          <h1>👋 How can I help you today?</h1>
          <p>Ask a question, get code explained, summarize or translate text —<br>
          everything runs locally on your computer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(fmt_time(message["created_at"]))
        if message["role"] == "assistant":
            with st.popover("📋 Copy response"):
                st.caption("Click the copy icon at the top-right of the box:")
                st.code(message["content"], language=None, wrap_lines=True)


def send_message(prompt: str) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                data = api(
                    "POST",
                    "/chat",
                    json={
                        "conversation_id": st.session_state.conversation_id,
                        "message": prompt,
                        "model": st.session_state.get("model_select"),
                        "temperature": st.session_state.get("temperature", 0.7),
                        "max_tokens": st.session_state.get("max_tokens", 1024),
                    },
                )
            except ApiError as exc:
                if exc.status_code == 404 and "Conversation" in str(exc):
                    st.session_state.conversation_id = None
                st.error(f"{exc}\n\nYour message was not saved — please try again.")
                return
    st.session_state.conversation_id = data["conversation_id"]
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    try:
        health = api("GET", "/health")
    except ApiError as exc:
        st.error(f"{exc}\n\nStart it with: `uvicorn backend.main:app --host 127.0.0.1 --port 8000`")
        st.stop()

    render_sidebar(health)
    prompt = st.chat_input("Type your message… (Shift+Enter for a new line)", max_chars=MAX_MESSAGE_LENGTH)

    conversation_id = st.session_state.conversation_id
    messages: list[dict] = []
    if conversation_id is not None:
        try:
            detail = api("GET", f"/conversations/{conversation_id}")
            messages = detail["messages"]
            st.markdown(f"#### {detail['title']}")
            st.caption(f"Model: `{st.session_state.get('model_select', health['default_model'])}`")
        except ApiError as exc:
            if exc.status_code == 404:
                st.session_state.conversation_id = None
            else:
                st.error(str(exc))

    if not messages and not prompt:
        render_welcome()
    for message in messages:
        render_message(message)

    if prompt:
        send_message(prompt)
