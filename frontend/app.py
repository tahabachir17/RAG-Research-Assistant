"""ChatGPT-inspired Streamlit client for the research-paper showcase."""

from __future__ import annotations

import streamlit as st

try:
    from frontend.ui.api_client import APIClientError, fetch_health, request_answer
    from frontend.ui.chat import render_conversation, render_empty_state
    from frontend.ui.sidebar import current_conversation, initialize_state, render_sidebar
    from frontend.ui.styles import apply_styles
except ModuleNotFoundError as exc:
    if exc.name != "frontend":
        raise
    # Supports `streamlit run app.py` when the current directory is frontend/.
    from ui.api_client import APIClientError, fetch_health, request_answer
    from ui.chat import render_conversation, render_empty_state
    from ui.sidebar import current_conversation, initialize_state, render_sidebar
    from ui.styles import apply_styles


def _conversation_title(prompt: str) -> str:
    words = prompt.strip().rstrip("?.!").split()
    title = " ".join(words[:5])
    return title + ("…" if len(words) > 5 else "")


st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_styles()
initialize_state()

health = fetch_health()
render_sidebar(health)
conversation = current_conversation()

regenerate = render_conversation(conversation)
suggested_prompt = None
if not conversation["messages"]:
    suggested_prompt = render_empty_state()

typed_prompt = st.chat_input("Ask about your research papers...")
prompt = suggested_prompt or typed_prompt

if regenerate is not None:
    assistant_index, prompt = regenerate
    conversation["messages"].pop(assistant_index)
    append_user = False
elif prompt:
    append_user = True
else:
    append_user = False

if prompt:
    if append_user:
        conversation["messages"].append({"role": "user", "content": prompt})
        if conversation["title"] == "New research chat":
            conversation["title"] = _conversation_title(prompt)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Searching the paper corpus and checking evidence..."):
            try:
                payload = request_answer(
                    prompt,
                    use_rerank=st.session_state.use_rerank,
                )
                message = {
                    "role": "assistant",
                    "content": payload["answer"],
                    "sources": payload.get("sources", []),
                    "retrieved_chunks": payload.get("retrieved_chunks", []),
                    "citations_valid": payload.get("citations_valid", False),
                    "latency_ms": payload.get("latency_ms"),
                    "used_rerank": payload.get("used_rerank", False),
                }
            except APIClientError as exc:
                message = {
                    "role": "assistant",
                    "content": str(exc),
                    "sources": [],
                    "retrieved_chunks": [],
                    "error": True,
                }
        conversation["messages"].append(message)
    st.rerun()
