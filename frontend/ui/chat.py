"""Welcome state and compact conversation rendering."""

from __future__ import annotations

import streamlit as st

try:
    from frontend.ui.sources import render_sources
except ModuleNotFoundError as exc:
    if exc.name != "frontend":
        raise
    from ui.sources import render_sources

SUGGESTIONS = (
    ("Find relevant papers", "Find papers about retrieval-augmented generation."),
    ("Compare retrieval methods", "Compare BM25 and dense retrieval using the indexed papers."),
    ("Explain a methodology", "Explain the methodology of the Quasi-Recurrent Neural Networks paper."),
    ("Summarize findings", "Summarize the main findings of Neural Enquirer."),
)


def render_empty_state() -> str | None:
    st.markdown(
        """
        <section class="welcome">
          <div class="welcome-icon">R</div>
          <h1>Research Assistant</h1>
          <p>Ask questions across your indexed research papers.<br>
          Every answer stays connected to the evidence it used.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    selected = None
    columns = st.columns(2, gap="small")
    for index, (title, prompt) in enumerate(SUGGESTIONS):
        with columns[index % 2]:
            if st.button(
                f"{title}\n\n{prompt}",
                key=f"suggestion-{index}",
                use_container_width=True,
            ):
                selected = prompt
    return selected


def render_conversation(conversation: dict) -> tuple[int, str] | None:
    messages = conversation["messages"]
    if messages:
        st.markdown(
            f'<div class="conversation-title">{_escape(conversation["title"])}</div>',
            unsafe_allow_html=True,
        )

    regenerate = None
    for index, message in enumerate(messages):
        role = message["role"]
        avatar = "👤" if role == "user" else "🔬"
        with st.chat_message(role, avatar=avatar):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

            if role == "assistant" and not message.get("error"):
                _render_answer_meta(message)
                render_sources(
                    message.get("sources", []),
                    message.get("retrieved_chunks", []),
                )
                action_col, feedback_col, spacer = st.columns([1.8, 1.4, 4.3])
                with action_col:
                    if st.button("↻ Regenerate", key=f"regenerate-{conversation['id']}-{index}"):
                        prompt = _preceding_user_prompt(messages, index)
                        if prompt:
                            regenerate = (index, prompt)
                with feedback_col:
                    feedback = st.feedback(
                        "thumbs",
                        key=f"feedback-{conversation['id']}-{index}",
                    )
                    if feedback is not None:
                        message["feedback"] = feedback
                with spacer:
                    st.empty()
    return regenerate


def _render_answer_meta(message: dict) -> None:
    parts = []
    if message.get("latency_ms") is not None:
        parts.append(f"{message['latency_ms'] / 1000:.1f}s")
    parts.append("citations verified" if message.get("citations_valid") else "citations need review")
    if message.get("used_rerank"):
        parts.append("reranker on")
    st.markdown(f'<div class="answer-meta">{" · ".join(parts)}</div>', unsafe_allow_html=True)


def _preceding_user_prompt(messages: list[dict], assistant_index: int) -> str | None:
    for message in reversed(messages[:assistant_index]):
        if message["role"] == "user":
            return message["content"]
    return None


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
