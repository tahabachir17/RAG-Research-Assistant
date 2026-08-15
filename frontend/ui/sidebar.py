"""Conversation navigation and compact application settings."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st


def initialize_state() -> None:
    if "conversations" not in st.session_state:
        conversation = _new_conversation()
        st.session_state.conversations = {conversation["id"]: conversation}
        st.session_state.current_conversation_id = conversation["id"]
    if "use_rerank" not in st.session_state:
        st.session_state.use_rerank = False


def current_conversation() -> dict:
    return st.session_state.conversations[st.session_state.current_conversation_id]


def render_sidebar(health: dict | None) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">R</span>'
            '<span><strong>Research Assistant</strong>'
            '<small>Evidence-grounded AI</small></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("＋  New chat", type="primary", use_container_width=True):
            _start_new_chat()
            st.rerun()

        st.markdown('<p class="sidebar-label">CONVERSATIONS</p>', unsafe_allow_html=True)
        conversations = list(st.session_state.conversations.values())
        for conversation in reversed(conversations):
            active = conversation["id"] == st.session_state.current_conversation_id
            label = f"{'●  ' if active else ''}{conversation['title']}"
            if st.button(
                label,
                key=f"conversation-{conversation['id']}",
                use_container_width=True,
            ):
                st.session_state.current_conversation_id = conversation["id"]
                st.rerun()

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        with st.expander("Settings & about"):
            st.toggle(
                "Use experimental reranker",
                key="use_rerank",
                help="Off by default: evaluation found added latency without recall lift.",
            )
            st.caption("The indexed paper corpus is frozen and read-only for this demo.")

        if health and health.get("status") == "ready":
            st.markdown('<div class="health ready"><i></i> Corpus and model ready</div>', unsafe_allow_html=True)
        elif health:
            st.markdown('<div class="health degraded"><i></i> Service degraded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="health degraded"><i></i> API unavailable</div>', unsafe_allow_html=True)


def _start_new_chat() -> None:
    current = current_conversation()
    if not current["messages"]:
        return
    conversation = _new_conversation()
    st.session_state.conversations[conversation["id"]] = conversation
    st.session_state.current_conversation_id = conversation["id"]


def _new_conversation() -> dict:
    conversation_id = uuid4().hex
    return {
        "id": conversation_id,
        "title": "New research chat",
        "messages": [],
    }
