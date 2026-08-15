"""Research evidence presentation for assistant messages."""

from __future__ import annotations

import streamlit as st


def render_sources(sources: list[dict], chunks: list[str]) -> None:
    if not sources and not chunks:
        return

    st.markdown('<div class="source-heading">Sources & retrieved evidence</div>', unsafe_allow_html=True)
    if not sources:
        st.caption("No source was cited in the answer. The passages considered by the model are available below.")
        for number, passage in enumerate(chunks, 1):
            with st.expander(f"Retrieved passage {number}"):
                st.markdown(f'<div class="passage">{_escape(passage)}</div>', unsafe_allow_html=True)
        return

    for source in sources:
        number = int(source.get("citation_number", 0))
        title = source.get("title") or "Untitled paper"
        section = source.get("section") or "Unknown section"
        with st.expander(f"[{number}] {title} · {section}"):
            st.markdown(f"**Section:** {section}")
            if source.get("url"):
                st.link_button("Open paper ↗", source["url"])
            passage = chunks[number - 1] if 0 < number <= len(chunks) else None
            if passage:
                st.markdown("**Passage sent to the model**")
                st.markdown(f'<div class="passage">{_escape(passage)}</div>', unsafe_allow_html=True)


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
