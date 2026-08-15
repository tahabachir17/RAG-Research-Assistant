"""Maintainable styling for the Streamlit chat shell."""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2f7d68;
            --surface-soft: rgba(127, 127, 127, 0.08);
            --border-soft: rgba(127, 127, 127, 0.20);
            --text-muted: rgba(127, 127, 127, 0.92);
            --accent: #2f7d68;
        }
        .block-container {
            max-width: 900px;
            padding-top: 2.25rem;
            padding-bottom: 7rem;
        }
        [data-testid="stSidebar"] { border-right: 1px solid var(--border-soft); }
        [data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem; }
        .brand { display: flex; align-items: center; gap: .72rem; margin: .15rem 0 1.3rem; }
        .brand-mark, .welcome-icon {
            display: grid; place-items: center; background: var(--accent); color: white;
            font-weight: 700; border-radius: 10px;
        }
        .brand-mark { width: 34px; height: 34px; }
        .brand span:last-child { display: flex; flex-direction: column; line-height: 1.25; }
        .brand small { color: var(--text-muted); font-size: .72rem; }
        .sidebar-label {
            color: var(--text-muted); font-size: .67rem; font-weight: 650;
            letter-spacing: .09em; margin: 1.6rem .25rem .5rem;
        }
        .sidebar-spacer { height: min(28vh, 220px); }
        .health { font-size: .76rem; color: var(--text-muted); padding: .85rem .25rem .2rem; }
        .health i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: .4rem; }
        .health.ready i { background: #27a06d; box-shadow: 0 0 0 3px rgba(39,160,109,.12); }
        .health.degraded i { background: #cc695e; }
        .welcome { text-align: center; padding: clamp(4rem, 12vh, 8rem) 1rem 2.1rem; }
        .welcome-icon { width: 48px; height: 48px; margin: 0 auto 1rem; border-radius: 14px; }
        .welcome h1 { font-size: clamp(1.7rem, 4vw, 2.2rem); margin: 0 0 .55rem; letter-spacing: -.035em; }
        .welcome p { color: var(--text-muted); font-size: .98rem; line-height: 1.65; margin: 0; }
        .conversation-title {
            font-size: .78rem; color: var(--text-muted); text-align: center;
            padding: .15rem 0 1.4rem; border-bottom: 1px solid var(--border-soft);
            margin-bottom: 1.2rem;
        }
        [data-testid="stChatMessage"] {
            background: transparent; border: 0; padding: .8rem .15rem 1.2rem;
        }
        [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
            border-top: 1px solid var(--border-soft);
        }
        [data-testid="stChatMessageContent"] { line-height: 1.65; }
        .answer-meta { color: var(--text-muted); font-size: .72rem; margin: .65rem 0 .25rem; }
        .source-heading { font-size: .78rem; font-weight: 650; margin: 1rem 0 .45rem; }
        .passage {
            background: var(--surface-soft); border-left: 3px solid var(--accent);
            border-radius: 5px; color: var(--text-color); line-height: 1.58;
            padding: .85rem 1rem; font-size: .86rem;
        }
        [data-testid="stChatInput"] { max-width: 900px; margin: 0 auto; }
        [data-testid="stBottom"] { background: linear-gradient(transparent, var(--background-color) 28%); }
        .stButton > button { border-radius: 9px; }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }
        .stButton > button[kind="primary"]:hover {
            background: #286b5a;
            border-color: #286b5a;
        }
        @media (max-width: 720px) {
            .block-container { padding: 1.1rem .8rem 6.5rem; }
            .welcome { padding-top: 2.6rem; }
            [data-testid="column"] { min-width: 100% !important; }
            .sidebar-spacer { height: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
