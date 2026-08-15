"""HTTP adapter for the existing showcase API."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", "600"))


class APIClientError(RuntimeError):
    """A friendly API failure safe to render directly in the chat."""


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def request_answer(
    question: str, *, use_rerank: bool, top_k: int = 5
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "question": question.strip(),
                "use_rerank": use_rerank,
                "top_k": top_k,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise APIClientError(
            "The assistant is taking longer than expected. Please try again shortly."
        ) from exc
    except requests.ConnectionError as exc:
        raise APIClientError(
            "The assistant service is unavailable. Check that the API is running."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("The assistant could not complete that request.") from exc

    if not response.ok:
        detail = _error_detail(response)
        if response.status_code == 503:
            detail = detail or "The generation provider is temporarily unavailable."
        raise APIClientError(detail or "The assistant could not answer that question.")
    try:
        payload = response.json()
        if not isinstance(payload.get("answer"), str):
            raise ValueError("missing answer")
        return payload
    except (ValueError, AttributeError) as exc:
        raise APIClientError("The assistant returned an unexpected response.") from exc


def _error_detail(response: requests.Response) -> str | None:
    try:
        detail = response.json().get("detail")
        return str(detail) if detail else None
    except (ValueError, AttributeError):
        return None
