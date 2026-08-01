from __future__ import annotations

import re


_ARXIV_VERSION_RE = re.compile(r"v(?P<version>\d+)$", re.I)


def canonical_arxiv_id(paper_id: str) -> str:
    """Return the stable corpus identity for an ArXiv paper."""

    value = str(paper_id or "").strip().rstrip("/").split("/")[-1]
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    value = _ARXIV_VERSION_RE.sub("", value)
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not value:
        raise ValueError("paper_id does not contain a usable identity")
    return value


def arxiv_version(paper_id: str) -> str | None:
    match = _ARXIV_VERSION_RE.search(str(paper_id or "").strip())
    return f"v{match.group('version')}" if match else None
