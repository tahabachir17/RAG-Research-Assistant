from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit


_ARXIV_VERSION_RE = re.compile(r"v(?P<version>\d+)$", re.I)
_ARXIV_PREFIX_RE = re.compile(r"^arxiv\s*:\s*", re.I)


def canonical_arxiv_id(paper_id: str) -> str:
    """Return the stable corpus identity for an ArXiv paper."""

    value = _normalized_id(paper_id)
    value = _ARXIV_VERSION_RE.sub("", value)
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not value:
        raise ValueError("paper_id does not contain a usable identity")
    return value


def arxiv_version(paper_id: str) -> str | None:
    match = _ARXIV_VERSION_RE.search(_normalized_id(paper_id))
    return f"v{match.group('version')}" if match else None


def _normalized_id(paper_id: str) -> str:
    """Extract an identifier from short IDs, arXiv labels, and abs/PDF URLs."""

    value = str(paper_id or "").strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        path = unquote(parsed.path).strip("/")
        parts = path.split("/")
        if parts and parts[0].casefold() in {"abs", "pdf"}:
            value = "/".join(parts[1:])
        else:
            value = parts[-1] if parts else ""
    else:
        value = value.split("#", 1)[0].split("?", 1)[0].strip().strip("/")
        value = _ARXIV_PREFIX_RE.sub("", value)
        for prefix in ("abs/", "pdf/"):
            if value.casefold().startswith(prefix):
                value = value[len(prefix) :]
                break
    value = value.rstrip("/")
    if value.casefold().endswith(".pdf"):
        value = value[:-4]
    return value.strip()
