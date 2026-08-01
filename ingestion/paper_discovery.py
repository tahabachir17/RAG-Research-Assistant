from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

try:
    from .arxiv_scraper import ArxivScraper, Paper
except ImportError:
    from arxiv_scraper import ArxivScraper, Paper


class PaperDiscovery(Protocol):
    def search(self, query: str, max_results: int = 50) -> list[Paper]: ...


def _validate(query: str, max_results: int) -> None:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results < 1:
        raise ValueError("max_results must be greater than zero")


@dataclass(slots=True)
class FeymanPaperDiscovery:
    api_url: str | None = None
    api_key: str | None = None
    command: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.api_url = self.api_url or os.getenv("FEYMAN_API_URL")
        self.api_key = self.api_key or os.getenv("FEYMAN_API_KEY")
        self.command = self.command or os.getenv("FEYMAN_DISCOVERY_COMMAND")

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        _validate(query, max_results)
        payload = self._search_http(query, max_results) if self.api_url else self._search_command(query, max_results)
        return dedupe_papers(_paper_from_feyman(item) for item in _extract_items(payload))[:max_results]

    def _search_http(self, query: str, max_results: int) -> Any:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(self.api_url, params={"query": query, "q": query, "max_results": max_results, "limit": max_results}, headers=headers)
            response.raise_for_status()
            return response.json()

    def _search_command(self, query: str, max_results: int) -> Any:
        if not self.command:
            raise ValueError("Configure Feyman discovery with FEYMAN_API_URL or FEYMAN_DISCOVERY_COMMAND")
        args = shlex.split(self.command) + ["--query", query, "--max-results", str(max_results)]
        completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=self.timeout)
        return json.loads(completed.stdout)


class ArxivPaperDiscovery:
    def __init__(self, scraper: ArxivScraper | None = None) -> None:
        self.scraper = scraper or ArxivScraper()

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        return self.scraper.search(query=query, max_results=max_results)


@dataclass(slots=True)
class AlphaXivMCPPaperDiscovery:
    command: str | None = None
    timeout: float = 90.0
    difficulty: int = 5
    use_groq_planner: bool = True
    groq_api_key: str | None = None
    groq_model: str | None = None
    last_payload: Any = None
    last_arguments: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.command = self.command or os.getenv("ALPHAXIV_MCP_COMMAND", "npx -y mcp-remote https://api.alphaxiv.org/mcp/v1")
        self.difficulty = max(1, min(int(self.difficulty), 10))
        self.groq_api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_model = self.groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        _validate(query, max_results)
        payload = self._call_discover_papers(query, max_results)
        self.last_payload = payload
        items = _extract_items(payload)
        if not items:
            preview = json.dumps(payload, ensure_ascii=False, default=str)[:2000]
            tried = json.dumps(self.last_arguments or [], ensure_ascii=False, default=str)
            raise ValueError(f"alphaXiv MCP returned no parseable paper items. Tried arguments: {tried}. Raw payload preview: {preview}")
        return dedupe_papers(score_papers(query, [_paper_from_alphaxiv(item) for item in items], use_embeddings=False))[:max_results]

    def _call_discover_papers(self, query: str, max_results: int) -> Any:
        self.last_arguments = _alphaxiv_argument_sets(query, self.difficulty, self.groq_api_key if self.use_groq_planner else None, self.groq_model)
        payload: Any = {"results": []}
        for arguments in self.last_arguments:
            payload = _coerce_mcp_tool_payload(self._call_mcp_tool("discover_papers", arguments))
            if isinstance(payload, dict):
                payload.setdefault("requested_max_results", max_results)
                payload.setdefault("alphaxiv_arguments", arguments)
            if _extract_items(payload):
                return payload
        return payload
    def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        args = shlex.split(self.command or "")
        if not args:
            raise ValueError("ALPHAXIV_MCP_COMMAND is empty")
        args[0] = _resolve_executable(args[0])
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8")
        try:
            self._send_mcp(process, 1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "rag-ai-research-papers", "version": "0.1.0"}})
            self._read_mcp_response(process, 1)
            self._send_mcp_notification(process, "notifications/initialized", {})
            self._send_mcp(process, 2, "tools/call", {"name": tool_name, "arguments": arguments})
            return self._read_mcp_response(process, 2).get("result")
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _send_mcp(self, process: subprocess.Popen[str], message_id: int, method: str, params: dict[str, Any]) -> None:
        if not process.stdin:
            raise RuntimeError("MCP process stdin is unavailable")
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}) + "\n")
        process.stdin.flush()

    def _send_mcp_notification(self, process: subprocess.Popen[str], method: str, params: dict[str, Any]) -> None:
        if not process.stdin:
            raise RuntimeError("MCP process stdin is unavailable")
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        process.stdin.flush()

    def _read_mcp_response(self, process: subprocess.Popen[str], message_id: int) -> dict[str, Any]:
        if not process.stdout:
            raise RuntimeError("MCP process stdout is unavailable")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    raise RuntimeError(f"alphaXiv MCP command exited with code {process.returncode}")
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(f"alphaXiv MCP error: {message['error']}")
            return message
        raise TimeoutError(f"Timed out waiting for MCP response id={message_id}")


@dataclass(slots=True)
class AlphaXivThenArxivDiscovery:
    primary: Any | None = None
    fallback: Any | None = None
    last_primary_error: str | None = None
    last_provider_used: str | None = None

    def __post_init__(self) -> None:
        self.primary = self.primary or AlphaXivMCPPaperDiscovery()
        self.fallback = self.fallback or ArxivPaperDiscovery()

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        _validate(query, max_results)
        try:
            papers = self.primary.search(query=query, max_results=max_results)
            self.last_primary_error = None
            self.last_provider_used = "alphaxiv-mcp"
            for paper in papers:
                paper.metadata.setdefault("discovery_pipeline", "alphaxiv_then_arxiv")
                paper.metadata.setdefault("discovery_provider_used", "alphaxiv-mcp")
            return papers
        except Exception as exc:
            self.last_primary_error = str(exc)
        papers = self.fallback.search(query=query, max_results=max_results)
        self.last_provider_used = "arxiv"
        for paper in papers:
            paper.metadata.setdefault("discovery_pipeline", "alphaxiv_then_arxiv")
            paper.metadata.setdefault("discovery_provider_used", "arxiv")
            paper.metadata.setdefault("fallback_reason", self.last_primary_error)
        return papers


@dataclass(slots=True)
class ResearchAPIPaperDiscovery:
    scraper: ArxivScraper | None = None
    use_embeddings: bool = True

    def __post_init__(self) -> None:
        self.scraper = self.scraper or ArxivScraper()

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        _validate(query, max_results)
        papers = self.scraper.search(query=query, max_results=max(max_results, max_results * 3))
        return dedupe_papers(score_papers(query, dedupe_papers(papers), self.use_embeddings))[:max_results]


def build_discovery(provider: str = "auto") -> PaperDiscovery:
    provider = provider.strip().lower()
    if provider in {"auto", "default", "alphaxiv-then-arxiv", "alphaxiv_then_arxiv"}:
        return AlphaXivThenArxivDiscovery()
    if provider == "feyman":
        return FeymanPaperDiscovery()
    if provider == "arxiv":
        return ArxivPaperDiscovery()
    if provider in {"alphaxiv", "alphaxiv-mcp", "alphaxiv_mcp"}:
        return AlphaXivMCPPaperDiscovery()
    if provider in {"research-apis", "research_apis", "enriched"}:
        return ResearchAPIPaperDiscovery()
    raise ValueError(f"Unsupported discovery provider: {provider}")


def score_papers(query: str, papers: Iterable[Paper], use_embeddings: bool = True) -> list[Paper]:
    papers = list(papers)
    if not papers:
        return []
    texts = [f"{paper.title}. {paper.summary}" for paper in papers]
    scores = _embedding_scores(query, texts) if use_embeddings else _lexical_scores(query, texts)
    if scores is None:
        scores = _lexical_scores(query, texts)
    for paper, score in zip(papers, scores):
        paper.metadata["local_relevance_score"] = round(float(score), 6)
        citation_count = paper.metadata.get("citation_count")
        influential_count = paper.metadata.get("influential_citation_count")
        paper.metadata["selection_score"] = round(float(score) + min(float(citation_count or 0), 500.0) / 5000.0 + min(float(influential_count or 0), 100.0) / 2000.0, 6)
    return sorted(papers, key=lambda paper: paper.metadata.get("selection_score", 0.0), reverse=True)


def dedupe_papers(papers: Iterable[Paper]) -> list[Paper]:
    seen: set[str] = set()
    deduped: list[Paper] = []
    for paper in papers:
        keys = [f"arxiv:{_strip_arxiv_version(paper.paper_id)}", f"doi:{str(paper.doi).lower().strip()}" if paper.doi else "", f"title:{_normalize_title(paper.title)}"]
        keys = [key for key in keys if key and not key.endswith(":")]
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        deduped.append(paper)
    return deduped
def _extract_items(payload: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if value in (None, "", []):
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, str):
            parsed = _parse_json_text(value)
            if parsed is not None:
                visit(parsed)
                return
            items.extend(_paper_like_items_from_text(value))
            return
        if not isinstance(value, dict):
            return
        if _looks_like_paper(value):
            items.append(value)
            return
        for key in ("papers", "paperResults", "paper_results", "results", "candidates", "recommendations", "recommendedPapers", "items", "data", "works", "documents", "structuredContent", "structured_content", "content", "text"):
            if key in value:
                visit(value[key])

    visit(payload)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = _extract_arxiv_id(_first(item, "arxiv_id", "arxivId", "paper_id", "paperId", "id", "url")) or _normalize_title(str(_first(item, "title", "name") or item))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _looks_like_paper(item: dict[str, Any]) -> bool:
    if _first(item, "title", "name") and (_first(item, "arxiv_id", "arxivId", "paper_id", "paperId", "id", "url") or _first(item, "abstract", "abstract_preview", "abstractPreview", "summary")):
        return True
    explicit_id = _first(item, "arxiv_id", "arxivId", "paper_id", "paperId", "url", "abs_url", "abstract_url", "pdf_url")
    return _extract_arxiv_id(explicit_id) is not None


def _paper_from_alphaxiv(item: dict[str, Any]) -> Paper:
    arxiv_id = _extract_arxiv_id(_first(item, "arxiv_id", "arxivId", "paper_id", "paperId", "id", "url", "abs_url", "abstract_url"))
    arxiv_id = arxiv_id or _extract_arxiv_id(str(item)) or _stable_id(item)
    title = _clean(_first(item, "title", "name") or "Untitled paper")
    summary = _clean(_first(item, "abstract", "abstract_preview", "abstractPreview", "summary", "description", "preview") or "")
    authors = item.get("authors") or item.get("author_names") or []
    if authors and isinstance(authors[0], dict):
        authors = [str(author.get("name", "")).strip() for author in authors]
    if isinstance(authors, str):
        authors = [part.strip() for part in re.split(r",| and ", authors) if part.strip()]
    organizations = item.get("organizations") or item.get("institutions") or []
    if isinstance(organizations, str):
        organizations = [organizations]
    categories = item.get("categories") or item.get("tags") or []
    if isinstance(categories, str):
        categories = [categories]
    entry_id = _first(item, "entry_id", "entryId", "url", "abs_url", "abstract_url") or f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = _first(item, "pdf_url", "pdfUrl", "pdf") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    primary_category = _first(item, "primary_category", "primaryCategory", "category") or (categories[0] if categories else "unknown")
    return Paper(
        paper_id=str(arxiv_id),
        title=title,
        authors=[str(author) for author in authors if author],
        summary=summary,
        published=_first(item, "published", "publication_date", "publicationDate", "date"),
        updated=_first(item, "updated", "updated_at"),
        primary_category=str(primary_category or "unknown"),
        categories=[str(category) for category in categories],
        pdf_url=str(pdf_url) if pdf_url else None,
        entry_id=str(entry_id) if entry_id else None,
        doi=_first(item, "doi", "DOI"),
        journal_ref=_first(item, "journal_ref", "journalRef", "venue"),
        comment=_first(item, "comment", "notes"),
        metadata={"discovery_source": "alphaxiv_mcp", "organizations": [str(org) for org in organizations if org], "alphaxiv_raw": item},
    )


def _paper_from_feyman(item: dict[str, Any]) -> Paper:
    arxiv_id = _first(item, "paper_id", "arxiv_id", "arxivId", "id", "external_id")
    entry_id = _first(item, "entry_id", "entryId", "url", "abs_url", "abstract_url")
    if not arxiv_id and entry_id:
        arxiv_id = str(entry_id).rstrip("/").split("/")[-1]
    pdf_url = _first(item, "pdf_url", "pdfUrl", "pdf", "download_url", "downloadUrl")
    open_access_pdf = item.get("openAccessPdf") or item.get("open_access_pdf")
    if not pdf_url and isinstance(open_access_pdf, dict):
        pdf_url = open_access_pdf.get("url")
    categories = item.get("categories") or item.get("fieldsOfStudy") or []
    if isinstance(categories, str):
        categories = [categories]
    authors = item.get("authors") or []
    if authors and isinstance(authors[0], dict):
        authors = [str(author.get("name", "")).strip() for author in authors]
    primary_category = _first(item, "primary_category", "primaryCategory", "category") or (categories[0] if categories else "unknown")
    return Paper(
        paper_id=str(arxiv_id or _stable_id(item)),
        title=_clean(_first(item, "title", "name") or "Untitled paper"),
        authors=[author for author in authors if author],
        summary=_clean(_first(item, "summary", "abstract", "description") or ""),
        published=_first(item, "published", "published_at", "publicationDate", "year"),
        updated=_first(item, "updated", "updated_at"),
        primary_category=str(primary_category or "unknown"),
        categories=[str(category) for category in categories],
        pdf_url=str(pdf_url) if pdf_url else None,
        entry_id=str(entry_id) if entry_id else None,
        doi=_first(item, "doi", "DOI"),
        journal_ref=_first(item, "journal_ref", "journalRef", "venue"),
        comment=_first(item, "comment", "notes"),
        metadata={"discovery_source": "feyman"},
    )
def _first(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", []):
            return value
    return None


def _stable_id(item: dict[str, Any]) -> str:
    title = _clean(str(item.get("title") or "paper"))
    return title.lower().replace(" ", "-")[:80] or "unknown-paper"


def _clean(value: str) -> str:
    return " ".join(str(value).split())


def _embedding_scores(query: str, texts: list[str]) -> list[float] | None:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return None
    try:
        model = SentenceTransformer(os.getenv("PAPER_DISCOVERY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        embeddings = model.encode([query, *texts], normalize_embeddings=True)
        scores = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().tolist()[0]
        return [float(score) for score in scores]
    except Exception:
        return None


def _lexical_scores(query: str, texts: list[str]) -> list[float]:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return [0.0 for _ in texts]
    return [len(query_terms & set(_tokenize(text))) / len(query_terms) for text in texts]


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _strip_arxiv_version(value: str | None) -> str:
    return re.sub(r"v\d+$", "", str(value or "").strip())


def _normalize_title(value: str) -> str:
    return " ".join(_tokenize(value))


def _resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if os.name == "nt" and not executable.lower().endswith((".exe", ".cmd", ".bat")):
        for suffix in (".cmd", ".exe", ".bat"):
            resolved = shutil.which(executable + suffix)
            if resolved:
                return resolved
    return executable


def _alphaxiv_argument_sets(query: str, difficulty: int, groq_api_key: str | None = None, groq_model: str | None = None) -> list[dict[str, Any]]:
    difficulty = max(1, min(int(difficulty), 10))
    argument_sets = _groq_alphaxiv_argument_sets(query, difficulty, groq_api_key, groq_model)
    argument_sets.append({"keywords": _keywords_from_query(query), "question": _alphaxiv_question(query), "difficulty": difficulty})
    lower = query.lower()
    if any(term in lower for term in ("earthquake", "seismic", "seismology")):
        argument_sets.append({
            "keywords": ["earthquake detection", "seismic", "deep learning", "phase picking"],
            "question": "Research papers about deep learning for earthquake and seismic analysis, including earthquake detection, phase picking, magnitude estimation, early warning, seismic monitoring, and spatio-temporal neural networks. Return concrete papers with arXiv IDs when available.",
            "difficulty": max(difficulty, 7),
        })
    argument_sets.append({
        "keywords": _keywords_from_query(query, include_query_phrase=True),
        "question": f"Find concrete research papers, not a prose summary, about: {query}. Prefer papers with arXiv IDs and include methods, datasets, applications, and related terms.",
        "difficulty": max(difficulty, 7),
    })
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for args in argument_sets:
        key = json.dumps(args, sort_keys=True)
        if key not in seen:
            unique.append(args)
            seen.add(key)
    return unique


def _groq_alphaxiv_argument_sets(query: str, difficulty: int, api_key: str | None, model: str | None) -> list[dict[str, Any]]:
    if not api_key:
        return []
    prompt = f'Return strict JSON only in this shape: {{"plans":[{{"keywords":["term 1","term 2"],"question":"focused paper-discovery question","difficulty":{difficulty}}}]}}. Return at most two plans, each with 1-4 keywords. User topic: {query!r}.'
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
                json={"model": model or "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "Return only valid JSON. No markdown."}, {"role": "user", "content": prompt}], "temperature": 0.2, "response_format": {"type": "json_object"}},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return []
    parsed = _parse_json_text(content)
    if not isinstance(parsed, dict):
        return []
    plans = parsed.get("plans") or parsed.get("queries") or parsed.get("arguments") or []
    if isinstance(plans, dict):
        plans = [plans]
    argument_sets: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        keywords = plan.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        question = str(plan.get("question") or plan.get("query") or "").strip()
        if not keywords or not question:
            continue
        try:
            plan_difficulty = int(plan.get("difficulty", difficulty))
        except (TypeError, ValueError):
            plan_difficulty = difficulty
        argument_sets.append({"keywords": keywords[:4], "question": question, "difficulty": max(1, min(plan_difficulty, 10)), "planner": "groq"})
    return argument_sets[:2]
def _alphaxiv_question(query: str) -> str:
    return f"Find ranked research papers that best match this topic: {query}. Include exact methods, applications, benchmarks, datasets, and related terms. Return concrete candidate papers with arXiv IDs when available."


def _keywords_from_query(query: str, limit: int = 4, include_query_phrase: bool = False) -> list[str]:
    lower = query.lower()
    stop_words = {"a", "an", "and", "are", "as", "for", "from", "how", "in", "of", "on", "or", "papers", "research", "the", "to", "using", "use", "with"}
    known_phrases = ["deep learning", "machine learning", "retrieval augmented generation", "retrieval-augmented generation", "large language model", "earthquake detection", "earthquake analysis", "phase picking", "magnitude estimation", "early warning", "seismic monitoring", "graph neural network"]
    keywords: list[str] = []
    if include_query_phrase and len(query.split()) <= 8:
        keywords.append(query.strip())
    for phrase in known_phrases:
        if phrase in lower and phrase not in keywords:
            keywords.append(phrase)
    if "earthquake" in lower and "seismic" not in keywords:
        keywords.append("seismic")
    for word in [word for word in _tokenize(query) if word not in stop_words]:
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords[:limit] or _tokenize(query)[:limit] or [query]


def _coerce_mcp_tool_payload(result: Any) -> Any:
    if isinstance(result, dict):
        for key in ("structuredContent", "structured_content"):
            if isinstance(result.get(key), (dict, list)):
                return result[key]
        content = result.get("content")
        if isinstance(content, list):
            parsed_items = []
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if isinstance(block.get("json"), (dict, list)):
                    parsed_items.append(block["json"])
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            if parsed_items:
                return {"results": parsed_items}
            for text in text_parts:
                parsed = _parse_json_text(text)
                if parsed is not None:
                    return parsed
            if text_parts:
                return {"results": _paper_like_items_from_text("\n".join(text_parts))}
        return result
    return result


def _parse_json_text(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    candidates = [value]
    candidates.extend(item.strip() for item in re.findall(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.I))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _paper_like_items_from_text(value: str) -> list[dict[str, Any]]:
    return [{"arxiv_id": arxiv_id, "title": f"alphaXiv paper {arxiv_id}", "abstract": ""} for arxiv_id in dict.fromkeys(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", value))]


def _extract_arxiv_id(value: Any) -> str | None:
    if value in (None, "", []):
        return None
    match = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", str(value), flags=re.I)
    return match.group(1) if match else None


def _extract_openalex_work(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "results" in payload:
        results = payload.get("results") or []
        return results[0] if results else None
    return payload
