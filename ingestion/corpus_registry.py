from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .arxiv_scraper import Paper
    from .identity import arxiv_version, canonical_arxiv_id
except ImportError:  # Allows direct script imports from the repository root.
    from arxiv_scraper import Paper
    from identity import arxiv_version, canonical_arxiv_id


class CorpusRegistry:
    """Transactional paper registry and discovery checkpoint store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    canonical_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    source_version TEXT,
                    status TEXT NOT NULL DEFAULT 'discovered',
                    metadata_json TEXT,
                    pdf_path TEXT,
                    processed_path TEXT,
                    failure_stage TEXT,
                    last_error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discovery_checkpoints (
                    run_key TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    canonical_id TEXT NOT NULL,
                    paper_json TEXT NOT NULL,
                    PRIMARY KEY (run_key, canonical_id)
                );
                CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_order
                    ON discovery_checkpoints(run_key, position);
                """
            )

    def checkpoint(self, run_key: str, papers: Iterable[Paper]) -> None:
        """Atomically replace a run checkpoint and register its papers."""

        if not str(run_key).strip():
            raise ValueError("run_key must be non-empty")
        now = _now()
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM discovery_checkpoints WHERE run_key=?", (run_key,)
            )
            for position, paper in enumerate(papers):
                canonical_id = canonical_arxiv_id(paper.paper_id)
                payload = json.dumps(paper.to_dict(), ensure_ascii=False, default=str)
                connection.execute(
                    """
                    INSERT INTO discovery_checkpoints
                        (run_key, position, canonical_id, paper_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_key, canonical_id) DO UPDATE SET
                        position=excluded.position, paper_json=excluded.paper_json
                    """,
                    (run_key, position, canonical_id, payload),
                )
                self._register(connection, paper, canonical_id, now)

    @staticmethod
    def _register(
        connection: sqlite3.Connection,
        paper: Paper,
        canonical_id: str,
        now: str,
    ) -> None:
        source_version = arxiv_version(paper.paper_id)
        current = connection.execute(
            "SELECT paper_id, source_version FROM papers WHERE canonical_id=?",
            (canonical_id,),
        ).fetchone()
        if current is None:
            connection.execute(
                """
                INSERT INTO papers
                    (canonical_id, paper_id, source_version, status,
                     created_at, updated_at)
                VALUES (?, ?, ?, 'discovered', ?, ?)
                """,
                (canonical_id, paper.paper_id, source_version, now, now),
            )
            return
        is_newer = _version_number(source_version) > _version_number(
            current["source_version"]
        )
        if is_newer:
            connection.execute(
                """
                UPDATE papers SET paper_id=?, source_version=?, status='discovered',
                    metadata_json=NULL, pdf_path=NULL, processed_path=NULL,
                    failure_stage=NULL, last_error=NULL, updated_at=?
                WHERE canonical_id=?
                """,
                (paper.paper_id, source_version, now, canonical_id),
            )
        else:
            connection.execute(
                "UPDATE papers SET updated_at=? WHERE canonical_id=?",
                (now, canonical_id),
            )

    def load_checkpoint(self, run_key: str) -> list[Paper]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT paper_json FROM discovery_checkpoints "
                "WHERE run_key=? ORDER BY position",
                (run_key,),
            ).fetchall()
        return [_paper_from_dict(json.loads(row["paper_json"])) for row in rows]

    def get(self, paper_id: str) -> dict[str, Any] | None:
        canonical_id = canonical_arxiv_id(paper_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE canonical_id=?", (canonical_id,)
            ).fetchone()
        return dict(row) if row else None

    def mark(
        self,
        paper_id: str,
        status: str,
        *,
        pdf_path: str | Path | None = None,
        processed_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        failure_stage: str | None = None,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> bool:
        canonical_id = canonical_arxiv_id(paper_id)
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE papers SET status=?,
                    pdf_path=COALESCE(?, pdf_path),
                    processed_path=COALESCE(?, processed_path),
                    metadata_json=COALESCE(?, metadata_json),
                    failure_stage=?, last_error=?,
                    attempts=attempts + ?, updated_at=?
                WHERE canonical_id=?
                """,
                (
                    status,
                    str(pdf_path) if pdf_path is not None else None,
                    str(processed_path) if processed_path is not None else None,
                    json.dumps(metadata, ensure_ascii=False, default=str)
                    if metadata is not None
                    else None,
                    failure_stage,
                    error,
                    1 if increment_attempts else 0,
                    now,
                    canonical_id,
                ),
            )
        return cursor.rowcount == 1

    def records(self, status: str | None = None) -> list[dict[str, Any]]:
        """Return registry rows, optionally filtered by status."""

        query = "SELECT * FROM papers"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            parameters = (status,)
        query += " ORDER BY canonical_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def export_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT metadata_json FROM papers
                WHERE metadata_json IS NOT NULL
                ORDER BY canonical_id
                """
            ).fetchall()
        records = [json.loads(row["metadata_json"]) for row in rows]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination


def _paper_from_dict(record: dict[str, Any]) -> Paper:
    fields = Paper.__dataclass_fields__
    known = {key: record.get(key) for key in fields if key != "metadata"}
    known["metadata"] = {
        **dict(record.get("metadata") or {}),
        **{key: value for key, value in record.items() if key not in fields},
    }
    return Paper(**known)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_number(version: str | None) -> int:
    if not version:
        return 0
    match = re.search(r"\d+", version)
    return int(match.group()) if match else 0
