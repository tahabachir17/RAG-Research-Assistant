"""Build evidence-aligned QASA, QASPER, and SciDQA generation tiers."""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from ingestion.arxiv_scraper import Paper
from ingestion.pipeline import IngestionPipeline
from processing.bm25_indexer import BM25Indexer
from processing.chunker import Chunk, SectionAwareChunker
from .retrieval_stack_diagnostic import ensure_benchmark_dense_collection

from .external_benchmarks import (
    ExternalBenchmarkBuilder,
    ExternalExample,
    canonical_paper_id,
    load_json_records,
)


LOGGER = logging.getLogger(__name__)
QASA_URL = "https://raw.githubusercontent.com/lgresearch/QASA/main/data/testset_answerable_1554_v1.1.json"
SCIDQA_URL = "https://huggingface.co/datasets/yale-nlp/SciDQA/resolve/main/test.jsonl"
QASPER_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/" "qasper-train-dev-v0.3.tgz"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    index = BM25Indexer.load(args.bm25_index)
    original_chunk_count = len(index.chunks)
    cache_dir = args.cache_dir
    loaders = {
        "qasa": lambda: _cached_url_json(cache_dir / "qasa_v1.1.json", QASA_URL),
        "qasper": lambda: _cached_qasper(cache_dir / "qasper.json"),
        "scidqa": lambda: _cached_url_json(cache_dir / "scidqa.jsonl", SCIDQA_URL),
    }
    preparer = None if args.no_ingest_missing else LocalPaperPreparer(args.data_dir)
    builder = ExternalBenchmarkBuilder(
        loaders=loaders,
        chunks=index.chunks,
        paper_preparer=preparer,
    )
    report = builder.build(
        args.output_dir,
        qasa_cap=args.qasa_cap,
        qasper_cap=args.qasper_cap,
        scidqa_cap=args.scidqa_cap,
        seeds=(args.qasa_seed, args.qasper_seed, args.scidqa_seed),
        threshold=args.alignment_threshold,
        ambiguity_margin=args.ambiguity_margin,
    )
    benchmark_index_path = args.output_dir / "external_bm25_index.pkl"
    if len(builder.chunks) > original_chunk_count:
        new_chunks = builder.chunks[original_chunk_count:]
        benchmark_index = BM25Indexer()
        benchmark_index.build([Chunk(**chunk) for chunk in new_chunks])
        temporary_index = benchmark_index_path.with_suffix(
            benchmark_index_path.suffix + ".part"
        )
        benchmark_index.save(temporary_index)
        temporary_index.replace(benchmark_index_path)
        report["bm25_index"] = str(benchmark_index_path)
        report["new_chunks"] = len(new_chunks)
        report_path = args.output_dir / "external_benchmark_report.json"
        temporary = report_path.with_suffix(report_path.suffix + ".part")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
    if benchmark_index_path.is_file() and not args.no_dense_index:
        report["qdrant_index"] = ensure_benchmark_dense_collection(
            benchmark_index_path,
            args.benchmark_qdrant_path,
            args.benchmark_collection,
            embedding_model=args.embedding_model,
            model_cache=args.model_cache,
        )
        report_path = args.output_dir / "external_benchmark_report.json"
        temporary = report_path.with_suffix(report_path.suffix + ".part")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
    print(json.dumps(report, indent=2))
    return 0


class LocalPaperPreparer:
    """Ingest missing arXiv papers with resume, then run the project chunker."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        pipeline: IngestionPipeline | None = None,
        chunker: SectionAwareChunker | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pipeline = pipeline or IngestionPipeline(
            data_dir=self.data_dir, discovery_provider="arxiv"
        )
        self.chunker = chunker or SectionAwareChunker()

    def __call__(self, example: ExternalExample) -> list[dict[str, Any]]:
        paper_id = canonical_paper_id(example.paper_id)
        if not resembl_arxiv_id(paper_id):
            candidates = self.pipeline.discovery.search(
                query=f'ti:"{example.title}"', max_results=3
            )
            expected = normalized_title(example.title)
            paper = next(
                (
                    candidate
                    for candidate in candidates
                    if title_overlap(expected, normalized_title(candidate.title)) >= 0.8
                ),
                None,
            )
            if paper is None:
                LOGGER.warning(
                    "Cannot resolve %s to an arXiv paper", example.source_id
                )
                return []
            paper_id = canonical_paper_id(paper.paper_id)
        else:
            paper = Paper(
                paper_id=paper_id,
                title=example.title,
                authors=[],
                summary="",
                published=None,
                updated=None,
                primary_category="unknown",
                categories=[],
                pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
                entry_id=f"https://arxiv.org/abs/{paper_id}",
            )
        processed = self.data_dir / "processed" / "raw_text" / f"{paper_id}.json"
        if not processed.is_file():
            result = self.pipeline.run(
                f"id:{paper_id}", max_results=1, resume=True, selected_papers=[paper]
            )
            if result.failed:
                LOGGER.warning("Ingestion failed for %s: %s", paper_id, result.errors)
                return []
        if not processed.is_file():
            return []
        document = json.loads(processed.read_text(encoding="utf-8"))
        return [chunk.to_dict() for chunk in self.chunker.chunk(document)]


def resembl_arxiv_id(value: str) -> bool:
    import re

    return bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})", value))


def normalized_title(value: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def title_overlap(expected: set[str], candidate: set[str]) -> float:
    return len(expected & candidate) / len(expected) if expected else 0.0


def _cached_url_json(path: Path, url: str) -> Any:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    return load_json_records(path)


def _cached_qasper(path: Path) -> Any:
    if path.is_file():
        return load_json_records(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(".tgz")
    if not archive.is_file():
        archive_temporary = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(QASPER_URL, timeout=120) as response:
            archive_temporary.write_bytes(response.read())
        archive_temporary.replace(archive)
    with tarfile.open(archive, "r:gz") as bundle:
        member = next(
            item
            for item in bundle.getmembers()
            if item.name.endswith("qasper-train-v0.3.json")
        )
        handle = bundle.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member.name)
        rows = json.loads(handle.read().decode("utf-8"))
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("evaluation/data/external_cache")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/data/external_benchmarks")
    )
    parser.add_argument("--qasa-cap", type=int, default=60)
    parser.add_argument("--qasper-cap", type=int, default=30)
    parser.add_argument("--scidqa-cap", type=int, default=15)
    parser.add_argument("--qasa-seed", type=int, default=1701)
    parser.add_argument("--qasper-seed", type=int, default=2701)
    parser.add_argument("--scidqa-seed", type=int, default=3701)
    parser.add_argument("--alignment-threshold", type=float, default=0.55)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--no-ingest-missing", action="store_true")
    parser.add_argument(
        "--benchmark-qdrant-path",
        type=Path,
        default=Path("evaluation/data/external_benchmarks/qdrant"),
    )
    parser.add_argument("--benchmark-collection", default="bench_external_chunks")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument(
        "--no-dense-index",
        action="store_true",
        help="Explicit diagnostic escape hatch; normal benchmark builds index Qdrant",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
