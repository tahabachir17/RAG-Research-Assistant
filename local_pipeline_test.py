import json
from pathlib import Path

from ingestion.arxiv_scraper import Paper
from ingestion.pdf_downloader import PDFDownloader
from ingestion.pipeline import IngestionPipeline
from processing.bm25_indexer import BM25Indexer
from processing.chunker import SectionAwareChunker
from processing.embedder import Embedder


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "arxiv"
METADATA_PATH = ROOT / "data" / "metadata" / "papers.json"
OUTPUT_DIR = ROOT / "data" / "local_pipeline_test"

# Start with False. Change to True after ingestion, chunking, and BM25 work.
RUN_EMBEDDINGS = True

# For the first embedding smoke test, only encode a few chunks.
EMBED_ALL_CHUNKS = False


class LocalPaperCatalog:
    """Return metadata already stored locally; no remote discovery."""

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        return self.papers[:max_results]


def load_local_papers() -> list[Paper]:
    records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return [
        Paper(
            paper_id=record["paper_id"],
            title=record["title"],
            authors=record.get("authors", []),
            summary=record.get("summary", ""),
            published=record.get("published"),
            updated=record.get("updated"),
            primary_category=record.get("primary_category", "unknown"),
            categories=record.get("categories", []),
            pdf_url=record.get("pdf_url"),
            entry_id=record.get("entry_id"),
            doi=record.get("doi"),
        )
        for record in records
    ]


def run_ingestion(papers: list[Paper]) -> None:
    pipeline = IngestionPipeline(
        data_dir=OUTPUT_DIR,
        discovery=LocalPaperCatalog(papers),
        discovery_provider="local",
        # Existing files are detected and marked as skipped; no download occurs.
        downloader=PDFDownloader(raw_dir=RAW_DIR),
    )

    result = pipeline.run(query="local-pdfs", max_results=len(papers))

    print("\nIngestion result")
    print(json.dumps(result.to_dict(), indent=2))

    assert result.discovered == len(papers)
    assert result.downloaded == 0
    assert result.skipped == len(papers)
    assert result.failed == 0
    assert result.processed == len(papers)


def run_processing() -> None:
    processed_dir = OUTPUT_DIR / "processed" / "raw_text"
    document_paths = sorted(processed_dir.rglob("*.json"))

    assert document_paths, "No processed ingestion documents were found"

    chunker = SectionAwareChunker(
        max_tokens=512,
        overlap_tokens=80,
        min_tokens=40,
    )

    all_chunks = []

    print("\nProcessed documents")
    for path in document_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(document)
        all_chunks.extend(chunks)

        populated_sections = {
            name: len(text)
            for name, text in document["sections"].items()
            if text.strip()
        }

        print(
            f"{document['paper_id']}: "
            f"{len(populated_sections)} populated sections, "
            f"{len(chunks)} chunks"
        )

        assert chunks, f"No chunks generated for {document['paper_id']}"

        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.start_char < chunk.end_char

            source = document["sections"].get(chunk.section)
            if source is not None:
                assert source[chunk.start_char : chunk.end_char] == chunk.text

    chunk_ids = [chunk.chunk_id for chunk in all_chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

    chunks_path = OUTPUT_DIR / "processed" / "chunks" / "chunks.json"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(
            [chunk.to_dict() for chunk in all_chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunks saved to: {chunks_path}")

    # Build and persist the sparse index.
    bm25 = BM25Indexer()
    bm25.build(all_chunks)

    bm25_path = OUTPUT_DIR / "processed" / "bm25_index.pkl"
    bm25.save(bm25_path)

    # Confirm that loading produces a usable index.
    restored_bm25 = BM25Indexer.load(bm25_path)
    results = restored_bm25.search("deep reinforcement learning", top_k=5)

    assert results, "BM25 returned no results"

    print("\nBM25 results")
    for result in results:
        print(
            f"{result['score']:.3f} | "
            f"{result['paper_id']} | "
            f"{result['section']} | "
            f"{result['text'][:100]}"
        )

    if RUN_EMBEDDINGS:
        chunks_to_embed = all_chunks if EMBED_ALL_CHUNKS else all_chunks[:10]

        embedder = Embedder()
        embedded_chunks = embedder.encode_chunks(chunks_to_embed)

        assert len(embedded_chunks) == len(chunks_to_embed)
        assert all(item["embedding"] for item in embedded_chunks)
        assert all(
            isinstance(value, float)
            for item in embedded_chunks
            for value in item["embedding"]
        )

        embeddings_path = (
            OUTPUT_DIR / "processed" / "embeddings" / "embeddings.json"
        )
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_path.write_text(
            json.dumps(embedded_chunks, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\nEmbedded chunks: {len(embedded_chunks)}")
        print(f"Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
        print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    local_papers = load_local_papers()
    run_ingestion(local_papers)
    run_processing()