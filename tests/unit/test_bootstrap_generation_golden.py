from __future__ import annotations

from evaluation.bootstrap_generation_golden import audit_payload, build_payload


def _chunks():
    chunks = []
    for paper_number in range(10):
        paper_id = f"paper-{paper_number}"
        metadata = {"title": f"Paper {paper_number}"}
        for index, section in enumerate(
            ("abstract", "methodology", "results", "limitations")
        ):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}-{index}",
                    "paper_id": paper_id,
                    "section": section,
                    "start_char": index * 100,
                    "text": f"{section} evidence for {paper_id}",
                    "metadata": metadata,
                }
            )
    return chunks


def test_bootstrap_is_aligned_and_remains_unreviewed():
    chunks = _chunks()
    paper_ids = [f"paper-{index}" for index in range(10)]
    payload = build_payload(chunks, paper_ids=paper_ids)

    assert len(payload["questions"]) == 20
    assert audit_payload(payload, chunks) == []
    assert all(not record["reviewed"] for record in payload["questions"])
    assert all(
        chunk_id.startswith(record["paper_id"])
        for record in payload["questions"]
        for chunk_id in record["retrieved_chunk_ids"]
    )


def test_audit_rejects_cross_paper_context():
    chunks = _chunks()
    paper_ids = [f"paper-{index}" for index in range(10)]
    payload = build_payload(chunks, paper_ids=paper_ids)
    payload["questions"][0]["retrieved_chunk_ids"][0] = "paper-1-0"

    assert any("belongs to paper-1" in failure for failure in audit_payload(payload, chunks))
