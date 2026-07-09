from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore[assignment]


@dataclass(slots=True)
class RawPage:
    page_number: int
    text: str
    char_count: int


@dataclass(slots=True)
class RawDocument:
    paper_id: str
    pdf_path: str
    pages_count: int
    pages: list[RawPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    def to_dict(self) -> dict:
        return asdict(self)


class PDFParser:
    """Extract page-level text from PDFs with PyMuPDF."""

    def extract(self, path: str | Path, paper_id: str | None = None) -> RawDocument:
        return extract_pdf_text(path, paper_id=paper_id)


def extract_pdf_text(pdf_path: str | Path, paper_id: str | None = None) -> RawDocument:
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required to extract PDF text. Install the PyMuPDF package."
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if paper_id is None:
        paper_id = pdf_path.stem

    pages: list[RawPage] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            pages.append(
                RawPage(page_number=page_number, text=text, char_count=len(text))
            )
        page_count = document.page_count

    return RawDocument(
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        pages_count=page_count,
        pages=pages,
    )
