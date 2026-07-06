from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any
import fitz
import json

@dataclass
class RawPage:
    page_number: int
    text: str
    char_count: int

@dataclass
class RawDocument:
    paper_id: str
    pdf_path: str
    pages_count: int
    pages: List[RawPage]


def extract_pdf_text(pdf_path: str | Path, paper_id: str | None = None) -> RawDocument:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if paper_id is None:
        paper_id = pdf_path.stem

    pages = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            char_count = len(text)

            pages.append(RawPage(
                page_number=page_number,
                text=text,
                char_count=char_count
            ))
        
        raw_doc = RawDocument(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            pages_count=doc.page_count,
            pages=pages
        )

    return raw_doc



RAW_DIR = Path("../data/raw/arxiv")
OUTPUT_DIR = Path("../data/processed/raw_text")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for category_folder in RAW_DIR.iterdir():

    # skip non-folders (safety check)
    if not category_folder.is_dir():
        continue

    print(f"\n📁 Processing category: {category_folder.name}")

    # loop through PDFs in category
    for pdf_file in category_folder.glob("*.pdf"):

        try:
            print(f"   📄 Extracting: {pdf_file.name}")

            # 1. extract text page-by-page
            raw_doc = extract_pdf_text(pdf_file)

            # 2. build output path (keep category structure)
            category_output_dir = OUTPUT_DIR / category_folder.name
            category_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = category_output_dir / f"{raw_doc.paper_id}.json"

            # 3. save JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(raw_doc), f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"   ❌ Failed {pdf_file.name}: {e}")