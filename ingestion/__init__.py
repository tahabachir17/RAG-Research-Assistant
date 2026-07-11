"""Ingestion package for research paper discovery and PDF ingestion."""

from .arxiv_scraper import ArxivScraper, Paper
from .citation_extractor import Citation, extract_citations
from .data_cleaner import clean_sections, clean_text
from .metadata_extractor import PaperMeta, extract_metadata
from .paper_discovery import (
    AlphaXivMCPPaperDiscovery,
    AlphaXivThenArxivDiscovery,
    ArxivPaperDiscovery,
    FeymanPaperDiscovery,
    PaperDiscovery,
    ResearchAPIPaperDiscovery,
    build_discovery,
    dedupe_papers,
    score_papers,
)
from .pdf_downloader import DownloadResult, PDFDownloader
from .pdf_parser import PDFParser, RawDocument, RawPage, extract_pdf_text
from .pipeline import IngestionPipeline, IngestionResult, run_ingestion
from .section_detector import SectionDetector, SectionedDoc, detect_section, extract_sections

__all__ = [
    "AlphaXivMCPPaperDiscovery",
    "AlphaXivThenArxivDiscovery",
    "ArxivPaperDiscovery",
    "ArxivScraper",
    "Citation",
    "DownloadResult",
    "FeymanPaperDiscovery",
    "IngestionPipeline",
    "IngestionResult",
    "PDFDownloader",
    "PDFParser",
    "Paper",
    "PaperDiscovery",
    "PaperMeta",
    "RawDocument",
    "RawPage",
    "ResearchAPIPaperDiscovery",
    "SectionDetector",
    "SectionedDoc",
    "build_discovery",
    "clean_sections",
    "clean_text",
    "dedupe_papers",
    "detect_section",
    "extract_citations",
    "extract_metadata",
    "extract_pdf_text",
    "extract_sections",
    "run_ingestion",
    "score_papers",
]