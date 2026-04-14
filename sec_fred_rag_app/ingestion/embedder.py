"""
Chunk SEC filing sections, generate local embeddings via sentence-transformers,
and upsert into ChromaDB.

Vector store schema per chunk:
    id          : "{ticker}_{accession}_{section}_{chunk_index}"
    embedding   : all-MiniLM-L6-v2 (384-dim)
    document    : raw chunk text
    metadata    : ticker, company_name, cik, form_type, filing_date,
                  section, chunk_index, accession_number
"""

import logging
import hashlib
from typing import Optional
from pathlib import Path
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from ingestion.filing_parser import chunk_text

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
COLLECTION_NAME = "sec_filings"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64


class FilingEmbedder:
    """
    Embed SEC filing sections and store vectors in a persistent ChromaDB collection.

    Usage:
        embedder = FilingEmbedder()
        embedder.embed_filing(
            sections={"mda": "...", "risk_factors": "..."},
            metadata={"ticker": "AAPL", "filing_date": "2023-10-27", ...}
        )
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        logger.info("[Embedder] Loading sentence-transformers model '%s'…", EMBED_MODEL)
        self.model = SentenceTransformer(EMBED_MODEL)

        chroma_dir = chroma_path or CHROMA_PATH
        chroma_dir.mkdir(parents=True, exist_ok=True)

        self.chroma = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[Embedder] ChromaDB at %s, collection '%s' (%d docs)",
            chroma_dir,
            collection_name,
            self.collection.count(),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts locally using sentence-transformers."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            logger.debug("[Embedder] Embedding batch %d/%d", i // EMBED_BATCH_SIZE + 1,
                         -(-len(texts) // EMBED_BATCH_SIZE))
            vecs = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def _make_chunk_id(
        self,
        ticker: str,
        accession_number: str,
        section: str,
        chunk_index: int,
    ) -> str:
        """Create a deterministic, unique ID for a chunk."""
        raw = f"{ticker}_{accession_number}_{section}_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def embed_filing(
        self,
        sections: dict[str, str],
        metadata: dict,
        chunk_size: int = 800,
        overlap_sentences: int = 2,
    ) -> int:
        """
        Chunk and embed all sections of a single filing, upsert into ChromaDB.

        Args:
            sections: {section_key: text} dict from FilingParser.
            metadata: Must include ticker, company_name, cik, form_type,
                      filing_date, accession_number.

        Returns:
            Total number of chunks upserted.
        """
        ticker = metadata.get("ticker", "UNKNOWN").upper()
        accession = metadata.get("accession_number", "")

        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []

        for section_key, text in sections.items():
            chunks = chunk_text(text, chunk_size_chars=chunk_size,
                                overlap_sentences=overlap_sentences)
            for idx, chunk in enumerate(chunks):
                chunk_id = self._make_chunk_id(ticker, accession, section_key, idx)
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metas.append(
                    {
                        "ticker": ticker,
                        "company_name": metadata.get("company_name", ""),
                        "cik": metadata.get("cik", ""),
                        "form_type": metadata.get("form_type", ""),
                        "filing_date": metadata.get("filing_date", ""),
                        "section": section_key,
                        "chunk_index": idx,
                        "accession_number": accession,
                    }
                )

        if not all_docs:
            logger.warning("[Embedder] No chunks produced for %s %s", ticker, accession)
            return 0

        logger.info(
            "[Embedder] Embedding %d chunks for %s (%s)…", len(all_docs), ticker, accession
        )
        embeddings = self.embed_texts(all_docs)

        self.collection.upsert(
            ids=all_ids,
            documents=all_docs,
            embeddings=embeddings,
            metadatas=all_metas,
        )
        logger.info("[Embedder] Upserted %d chunks for %s %s", len(all_docs), ticker, accession)
        return len(all_docs)

    def get_collection_stats(self) -> dict:
        """Return basic stats about the current ChromaDB collection."""
        count = self.collection.count()
        return {"collection": COLLECTION_NAME, "total_chunks": count}

    def list_tickers(self) -> list[str]:
        """Return list of unique tickers stored in the collection."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        tickers = sorted({m.get("ticker", "") for m in results["metadatas"] if m.get("ticker")})
        return tickers

    def get_filing_summary(self) -> list[dict]:
        """
        Return one summary dict per (ticker, accession_number) pair,
        including filing_date, form_type, sections found, and chunk count.
        """
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        summary: dict[str, dict] = {}
        for meta in results["metadatas"]:
            key = f"{meta.get('ticker')}_{meta.get('accession_number')}"
            if key not in summary:
                summary[key] = {
                    "ticker": meta.get("ticker"),
                    "company_name": meta.get("company_name"),
                    "filing_date": meta.get("filing_date"),
                    "form_type": meta.get("form_type"),
                    "accession_number": meta.get("accession_number"),
                    "sections": set(),
                    "chunk_count": 0,
                }
            summary[key]["sections"].add(meta.get("section", ""))
            summary[key]["chunk_count"] += 1

        rows = []
        for v in summary.values():
            v["sections"] = sorted(v["sections"])
            rows.append(v)
        rows.sort(key=lambda x: (x["ticker"], x["filing_date"]), reverse=True)
        return rows
