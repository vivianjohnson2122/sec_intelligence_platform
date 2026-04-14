"""
LangChain RAG chain over SEC filings stored in ChromaDB.

Retrieval is metadata-filtered so queries can be scoped to a specific
ticker, section, form type, or date range before vector similarity search.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
COLLECTION_NAME = "sec_filings"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_SYSTEM_PROMPT = PromptTemplate.from_template(
    """You are a financial analyst assistant with expertise in SEC filings.
Use the following excerpts from SEC filings to answer the question.
Be specific and cite evidence from the text. If the information is not in the context,
say so clearly rather than speculating.

Context:
{context}

Question: {question}

Answer:"""
)


class FilingRAGChain:
    """
    RAG chain for querying SEC filings with optional metadata filtering.

    Usage:
        chain = FilingRAGChain()
        result = chain.query(
            question="What are the main risks facing Apple?",
            filters={"ticker": "AAPL", "section": "risk_factors"}
        )
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = LLM_MODEL,
        n_results: int = 6,
    ) -> None:
        chroma_dir = chroma_path or CHROMA_PATH

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            client=chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            ),
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        logger.info("[RAG] Loading HuggingFace LLM '%s'…", llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        pipe = hf_pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            truncation=True,
            max_length=2048,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            return_full_text=False,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.n_results = n_results

    def _build_where_filter(
        self,
        tickers: Optional[list[str]] = None,
        sections: Optional[list[str]] = None,
        form_types: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """
        Build a ChromaDB metadata 'where' filter dict from optional parameters.

        Only equality / $in conditions are used here because ChromaDB 1.x
        requires numeric operands for $gte/$lte.  Date range filtering is
        handled in Python after retrieval (see _filter_by_date).

        Multiple conditions are AND-ed together using the $and operator.
        Single-value lists are simplified to equality checks.

        Returns None if no filters are specified.
        """
        conditions: list[dict] = []

        if tickers:
            if len(tickers) == 1:
                conditions.append({"ticker": {"$eq": tickers[0].upper()}})
            else:
                conditions.append({"ticker": {"$in": [t.upper() for t in tickers]}})

        if sections:
            if len(sections) == 1:
                conditions.append({"section": {"$eq": sections[0]}})
            else:
                conditions.append({"section": {"$in": sections}})

        if form_types:
            if len(form_types) == 1:
                conditions.append({"form_type": {"$eq": form_types[0]}})
            else:
                conditions.append({"form_type": {"$in": form_types}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _build_context(self, docs: list, max_context_tokens: int = 1200) -> str:
        """
        Concatenate doc chunks up to a token budget so the full prompt
        stays well within the model's max_length and avoids buffer overflows.
        """
        parts: list[str] = []
        used = 0
        for doc in docs:
            tokens = len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
            if used + tokens > max_context_tokens:
                break
            parts.append(doc.page_content)
            used += tokens
        return "\n\n".join(parts)

    @staticmethod
    def _filter_by_date(docs: list, date_start: Optional[str], date_end: Optional[str]) -> list:
        """Post-filter documents by filing_date string (ISO YYYY-MM-DD, lexicographic order)."""
        if not date_start and not date_end:
            return docs
        filtered = []
        for doc in docs:
            d = doc.metadata.get("filing_date", "")
            if date_start and d < date_start:
                continue
            if date_end and d > date_end:
                continue
            filtered.append(doc)
        return filtered

    def query(
        self,
        question: str,
        tickers: Optional[list[str]] = None,
        sections: Optional[list[str]] = None,
        form_types: Optional[list[str]] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> dict:
        """
        Run a RAG query against the SEC filings vector store.

        Args:
            question: Natural language question.
            tickers: Optional list of ticker symbols to filter on.
            sections: Optional list of section keys (e.g. ["mda", "risk_factors"]).
            form_types: Optional list of form types (e.g. ["10-K"]).
            date_start: Optional ISO date string (YYYY-MM-DD) — filter filings after this date.
            date_end: Optional ISO date string — filter filings before this date.
            n_results: Override default number of chunks to retrieve.

        Returns:
            dict with keys:
                "answer"  : str — the LLM's answer
                "sources" : list[dict] — each with ticker, filing_date, section,
                            accession_number, text (snippet)
                "question": str — echoed back
        """
        k = n_results or self.n_results
        where = self._build_where_filter(
            tickers=tickers,
            sections=sections,
            form_types=form_types,
        )

        search_kwargs: dict = {"k": k}
        if where:
            search_kwargs["filter"] = where

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        try:
            source_docs = retriever.invoke(question)
            source_docs = self._filter_by_date(source_docs, date_start, date_end)
            context = self._build_context(source_docs, max_context_tokens=1200)
            chain = _SYSTEM_PROMPT | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        except Exception as exc:
            logger.error("[RAG] Chain error: %s", exc)
            return {
                "answer": f"Error running query: {exc}",
                "sources": [],
                "question": question,
            }

        sources = []
        for doc in source_docs:
            meta = doc.metadata
            sources.append(
                {
                    "ticker": meta.get("ticker", ""),
                    "company_name": meta.get("company_name", ""),
                    "filing_date": meta.get("filing_date", ""),
                    "form_type": meta.get("form_type", ""),
                    "section": meta.get("section", ""),
                    "accession_number": meta.get("accession_number", ""),
                    "text": doc.page_content[:400],
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }

    def get_available_tickers(self) -> list[str]:
        """Return sorted list of unique ticker symbols in the vector store."""
        try:
            collection = self.vectorstore._collection
            if collection.count() == 0:
                return []
            results = collection.get(include=["metadatas"])
            tickers = sorted({m.get("ticker", "") for m in results["metadatas"] if m.get("ticker")})
            return tickers
        except Exception as exc:
            logger.error("[RAG] Failed to list tickers: %s", exc)
            return []

    def get_available_sections(self) -> list[str]:
        """Return sorted list of unique section keys in the vector store."""
        try:
            collection = self.vectorstore._collection
            if collection.count() == 0:
                return []
            results = collection.get(include=["metadatas"])
            sections = sorted({m.get("section", "") for m in results["metadatas"] if m.get("section")})
            return sections
        except Exception as exc:
            logger.error("[RAG] Failed to list sections: %s", exc)
            return []
