"""
Sentiment analysis for SEC filing sections using VADER with a custom financial lexicon.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
tool well-suited to short financial text. We augment the base lexicon with financial
domain terms to improve accuracy on 10-K/10-Q language.

Output:
    sentiment_scores.parquet — one row per (ticker, filing_date, section) with
    columns: pos, neu, neg, compound
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SENTIMENT_OUTPUT = Path(os.getenv("SENTIMENT_PATH", "./data/sentiment_scores.parquet"))

# Custom financial lexicon additions:
# positive = higher score (max +4), negative = lower score (min -4)
FINANCIAL_LEXICON: dict[str, float] = {
    # Strongly positive
    "outperform": 2.5,
    "beat": 2.0,
    "exceeded": 2.0,
    "robust": 1.8,
    "record": 1.5,
    "accelerating": 1.5,
    "resilient": 1.5,
    "momentum": 1.2,
    "growth": 1.0,
    "profitable": 1.5,
    "profitability": 1.5,
    "innovative": 1.2,
    "synergies": 1.0,
    "upside": 1.3,
    # Strongly negative
    "impairment": -2.5,
    "writedown": -2.5,
    "write-down": -2.5,
    "write-off": -2.5,
    "litigation": -1.8,
    "lawsuit": -1.8,
    "default": -2.5,
    "breach": -2.0,
    "headwind": -1.5,
    "headwinds": -1.5,
    "uncertainty": -1.5,
    "downturn": -1.8,
    "restructuring": -1.5,
    "layoff": -2.0,
    "layoffs": -2.0,
    "restatement": -3.0,
    "fraud": -3.5,
    "misconduct": -2.8,
    "decline": -1.5,
    "declining": -1.5,
    "deterioration": -2.0,
    "adverse": -1.5,
    "unfavorable": -1.5,
    "volatile": -1.3,
    "volatility": -1.3,
    "risk": -1.0,
    "risks": -1.0,
    "material weakness": -3.0,
    "going concern": -3.5,
    "liquidity": -0.5,
    "inflationary": -1.2,
    "macroeconomic": -0.5,
}


class FilingSentimentAnalyzer:
    """
    Score SEC filing sections using VADER augmented with a financial lexicon.

    Usage:
        analyzer = FilingSentimentAnalyzer()
        scores = analyzer.score_text("Revenue grew 15% driven by strong product demand.")
        # scores = {"pos": 0.23, "neu": 0.77, "neg": 0.0, "compound": 0.65}
    """

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()
        # Inject custom financial terms into VADER's lexicon
        self.vader.lexicon.update(FINANCIAL_LEXICON)

    def score_text(self, text: str) -> dict[str, float]:
        """
        Score a text string and return VADER sentiment dict.

        Returns:
            dict with keys: pos, neu, neg, compound
            compound is in [-1, 1]; pos/neu/neg are proportions summing to ~1.
        """
        if not text or not text.strip():
            return {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
        scores = self.vader.polarity_scores(text[:10_000])
        return {
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "compound": float(scores["compound"]),
        }

    def score_sections(
        self,
        sections: dict[str, str],
        metadata: dict,
    ) -> list[dict]:
        """
        Score every section in a filing and return a list of score records.

        Args:
            sections: {section_key: text} from FilingParser.parse().
            metadata: Must include ticker, company_name, cik, form_type,
                      filing_date, accession_number.

        Returns:
            List of dicts each with: ticker, company_name, cik, form_type,
            filing_date, accession_number, section, pos, neu, neg, compound.
        """
        records = []
        for section_key, text in sections.items():
            scores = self.score_text(text)
            records.append(
                {
                    "ticker": metadata.get("ticker", ""),
                    "company_name": metadata.get("company_name", ""),
                    "cik": metadata.get("cik", ""),
                    "form_type": metadata.get("form_type", ""),
                    "filing_date": metadata.get("filing_date", ""),
                    "accession_number": metadata.get("accession_number", ""),
                    "section": section_key,
                    **scores,
                }
            )
        return records

    def build_sentiment_series(
        self,
        all_records: list[dict],
    ) -> pd.DataFrame:
        """
        Combine score records from multiple filings into a sorted DataFrame.

        Returns:
            DataFrame with columns: ticker, company_name, cik, form_type,
            filing_date, accession_number, section, pos, neu, neg, compound.
            Sorted by ticker, filing_date, section.
        """
        if not all_records:
            return pd.DataFrame(
                columns=["ticker", "company_name", "cik", "form_type", "filing_date",
                         "accession_number", "section", "pos", "neu", "neg", "compound"]
            )
        df = pd.DataFrame(all_records)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.sort_values(["ticker", "filing_date", "section"]).reset_index(drop=True)
        return df

    def save_scores(self, df: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Append new scores to the Parquet store, deduplicating by accession+section."""
        out = path or SENTIMENT_OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists():
            existing = pd.read_parquet(out)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["accession_number", "section"], keep="last"
            )
        else:
            combined = df

        combined.to_parquet(out, index=False)
        logger.info("[Sentiment] Saved %d rows to %s", len(combined), out)
        return out

    @staticmethod
    def load_scores(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Load sentiment scores from Parquet if the file exists."""
        p = path or SENTIMENT_OUTPUT
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        return df
