"""
Unit tests for analytics/sentiment.py.

All tests run entirely offline — VADER uses a local lexicon file.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.sentiment import FilingSentimentAnalyzer, FINANCIAL_LEXICON


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLEARLY_POSITIVE_TEXT = (
    "The company reported record revenue and outstanding profitability. "
    "Earnings per share significantly exceeded analyst expectations. "
    "Management raised full-year guidance citing robust demand and strong momentum. "
    "Free cash flow generation was exceptional this quarter. "
    "The dividend was increased by 20 percent reflecting financial strength."
)

CLEARLY_NEGATIVE_TEXT = (
    "The company recorded a significant impairment charge and a writedown of assets. "
    "Ongoing litigation and regulatory investigations create material uncertainty. "
    "Revenue declined sharply due to headwinds in the macroeconomic environment. "
    "The going concern disclosure raises serious questions about future viability. "
    "Layoffs and restructuring charges will adversely impact near-term results."
)

NEUTRAL_TEXT = (
    "The company filed its annual report with the Securities and Exchange Commission. "
    "The fiscal year ended on December 31. "
    "Operations are conducted across the United States and certain foreign jurisdictions."
)


# ---------------------------------------------------------------------------
# FilingSentimentAnalyzer.score_text() tests
# ---------------------------------------------------------------------------

class TestScoreText:
    """Tests for the core score_text() method."""

    def setup_method(self) -> None:
        self.analyzer = FilingSentimentAnalyzer()

    def test_returns_dict_with_required_keys(self) -> None:
        """score_text() must return a dict with pos, neu, neg, compound."""
        result = self.analyzer.score_text("Revenue grew this quarter.")
        for key in ("pos", "neu", "neg", "compound"):
            assert key in result, f"Missing key: {key}"

    def test_compound_in_valid_range(self) -> None:
        """Compound score must always be in [-1, 1]."""
        texts = [CLEARLY_POSITIVE_TEXT, CLEARLY_NEGATIVE_TEXT, NEUTRAL_TEXT, ""]
        for text in texts:
            result = self.analyzer.score_text(text)
            assert -1.0 <= result["compound"] <= 1.0, (
                f"Compound {result['compound']} out of range for: '{text[:50]}'"
            )

    def test_clearly_positive_text_scores_positive(self) -> None:
        """Positive financial language should yield a positive compound score."""
        result = self.analyzer.score_text(CLEARLY_POSITIVE_TEXT)
        assert result["compound"] > 0, (
            f"Expected positive compound for positive text, got {result['compound']}"
        )

    def test_clearly_negative_text_scores_negative(self) -> None:
        """Negative financial language should yield a negative compound score."""
        result = self.analyzer.score_text(CLEARLY_NEGATIVE_TEXT)
        assert result["compound"] < 0, (
            f"Expected negative compound for negative text, got {result['compound']}"
        )

    def test_positive_stronger_than_negative(self) -> None:
        """Positive text should score higher than negative text."""
        pos_score = self.analyzer.score_text(CLEARLY_POSITIVE_TEXT)["compound"]
        neg_score = self.analyzer.score_text(CLEARLY_NEGATIVE_TEXT)["compound"]
        assert pos_score > neg_score

    def test_empty_string_returns_neutral(self) -> None:
        """Empty input should return a neutral score without raising."""
        result = self.analyzer.score_text("")
        assert result["compound"] == 0.0
        assert result["neu"] == 1.0

    def test_proportions_sum_to_one(self) -> None:
        """pos + neu + neg should sum to ~1.0 for non-trivial input."""
        result = self.analyzer.score_text(CLEARLY_POSITIVE_TEXT)
        total = result["pos"] + result["neu"] + result["neg"]
        assert abs(total - 1.0) < 0.01, f"pos+neu+neg = {total}, expected ~1.0"

    def test_financial_lexicon_applied(self) -> None:
        """Custom financial terms should influence the compound score."""
        # "impairment" is in the custom lexicon as -2.5
        result_impairment = self.analyzer.score_text("The company recorded an impairment.")
        result_neutral = self.analyzer.score_text("The company recorded a transaction.")
        assert result_impairment["compound"] < result_neutral["compound"], (
            "Custom lexicon term 'impairment' should reduce compound score"
        )

    def test_going_concern_very_negative(self) -> None:
        """'going concern' is a strong negative signal and should score very negatively."""
        result = self.analyzer.score_text(
            "Management has raised substantial doubt about the company's ability to continue "
            "as a going concern."
        )
        assert result["compound"] < -0.3

    def test_long_text_does_not_crash(self) -> None:
        """Texts over 10,000 chars should be handled without errors."""
        long_text = CLEARLY_POSITIVE_TEXT * 300  # ~150k chars
        result = self.analyzer.score_text(long_text)
        assert -1.0 <= result["compound"] <= 1.0


# ---------------------------------------------------------------------------
# FilingSentimentAnalyzer.score_sections() tests
# ---------------------------------------------------------------------------

class TestScoreSections:
    """Tests for score_sections() method."""

    def setup_method(self) -> None:
        self.analyzer = FilingSentimentAnalyzer()

    def test_returns_one_record_per_section(self) -> None:
        sections = {
            "mda": CLEARLY_POSITIVE_TEXT,
            "risk_factors": CLEARLY_NEGATIVE_TEXT,
            "business": NEUTRAL_TEXT,
        }
        metadata = {
            "ticker": "TEST",
            "company_name": "Test Corp",
            "cik": "0000000000",
            "form_type": "10-K",
            "filing_date": "2023-01-01",
            "accession_number": "0000000000-23-000001",
        }
        records = self.analyzer.score_sections(sections, metadata)
        assert len(records) == 3

    def test_records_contain_required_fields(self) -> None:
        sections = {"mda": CLEARLY_POSITIVE_TEXT}
        metadata = {
            "ticker": "AAPL", "company_name": "Apple Inc.", "cik": "0000320193",
            "form_type": "10-K", "filing_date": "2023-10-27",
            "accession_number": "0000320193-23-000077",
        }
        records = self.analyzer.score_sections(sections, metadata)
        required_fields = {
            "ticker", "company_name", "form_type", "filing_date",
            "section", "pos", "neu", "neg", "compound",
        }
        for field in required_fields:
            assert field in records[0], f"Missing field: {field}"

    def test_section_key_preserved_in_record(self) -> None:
        sections = {"risk_factors": CLEARLY_NEGATIVE_TEXT}
        metadata = {"ticker": "X", "company_name": "", "cik": "", "form_type": "10-K",
                    "filing_date": "2023-01-01", "accession_number": "X-001"}
        records = self.analyzer.score_sections(sections, metadata)
        assert records[0]["section"] == "risk_factors"


# ---------------------------------------------------------------------------
# FilingSentimentAnalyzer.build_sentiment_series() tests
# ---------------------------------------------------------------------------

class TestBuildSentimentSeries:
    """Tests for build_sentiment_series() method."""

    def setup_method(self) -> None:
        self.analyzer = FilingSentimentAnalyzer()

    def _make_records(self) -> list[dict]:
        base = {
            "company_name": "Test Corp", "cik": "0000000001",
            "form_type": "10-K", "accession_number": "0000000001-23-000001",
        }
        return [
            {**base, "ticker": "AAA", "filing_date": "2022-12-31",
             "section": "mda", "pos": 0.2, "neu": 0.7, "neg": 0.1, "compound": 0.45},
            {**base, "ticker": "AAA", "filing_date": "2021-12-31",
             "section": "risk_factors", "pos": 0.05, "neu": 0.5, "neg": 0.45, "compound": -0.6},
            {**base, "ticker": "BBB", "filing_date": "2022-06-30",
             "section": "mda", "pos": 0.1, "neu": 0.8, "neg": 0.1, "compound": 0.0},
        ]

    def test_returns_dataframe(self) -> None:
        df = self.analyzer.build_sentiment_series(self._make_records())
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self) -> None:
        df = self.analyzer.build_sentiment_series(self._make_records())
        assert len(df) == 3

    def test_required_columns_present(self) -> None:
        df = self.analyzer.build_sentiment_series(self._make_records())
        for col in ["ticker", "filing_date", "section", "compound", "pos", "neg", "neu"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_filing_date_parsed_as_datetime(self) -> None:
        df = self.analyzer.build_sentiment_series(self._make_records())
        assert pd.api.types.is_datetime64_any_dtype(df["filing_date"]), (
            "filing_date should be datetime dtype"
        )

    def test_empty_records_returns_empty_dataframe(self) -> None:
        df = self.analyzer.build_sentiment_series([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "compound" in df.columns

    def test_sorted_by_ticker_and_date(self) -> None:
        df = self.analyzer.build_sentiment_series(self._make_records())
        # Verify the dataframe is sorted: AAA 2021 should come before AAA 2022
        aaas = df[df["ticker"] == "AAA"].reset_index(drop=True)
        assert aaas.loc[0, "filing_date"] < aaas.loc[1, "filing_date"]


# ---------------------------------------------------------------------------
# Custom lexicon sanity checks
# ---------------------------------------------------------------------------

class TestFinancialLexicon:
    """Sanity checks on the custom financial lexicon."""

    def test_impairment_is_negative(self) -> None:
        assert FINANCIAL_LEXICON["impairment"] < 0

    def test_growth_is_positive(self) -> None:
        assert FINANCIAL_LEXICON["growth"] > 0

    def test_going_concern_is_very_negative(self) -> None:
        assert FINANCIAL_LEXICON["going concern"] < -2.0

    def test_no_zero_values(self) -> None:
        """Lexicon entries should not be 0 — that would be neutral and pointless."""
        for term, score in FINANCIAL_LEXICON.items():
            assert score != 0.0, f"'{term}' has zero score"
