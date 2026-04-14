"""
Unit tests for ingestion/filing_parser.py.

All tests run entirely offline — no API calls are made.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.filing_parser import FilingParser, chunk_text


# ---------------------------------------------------------------------------
# Fixtures / shared test data
# ---------------------------------------------------------------------------

MOCK_10K_HTML = """
<html>
<head><title>Annual Report</title></head>
<body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 10-K</p>
<p>ACME CORPORATION</p>

<h2>Item 1. Business</h2>
<p>Acme Corporation is a leading provider of innovative products and services.
The company was founded in 1985 and operates across 40 countries worldwide.
Our primary business segment is consumer electronics, generating 65% of revenue.
We employ approximately 50,000 people globally and have 12 manufacturing facilities.</p>

<h2>Item 1A. Risk Factors</h2>
<p>The following risks could materially adversely affect our business, financial
condition, results of operations and the trading price of our common stock.
We face intense competition in all markets in which we operate.
Our products may become obsolete due to rapid technological change.
We are exposed to significant foreign currency risk.
Supply chain disruptions could adversely impact our ability to deliver products.
Regulatory changes may increase our compliance costs substantially.
We may be subject to litigation and legal proceedings from time to time.</p>

<h2>Item 7. Management Discussion and Analysis</h2>
<p>Revenue for the fiscal year ended December 31 was $42.5 billion, representing
a 12% increase compared to the prior year. Operating income grew 18% to $8.3 billion
driven by margin expansion in our software segment and disciplined cost management.
Free cash flow generation remained robust at $6.7 billion, supporting continued
investment in research and development as well as shareholder return programs.</p>

<h2>Item 8. Financial Statements</h2>
<p>See consolidated financial statements included elsewhere in this Annual Report.
Total assets as of December 31 were $125.4 billion. Total stockholders equity was
$67.8 billion. Net income attributable to the company was $5.2 billion.</p>
</body>
</html>
"""

MOCK_UPPERCASE_HTML = """
<html><body>
<p>FORM 10-K ANNUAL REPORT</p>
<p>SOME INTRODUCTION TEXT THAT DOES NOT CONTAIN LOWERCASE HEADERS</p>

ITEM 1 BUSINESS

This corporation manufactures and sells industrial equipment across North America.
The company has operated since 1972 and holds 15% market share in its primary segment.
Annual revenues exceed two billion dollars with operations in eight states.

ITEM 1A RISK FACTORS

The company faces the following material risks. Competition from foreign manufacturers
may erode pricing power. Raw material costs have been volatile and may increase.
Key customer concentration represents a significant business risk factor.
Regulatory compliance requirements continue to increase operational complexity.

ITEM 7 MANAGEMENT DISCUSSION AND ANALYSIS

Operating performance improved this fiscal year driven by pricing actions and volume
growth in the midwest region. The company successfully renegotiated three major contracts.
Capital expenditures were held below budget through disciplined project management.
</body></html>
"""

MINIMAL_HTML = "<html><body><p>Short text without sections.</p></body></html>"


# ---------------------------------------------------------------------------
# FilingParser.parse() tests
# ---------------------------------------------------------------------------

class TestFilingParser:
    """Tests for FilingParser section extraction."""

    def setup_method(self) -> None:
        self.parser = FilingParser()

    def test_extracts_mda_section(self) -> None:
        """MD&A section should be extracted from standard 10-K HTML."""
        sections = self.parser.parse(MOCK_10K_HTML, form_type="10-K")
        assert "mda" in sections, "Expected 'mda' key in parsed sections"
        assert len(sections["mda"]) > 100
        # Content check: mentions revenue growth
        assert "revenue" in sections["mda"].lower()

    def test_extracts_risk_factors_section(self) -> None:
        """Risk Factors section should be extracted and contain risk language."""
        sections = self.parser.parse(MOCK_10K_HTML, form_type="10-K")
        assert "risk_factors" in sections, "Expected 'risk_factors' key"
        text = sections["risk_factors"].lower()
        assert "risk" in text or "adverse" in text

    def test_extracts_multiple_sections(self) -> None:
        """A well-formed 10-K should yield at least 3 named sections."""
        sections = self.parser.parse(MOCK_10K_HTML, form_type="10-K")
        assert len(sections) >= 3, f"Expected ≥3 sections, got {len(sections)}: {list(sections)}"

    def test_uppercase_fallback_triggered(self) -> None:
        """Parser should fall back to uppercase pattern matching when needed."""
        sections = self.parser.parse(MOCK_UPPERCASE_HTML, form_type="10-K",
                                     accession_number="TEST-UPPERCASE-001")
        # Should still find risk_factors and mda via uppercase fallback
        found_keys = set(sections.keys())
        assert found_keys & {"risk_factors", "mda", "business"}, (
            f"Expected at least one named section via fallback, got: {found_keys}"
        )

    def test_empty_html_returns_fallback_section(self) -> None:
        """Minimal/empty HTML should return a full_text fallback rather than crashing."""
        sections = self.parser.parse(MINIMAL_HTML, form_type="10-K",
                                     accession_number="TEST-EMPTY-001")
        assert sections, "Expected at least one section even for empty input"
        # full_text fallback or a section_ key
        assert any(
            k in ("full_text",) or k.startswith("section_")
            for k in sections
        ), f"Unexpected section keys: {list(sections)}"

    def test_section_text_not_too_short(self) -> None:
        """Each extracted section should have meaningful content (>200 chars)."""
        sections = self.parser.parse(MOCK_10K_HTML, form_type="10-K")
        for key, text in sections.items():
            assert len(text) > 200, f"Section '{key}' is too short ({len(text)} chars)"

    def test_html_to_text_strips_tags(self) -> None:
        """html_to_text() should remove all HTML tags."""
        raw = "<p>Hello <b>world</b></p><script>bad()</script>"
        text = self.parser.html_to_text(raw)
        assert "<" not in text
        assert "bad()" not in text
        assert "Hello" in text
        assert "world" in text


# ---------------------------------------------------------------------------
# chunk_text() tests
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for the chunk_text utility function."""

    _SAMPLE = (
        "The company reported record revenue this quarter. "
        "Earnings per share exceeded analyst expectations by 15 percent. "
        "Management raised full-year guidance citing strong demand. "
        "International sales grew 22 percent year-over-year. "
        "The board approved a new share repurchase programme worth two billion dollars. "
        "Operating margins expanded by 120 basis points due to cost discipline. "
        "Research and development spending increased to support new product launches. "
        "Capital expenditures are expected to remain below prior-year levels. "
        "The company maintains a strong balance sheet with minimal leverage. "
        "Free cash flow conversion remains above 90 percent of net income. "
    )

    def test_produces_chunks(self) -> None:
        """chunk_text should return at least one chunk for non-trivial input."""
        chunks = chunk_text(self._SAMPLE * 5, chunk_size_chars=400)
        assert len(chunks) >= 1

    def test_chunks_not_empty(self) -> None:
        """No chunk should be empty or whitespace-only."""
        chunks = chunk_text(self._SAMPLE * 3, chunk_size_chars=300)
        for c in chunks:
            assert c.strip(), "Found empty chunk"

    def test_chunks_honour_size_limit(self) -> None:
        """Chunks should generally stay within 2× the requested size."""
        size = 300
        chunks = chunk_text(self._SAMPLE * 5, chunk_size_chars=size)
        for c in chunks:
            # Allow 2× headroom for sentences that exceed the limit on their own
            assert len(c) < size * 2, f"Chunk too large: {len(c)} chars"

    def test_overlap_creates_repeated_content(self) -> None:
        """With overlap > 0 there should be some repeated text across consecutive chunks."""
        # Build a text long enough to produce 3+ chunks
        long_text = self._SAMPLE * 10
        chunks = chunk_text(long_text, chunk_size_chars=400, overlap_sentences=2)
        assert len(chunks) >= 3
        # Last sentence of chunk N should appear somewhere in chunk N+1
        # (not guaranteed for every pair, but should be true for at least one)
        found_overlap = False
        for i in range(len(chunks) - 1):
            last_sentence = chunks[i].split(". ")[-1].strip()
            if last_sentence and last_sentence[:30] in chunks[i + 1]:
                found_overlap = True
                break
        assert found_overlap, "Expected some overlap between consecutive chunks"

    def test_short_text_returns_single_chunk(self) -> None:
        """Very short text should come back as a single chunk."""
        short = "Revenue grew strongly this period."
        chunks = chunk_text(short)
        assert len(chunks) == 1

    def test_minimum_length_filter(self) -> None:
        """Chunks shorter than 50 chars should be filtered out."""
        tiny = "Hi. Ok. Yes. No."
        chunks = chunk_text(tiny, chunk_size_chars=800)
        for c in chunks:
            assert len(c) >= 50 or not chunks, f"Chunk below min length: '{c}'"
