"""
Parse raw SEC filing HTML into structured named sections.

10-K filings follow a standard structure:
    Item 1   - Business
    Item 1A  - Risk Factors
    Item 7   - Management's Discussion and Analysis (MD&A)
    Item 7A  - Quantitative Disclosures About Market Risk
    Item 8   - Financial Statements

We extract these sections by searching for their standard headers,
giving us semantically meaningful chunks for RAG retrieval.

Fallback strategy (applied when fewer than 2 sections are found):
    1. Search for uppercase "ITEM 1A" style headers via regex
    2. Split on <hr> tags or repeated dash lines
"""

import re
import logging
from typing import Optional
from bs4 import BeautifulSoup
from spacy.lang.en import English

logger = logging.getLogger(__name__)

# (section_key, display_name, [header_patterns]) — for 10-K
SECTION_PATTERNS_10K: list[tuple[str, str, list[str]]] = [
    (
        "business",
        "Business Overview",
        [r"item\s*1\b(?!a)", r"description of business"],
    ),
    (
        "risk_factors",
        "Risk Factors",
        [r"item\s*1a", r"risk factors"],
    ),
    (
        "mda",
        "Management's Discussion & Analysis",
        [r"item\s*7\b(?!a)", r"management.{0,10}discussion", r"md&a"],
    ),
    (
        "market_risk",
        "Quantitative Market Risk Disclosures",
        [r"item\s*7a", r"quantitative.*market risk"],
    ),
    (
        "financials",
        "Financial Statements",
        [r"item\s*8\b", r"financial statements"],
    ),
]

# Patterns for 10-Q (items differ from 10-K)
SECTION_PATTERNS_10Q: list[tuple[str, str, list[str]]] = [
    (
        "mda",
        "Management's Discussion & Analysis",
        [r"item\s*2\b", r"management.{0,10}discussion"],
    ),
    (
        "market_risk",
        "Quantitative Market Risk Disclosures",
        [r"item\s*3\b", r"quantitative.*market risk"],
    ),
    (
        "financials",
        "Financial Statements",
        [r"item\s*1\b(?!a)", r"financial statements"],
    ),
    (
        "risk_factors",
        "Risk Factors",
        [r"item\s*1a", r"risk factors"],
    ),
]

# Uppercase fallback patterns: match "ITEM 1A" / "ITEM 7" style headers
_UPPERCASE_FALLBACK_PATTERNS: list[tuple[str, str, str]] = [
    ("business",     "Business Overview",                    r"ITEM\s+1\b(?!A)"),
    ("risk_factors", "Risk Factors",                         r"ITEM\s+1A"),
    ("mda",          "Management's Discussion & Analysis",   r"ITEM\s+7\b(?!A)"),
    ("market_risk",  "Quantitative Market Risk Disclosures", r"ITEM\s+7A"),
    ("financials",   "Financial Statements",                 r"ITEM\s+8\b"),
]


class FilingParser:
    """
    Parse a raw SEC filing (HTML string) into named sections.

    Usage:
        parser = FilingParser()
        sections = parser.parse(html_text, form_type="10-K",
                                accession_number="0000320193-23-000077")
        # sections = {"mda": "...", "risk_factors": "...", ...}
    """

    def html_to_text(self, html: str) -> str:
        """Strip HTML tags, remove script/style/head, and normalize whitespace."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_positions(
        self,
        text: str,
        patterns: list[tuple[str, str, list[str]]],
        case_sensitive: bool = False,
    ) -> list[tuple[int, str, str]]:
        """
        Search *text* for section header patterns and return
        sorted list of (position, section_key, display_name).
        """
        search_text = text if case_sensitive else text.lower()
        found: list[tuple[int, str, str]] = []

        for section_key, display_name, header_patterns in patterns:
            for pattern in header_patterns:
                flags = 0 if case_sensitive else re.IGNORECASE
                matches = list(re.finditer(pattern, search_text, flags))
                if matches:
                    pos = matches[-1].start()
                    found.append((pos, section_key, display_name))
                    break

        found.sort(key=lambda x: x[0])
        return found

    def _positions_to_sections(
        self,
        text: str,
        found_positions: list[tuple[int, str, str]],
        max_section_chars: int = 30_000,
    ) -> dict[str, str]:
        """Slice text between consecutive found positions into sections dict."""
        sections: dict[str, str] = {}
        for i, (pos, key, name) in enumerate(found_positions):
            end = found_positions[i + 1][0] if i + 1 < len(found_positions) else pos + max_section_chars
            section_text = text[pos:end].strip()
            if len(section_text) > 200:
                sections[key] = section_text
                logger.debug("[Parser] Extracted '%s': %d chars", name, len(section_text))
        return sections

    def _uppercase_fallback(
        self,
        text: str,
        accession_number: str,
    ) -> dict[str, str]:
        """
        Fallback 1: Search for uppercase 'ITEM 1A' style headers
        (some filings use all-caps headers instead of sentence case).
        """
        logger.warning(
            "[Parser] Using uppercase header fallback for accession %s", accession_number
        )
        patterns_for_search = [
            (key, name, [pat]) for key, name, pat in _UPPERCASE_FALLBACK_PATTERNS
        ]
        found = self._find_positions(text, patterns_for_search, case_sensitive=True)
        return self._positions_to_sections(text, found)

    def _hr_split_fallback(
        self,
        html: str,
        accession_number: str,
    ) -> dict[str, str]:
        """
        Fallback 2: Split the original HTML on <hr> tags or lines of repeated dashes,
        then assign generic section keys based on order.
        """
        logger.warning(
            "[Parser] Using <hr>/dash fallback for accession %s", accession_number
        )
        soup = BeautifulSoup(html, "lxml")

        # Try <hr> boundaries first
        hr_tags = soup.find_all("hr")
        if len(hr_tags) >= 2:
            parts: list[str] = []
            prev_end = 0
            html_str = str(soup)
            for hr in hr_tags:
                hr_pos = html_str.find(str(hr), prev_end)
                if hr_pos == -1:
                    continue
                chunk_html = html_str[prev_end:hr_pos]
                chunk_text = BeautifulSoup(chunk_html, "lxml").get_text(separator=" ").strip()
                if len(chunk_text) > 300:
                    parts.append(chunk_text)
                prev_end = hr_pos + len(str(hr))
        else:
            # Fall back to splitting clean text on lines of 10+ dashes
            clean = self.html_to_text(html)
            parts = re.split(r"-{10,}", clean)

        sections: dict[str, str] = {}
        for idx, part in enumerate(parts[:8]):
            part = part.strip()
            if len(part) > 200:
                sections[f"section_{idx + 1}"] = part[:30_000]
        return sections

    def extract_sections(
        self,
        text: str,
        patterns: list[tuple[str, str, list[str]]],
    ) -> dict[str, str]:
        """
        Find section boundaries by searching for header patterns,
        then slice the text between consecutive headers.
        """
        found = self._find_positions(text, patterns)
        if not found:
            return {}
        return self._positions_to_sections(text, found)

    def parse(
        self,
        raw_html: str,
        form_type: str = "10-K",
        accession_number: str = "unknown",
    ) -> dict[str, str]:
        """
        Parse raw HTML into {section_key: cleaned_text}.

        Applies a three-tier fallback strategy:
            1. Primary: lowercase pattern search
            2. Fallback 1: uppercase 'ITEM N' regex search
            3. Fallback 2: <hr> tag or dash-line splits
            4. Last resort: return full text (capped at 50k chars)
        """
        clean_text = self.html_to_text(raw_html)
        patterns = SECTION_PATTERNS_10Q if form_type == "10-Q" else SECTION_PATTERNS_10K

        # Primary extraction
        sections = self.extract_sections(clean_text, patterns)

        # Fallback 1: uppercase headers
        if len(sections) < 2:
            sections = self._uppercase_fallback(clean_text, accession_number)

        # Fallback 2: <hr>/dash split on the original HTML
        if len(sections) < 2:
            sections = self._hr_split_fallback(raw_html, accession_number)

        # Last resort: return full text
        if not sections:
            logger.warning(
                "[Parser] All extraction methods failed for accession %s; using full text",
                accession_number,
            )
            sections = {"full_text": clean_text[:50_000]}

        return sections


def chunk_text(
    text: str,
    chunk_size_chars: int = 800,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Split text into overlapping chunks suitable for embedding.

    Uses spaCy sentencizer for sentence boundary detection.
    Sentence-level overlap preserves cross-sentence context.

    Args:
        text: Input text string.
        chunk_size_chars: Target character count per chunk.
        overlap_sentences: Number of trailing sentences to carry
                           into the next chunk as overlap.

    Returns:
        List of non-empty text chunk strings (each > 50 chars).
    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    sentences = [str(s) for s in nlp(text).sents]

    chunks: list[str] = []
    curr_chunk: list[str] = []
    curr_len: int = 0

    for sentence in sentences:
        s_len = len(sentence)
        if curr_len + s_len > chunk_size_chars and curr_chunk:
            chunks.append(" ".join(curr_chunk))
            # carry overlap into next chunk (fixed: was missing slice colon)
            curr_chunk = curr_chunk[-overlap_sentences:]
            curr_len = sum(len(s) for s in curr_chunk)
        curr_chunk.append(sentence)
        curr_len += s_len

    if curr_chunk:
        chunks.append(" ".join(curr_chunk))

    return [c for c in chunks if len(c) > 50]
