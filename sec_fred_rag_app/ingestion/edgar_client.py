"""
Wrapper around the SEC EDGAR REST API.

Key Endpoints Used:
    - /files/company_tickers.json  -> ticker -> CIK map
    - /submissions/CIK{cik}.json   -> company metadata + filing history
    - /Archives/edgar/data         -> actual filing documents

The get_filing_text() method correctly resolves the index page to the
primary document, avoiding -index.htm pages.
"""

import os
import re
import time
import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
EDGAR_WWW = "https://www.sec.gov"
RATE_LIMIT_DELAY = 0.11  # SEC asks for max 10 req/sec

_user_agent = os.getenv("SEC_USER_AGENT", "research-tool contact@example.com")
HEADERS = {"User-Agent": _user_agent}


class EdgarClient:
    """
    Fetch company filings from SEC EDGAR.

    Usage:
        client = EdgarClient()
        cik = client.get_cik("AAPL")
        filings = client.get_filings(cik, form_type="10-K", limit=5)
        text = client.get_filing_text(filings[0])
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get_raw(self, url: str, timeout: int = 30) -> requests.Response:
        """Rate-limited GET, returns raw Response object."""
        time.sleep(RATE_LIMIT_DELAY)
        resp = self.session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _get(self, url: str) -> dict:
        """Rate-limited GET that returns parsed JSON."""
        return self._get_raw(url).json()

    def get_cik(self, ticker: str) -> str:
        """
        Resolve a ticker symbol to a zero-padded 10-digit CIK string.

        Example: "AAPL" -> "0000320193"
        """
        ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(ticker_map_url)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry["ticker"].upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.info("[EDGAR] %s -> CIK %s (%s)", ticker_upper, cik, entry["title"])
                return cik
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list")

    def get_company_info(self, cik: str) -> dict:
        """Return a dict of company metadata: name, SIC code, state, ticker."""
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self._get(url)
        return {
            "cik": cik,
            "name": data.get("name"),
            "ticker": data.get("tickers", [None])[0],
            "sic": data.get("sic"),
            "sic_description": data.get("sicDescription"),
            "state": data.get("stateOfIncorporation"),
        }

    def get_filings(
        self,
        cik: str,
        form_type: str = "10-K",
        limit: int = 5,
    ) -> list[dict]:
        """
        Return a list of filing metadata dicts for a given form type.

        Each dict contains:
            accession_number, filing_date, form_type,
            primary_document, document_url, index_url
        """
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self._get(url)

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type:
                accession_clean = accessions[i].replace("-", "")
                doc = primary_docs[i]
                cik_stripped = cik.lstrip("0")
                results.append(
                    {
                        "cik": cik,
                        "accession_number": accessions[i],
                        "filing_date": dates[i],
                        "form_type": form,
                        "primary_document": doc,
                        "document_url": (
                            f"{EDGAR_ARCHIVE}/{cik_stripped}/{accession_clean}/{doc}"
                        ),
                        "index_url": (
                            f"{EDGAR_ARCHIVE}/{cik_stripped}/{accession_clean}/"
                            f"{accessions[i]}-index.htm"
                        ),
                    }
                )
                if len(results) >= limit:
                    break

        logger.info("[EDGAR] Found %d %s filings for CIK %s", len(results), form_type, cik)
        return results

    def _build_document_url(self, filing: dict, filename: str) -> str:
        """Build a full EDGAR archive URL for a document filename."""
        cik = filing["cik"].lstrip("0")
        acc_clean = filing["accession_number"].replace("-", "")
        return f"{EDGAR_ARCHIVE}/{cik}/{acc_clean}/{filename}"

    def _is_submission_wrapper(self, text: str) -> bool:
        """True when the payload is the SEC full-submission wrapper, not the 10-K/10-Q HTML."""
        sample = text.lstrip()[:4000]
        return (
            sample.startswith("<SEC-DOCUMENT>")
            or sample.startswith("<SUBMISSION>")
            or (
                "<SEC-HEADER>" in sample
                and "<DOCUMENT>" in sample
                and "<FILENAME>" in sample
                and "Document 1 - file:" in sample
            )
        )

    def _extract_primary_filename(self, wrapper_text: str, form_type: str) -> Optional[str]:
        """Parse the primary HTML filename out of a SEC submission wrapper."""
        form_pattern = rf"<TYPE>\s*{re.escape(form_type)}\s*\n.*?<FILENAME>\s*([^\s<]+)"
        match = re.search(form_pattern, wrapper_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        for filename_match in re.finditer(
            r"<FILENAME>\s*([^\s<]+\.htm[l]?)", wrapper_text, re.IGNORECASE
        ):
            name = filename_match.group(1).strip()
            lower = name.lower()
            if re.search(r"xex\d+|^ex[-\d]", lower):
                continue
            if re.match(r"r\d+\.htm", lower):
                continue
            if "-index.htm" in lower:
                continue
            return name
        return None

    def _is_non_primary_document(self, href: str) -> bool:
        """Filter out exhibits, XBRL render pages, and filing index pages."""
        lower = href.lower()
        filename = lower.rsplit("/", 1)[-1]

        if "-index.htm" in lower or "-index.html" in lower:
            return True
        if lower.endswith(".txt"):
            return True
        if re.search(r"xex\d+|^ex[-\d]", filename):
            return True
        if re.match(r"r\d+\.htm", filename):
            return True
        return False

    def _resolve_primary_document(
        self,
        index_url: str,
        accession_number: str,
        primary_document: Optional[str] = None,
        form_type: str = "10-K",
    ) -> Optional[str]:
        """
        Fetch the filing index page and return the URL of the primary document.

        Prefers the SEC-provided primary_document filename, then other .htm/.html files.
        Returns the full URL of the document to fetch, or None on failure.
        """
        try:
            resp = self._get_raw(index_url, timeout=30)
        except requests.RequestException as exc:
            logger.warning("[EDGAR] Failed to fetch index %s: %s", index_url, exc)
            return None

        if self._is_submission_wrapper(resp.text):
            inner = self._extract_primary_filename(resp.text, form_type=form_type)
            if inner:
                return self._absolute_archive_url(index_url, inner)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        candidates: list[tuple[int, str]] = []  # (priority, href)

        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            lower = href.lower()

            if self._is_non_primary_document(href):
                continue
            if not (lower.endswith(".htm") or lower.endswith(".html")):
                continue

            if primary_document and primary_document.lower() in lower:
                candidates.append((0, href))
            else:
                acc_clean = accession_number.replace("-", "")
                if acc_clean in href:
                    candidates.append((1, href))
                else:
                    candidates.append((2, href))

        if not candidates:
            logger.warning(
                "[EDGAR] No suitable document found in index %s", index_url
            )
            return None

        candidates.sort(key=lambda x: x[0])
        return self._absolute_archive_url(index_url, candidates[0][1])

    def _absolute_archive_url(self, index_url: str, href: str) -> str:
        """Convert an EDGAR index href into a full https://www.sec.gov URL."""
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{EDGAR_WWW}{href}"
        base = index_url.rsplit("/", 1)[0]
        return f"{base}/{href.lstrip('/')}"

    def _fetch_document_text(self, url: str, timeout: int = 60) -> Optional[str]:
        try:
            resp = self._get_raw(url, timeout=timeout)
            return resp.text
        except requests.RequestException as exc:
            logger.warning("[EDGAR] Failed to fetch %s: %s", url, exc)
            return None

    def get_filing_text(self, filing: dict) -> Optional[str]:
        """
        Download the raw HTML of a filing's primary document.

        Prefers the SEC submissions API primary_document URL, validates the payload,
        and falls back to index-page resolution when needed.
        Returns the raw HTML string, or None on failure.
        """
        accession = filing.get("accession_number", "")
        form_type = filing.get("form_type", "10-K")
        primary_doc = filing.get("primary_document", "")
        document_url = filing.get("document_url", "")
        index_url = filing.get("index_url", "")

        if not index_url.endswith("-index.htm") and accession:
            cik = filing.get("cik", "").lstrip("0")
            acc_clean = accession.replace("-", "")
            index_url = f"{EDGAR_ARCHIVE}/{cik}/{acc_clean}/{accession}-index.htm"

        urls_to_try: list[str] = []
        if document_url:
            urls_to_try.append(document_url)
        if primary_doc:
            built = self._build_document_url(filing, primary_doc)
            if built not in urls_to_try:
                urls_to_try.append(built)

        resolved_url = self._resolve_primary_document(
            index_url, accession, primary_doc, form_type=form_type
        )
        if resolved_url and resolved_url not in urls_to_try:
            urls_to_try.append(resolved_url)

        for url in urls_to_try:
            text = self._fetch_document_text(url)
            if not text:
                continue

            if self._is_submission_wrapper(text):
                inner_name = self._extract_primary_filename(text, form_type)
                if not inner_name:
                    continue
                inner_url = self._build_document_url(filing, inner_name)
                logger.info(
                    "[EDGAR] Submission wrapper at %s; fetching primary doc %s",
                    url,
                    inner_name,
                )
                inner_text = self._fetch_document_text(inner_url)
                if inner_text and not self._is_submission_wrapper(inner_text):
                    return inner_text
                continue

            if len(text.strip()) > 500:
                logger.info("[EDGAR] Fetched primary document: %s", url)
                return text

        logger.error("[EDGAR] Could not fetch primary document for %s", accession)
        return None

    def get_filings_for_tickers(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        limit_per_ticker: int = 5,
    ) -> list[dict]:
        """Convenience wrapper: fetch filings for multiple tickers at once."""
        all_filings: list[dict] = []
        for ticker in tickers:
            try:
                cik = self.get_cik(ticker)
                filings = self.get_filings(cik, form_type=form_type, limit=limit_per_ticker)
                for f in filings:
                    f["ticker"] = ticker.upper()
                all_filings.extend(filings)
            except Exception as exc:
                logger.error("[EDGAR] Error processing %s: %s", ticker, exc)
        return all_filings
