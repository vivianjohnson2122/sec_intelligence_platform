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
                        ),
                    }
                )
                if len(results) >= limit:
                    break

        logger.info("[EDGAR] Found %d %s filings for CIK %s", len(results), form_type, cik)
        return results

    def _resolve_primary_document(self, index_url: str, accession_number: str) -> Optional[str]:
        """
        Fetch the filing index page and return the URL of the primary document.

        Prefers .htm/.html files, avoids -index.htm files and .txt submission files.
        Returns the full URL of the document to fetch, or None on failure.
        """
        try:
            resp = self._get_raw(index_url, timeout=30)
        except requests.RequestException as exc:
            logger.warning("[EDGAR] Failed to fetch index %s: %s", index_url, exc)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # EDGAR index pages have a table with columns: Seq, Description, Document, Type, Size
        candidates: list[tuple[int, str]] = []  # (priority, href)

        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            lower = href.lower()

            # Skip index pages and submission text files
            if "-index.htm" in lower or lower.endswith(".txt"):
                continue

            # Only consider htm/html documents
            if not (lower.endswith(".htm") or lower.endswith(".html")):
                continue

            # Prefer files that look like the primary 10-K/10-Q document
            # (contain the accession number without dashes, or named like r*.htm)
            acc_clean = accession_number.replace("-", "")
            if acc_clean in href and not lower.endswith("-index.htm"):
                # Highest priority: direct accession-number document
                candidates.append((0, href))
            elif re.search(r"\d{18}", href):
                candidates.append((1, href))
            else:
                candidates.append((2, href))

        if not candidates:
            logger.warning(
                "[EDGAR] No suitable document found in index %s; falling back to direct URL",
                index_url,
            )
            return None

        # Sort by priority (lowest = best), take first
        candidates.sort(key=lambda x: x[0])
        best_href = candidates[0][1]

        # href may be relative (e.g. /Archives/edgar/data/...)
        if best_href.startswith("http"):
            return best_href
        return f"{EDGAR_WWW}{best_href}"

    def get_filing_text(self, filing: dict) -> Optional[str]:
        """
        Download the raw HTML of a filing's primary document.

        First fetches the index page to resolve the correct document URL,
        avoiding index pages and submission wrapper files.
        Returns the raw HTML string, or None on failure.
        """
        accession = filing.get("accession_number", "")
        index_url = filing.get("index_url", "")

        # Attempt to resolve the real document from the index
        resolved_url = self._resolve_primary_document(index_url, accession)

        # Fall back to the pre-computed document_url if resolution fails
        url = resolved_url or filing["document_url"]
        logger.info("[EDGAR] Fetching document: %s", url)

        try:
            resp = self._get_raw(url, timeout=60)
            return resp.text
        except requests.RequestException as exc:
            logger.error("[EDGAR] Failed to fetch %s: %s", url, exc)
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
