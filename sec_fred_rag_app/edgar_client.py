"""
Wrapper around the SEC EDGAR REST API
Docs: https://efts.sec.gov/LATEST/search-index?q=%22edgar%22&dateRange=custom

Key Endpoints Used: 
    - /cgi-bin/browse-edgar     -> search filings by ticker
    - /submissions              -> company metadata + filing history
    - /Archives/edgar/data      -> actual filing documents
"""

import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv


load_dotenv()

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}
RATE_LIMIT_DELAY = 0.11 # SEC asks for max 10 req/sec

class EdgarClient:
    """
    Fetch company filings from SEC EDGAR

    Usage: 
        client = EdgarClient()
        cik = client.get_cik("APPL")
        filings = client.get_filings(cik, form_type="10-K", limit=5)
        text = client.get_filing_text(filings[0])
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    

    def _get(self,
             url: str) -> dict:
        """
        Rate-limited GET with basic error handling 
        """
        time.sleep(RATE_LIMIT_DELAY)
        resp = self.session.get(url,
                                timeout=30)
        resp.raise_for_status()
        return resp.json()


    def get_cik(self,
                ticker: str) -> str:
        """
        Resolve a ticker symbol to a zero-padded 10 digit CIK
        EX/ "APPL" -> 0000320193
        """
        url = f"{EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
        # EDGAR lookup via ticker map (ticker -> number)
        ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(ticker_map_url)

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry['ticker'].upper() == ticker_upper:
                # matches 
                cik = str(entry['cik_str']).zfill(10)
                print(f"[EDGAR] {ticker_upper} -> CIK {cik} ({entry['title']})")
                return cik 
        # error - not found   
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list")
    
    
    def get_company_info(self,
                         cik: str) -> dict:
        """
        Return a dictionary of company metadata including name, SIC code, state
        """
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self._get(url)
        company_info = {
            "cik": cik,
            "name": data.get("name"),
            "ticker": data.get("tickers", [None])[0],
            "sic": data.get("sic"),
            "sic_description": data.get('sicDescription'),
            "state": data.get("stateOfIncorporation")
        }
        return company_info
    

    def get_filings(self,
                    cik: str,
                    form_type: str="10-K",
                    limit: int=5) -> list[dict]:
        """
        Return a list of filing metadata dictionaries for a given form type

        Each dict contains: 
            accession_number
            filing_date
            form_type
            primary_document
            document_url
            index_url
        """

        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self._get(url)

        # returns in dict with following keys: 
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type:
                accession = accessions[i].replace("-", "")
                doc = primary_docs[i]
                results.append({
                    "cik": cik,
                    "accession_number": accessions[i],
                    "filing_date": dates[i],
                    "form_type": form,
                    "primary_document": doc,
                    "document_url": f"{EDGAR_ARCHIVE}/{cik.lstrip('0')}/{accession}/{doc}",
                    "index_url": f"{EDGAR_ARCHIVE}/{cik.lstrip('0')}/{accession}/",
                })
                if len(results) >= limit:
                    break
        print(f"[EDGAR] Found {len(results)} {form_type} filings for CIK {cik}")
        return results


    def get_filing_text(self,
                        filing: dict) -> Optional[str]:
        """
        Download the raw HTML / text of a filing's primary document
        Retruns raw text string
        """

        url = filing['document_url']
        print(f"[EDGAR] Fetching: {url}")
        time.sleep(RATE_LIMIT_DELAY)

        try:
            resp = self.session.get(url,
                                    timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"[EDGAR] Failed to fetch {url}: {e}")
            return None
        
    
    def get_filings_for_tickers(self,
                                tickers: list[str],
                                form_type: str="10-K",
                                limit_per_ticker: int=5) -> list[dict]:
        """
        Convenience: fetch filings for multiple tickers at once
        """

        all_filings = []
        for ticker in tickers:
            try:
                cik = self.get_cik(ticker)
                filings = self.get_filings(cik,
                                           form_type=form_type,
                                           limit=limit_per_ticker)
                for f in filings:
                    f['ticker'] = ticker.upper()
                all_filings.extend(filings)
            except Exception as e:
                print(f"[EDGAR] Error processing {ticker}: {e}")
        
        return all_filings