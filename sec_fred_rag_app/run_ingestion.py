"""
Full Data Ingestion Pipeline 

Usage: 
- Ingest 10-k filings for a set of tickers and all FRED data
    -> python run_ingestion.py 

- Ingest specific tickers 
    -> python run_ingestion.py --tickers AAPL MSFT GOOGL --form 10-K --limit 3

- Only refresh FRED data 
    -> python run_ingestion.py --fred-only

- Force re-download even if cached 
    -> python run_ingestion.py --force 
"""

import argparse 
import sys 
from pathlib import Path

# add project root to path 
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.edgar.client import EdgarClient
