"""
Full data ingestion pipeline CLI.

Usage examples:
    # Ingest 10-K filings for a set of tickers + all FRED data
    python run_ingestion.py

    # Ingest specific tickers
    python run_ingestion.py --tickers AAPL MSFT GOOGL --form 10-K --limit 3

    # Only refresh FRED data
    python run_ingestion.py --fred-only

    # Only ingest SEC filings (skip FRED)
    python run_ingestion.py --sec-only --tickers NVDA --limit 5

    # Force re-download even if documents exist in ChromaDB
    python run_ingestion.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "NVDA"]


def ingest_fred(start: str = "2000-01-01") -> None:
    """Fetch all 17 FRED series and save the macro panel as Parquet."""
    from ingestion.fred_client import FredClient

    logger.info("=== FRED Ingestion ===")
    client = FredClient()
    panel = client.build_macro_panel(start=start)
    path = client.save_panel(panel)
    logger.info("FRED panel saved to %s  (%d rows × %d cols)", path, len(panel), len(panel.columns))


def ingest_sec(
    tickers: list[str],
    form_type: str,
    limit: int,
    force: bool = False,
) -> None:
    """Fetch, parse, embed, and score sentiment for SEC filings."""
    from ingestion.edgar_client import EdgarClient
    from ingestion.filing_parser import FilingParser
    from ingestion.embedder import FilingEmbedder
    from analytics.sentiment import FilingSentimentAnalyzer

    edgar = EdgarClient()
    parser = FilingParser()
    embedder = FilingEmbedder()
    analyzer = FilingSentimentAnalyzer()

    if not force:
        existing_tickers = set(embedder.list_tickers())
    else:
        existing_tickers = set()

    logger.info("=== SEC Ingestion  |  tickers=%s  form=%s  limit=%d ===",
                tickers, form_type, limit)

    all_sentiment: list[dict] = []

    for ticker in tickers:
        ticker_upper = ticker.upper()
        if ticker_upper in existing_tickers and not force:
            logger.info("[%s] Already in ChromaDB — skipping (use --force to re-ingest)", ticker_upper)
            continue

        try:
            cik = edgar.get_cik(ticker_upper)
            company_info = edgar.get_company_info(cik)
            filings = edgar.get_filings(cik, form_type=form_type, limit=limit)
        except Exception as exc:
            logger.error("[%s] Failed to fetch filings: %s", ticker_upper, exc)
            continue

        if not filings:
            logger.warning("[%s] No %s filings found", ticker_upper, form_type)
            continue

        for filing in filings:
            filing["ticker"] = ticker_upper
            filing["company_name"] = company_info.get("name", ticker_upper)

            logger.info("[%s] Processing %s (%s)", ticker_upper, filing["accession_number"],
                        filing["filing_date"])

            html = edgar.get_filing_text(filing)
            if not html:
                logger.warning("[%s] Could not fetch text for %s", ticker_upper,
                               filing["accession_number"])
                continue

            sections = parser.parse(
                html,
                form_type=form_type,
                accession_number=filing["accession_number"],
            )
            logger.info("[%s] Parsed %d sections", ticker_upper, len(sections))

            n_chunks = embedder.embed_filing(sections=sections, metadata=filing)
            logger.info("[%s] Embedded %d chunks", ticker_upper, n_chunks)

            records = analyzer.score_sections(sections=sections, metadata=filing)
            all_sentiment.extend(records)

    if all_sentiment:
        df = analyzer.build_sentiment_series(all_sentiment)
        out = analyzer.save_scores(df)
        logger.info("Sentiment scores saved to %s  (%d rows)", out, len(df))

    stats = embedder.get_collection_stats()
    logger.info("ChromaDB total chunks: %d", stats["total_chunks"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC + FRED data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        metavar="TICKER",
        help=f"Ticker symbols to ingest (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        choices=["10-K", "10-Q"],
        help="SEC form type to fetch (default: 10-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        metavar="N",
        help="Max filings per ticker (default: 3)",
    )
    parser.add_argument(
        "--fred-only",
        action="store_true",
        help="Only refresh FRED data, skip SEC ingestion",
    )
    parser.add_argument(
        "--sec-only",
        action="store_true",
        help="Only ingest SEC filings, skip FRED",
    )
    parser.add_argument(
        "--fred-start",
        default="2000-01-01",
        metavar="YYYY-MM-DD",
        help="Start date for FRED series (default: 2000-01-01)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest tickers already present in ChromaDB",
    )

    args = parser.parse_args()

    try:
        if not args.sec_only:
            ingest_fred(start=args.fred_start)

        if not args.fred_only:
            ingest_sec(
                tickers=args.tickers,
                form_type=args.form,
                limit=args.limit,
                force=args.force,
            )

        logger.info("=== Ingestion complete ===")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
