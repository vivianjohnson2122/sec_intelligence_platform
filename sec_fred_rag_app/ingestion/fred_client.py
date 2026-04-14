"""
FRED (Federal Reserve Economic Data) API wrapper.

Fetches 17 macro series, engineers features (YoY, MoM, yield curve),
and stores the result as Parquet files in ./data/fred/.

Series categories:
    Interest Rates : DFF, GS10, GS2
    Inflation      : CPIAUCSL, PCEPI, T10YIE
    Growth         : GDP, INDPRO, RSAFS, HOUST
    Labor          : UNRATE, PAYEMS, ICSA
    Credit         : BAMLH0A0HYM2, BAMLC0A0CM
    Sentiment      : UMCSENT, VIXCLS
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FRED_DATA_DIR = Path(os.getenv("FRED_DATA_DIR", "./data/fred"))

FRED_SERIES: dict[str, dict] = {
    "DFF":          {"name": "Fed Funds Rate",              "category": "Interest Rates"},
    "GS10":         {"name": "10Y Treasury Yield",          "category": "Interest Rates"},
    "GS2":          {"name": "2Y Treasury Yield",           "category": "Interest Rates"},
    "CPIAUCSL":     {"name": "CPI (All Urban)",             "category": "Inflation"},
    "PCEPI":        {"name": "PCE Price Index",             "category": "Inflation"},
    "T10YIE":       {"name": "10Y Breakeven Inflation",     "category": "Inflation"},
    "GDP":          {"name": "GDP",                         "category": "Growth"},
    "INDPRO":       {"name": "Industrial Production Index", "category": "Growth"},
    "RSAFS":        {"name": "Retail Sales",                "category": "Growth"},
    "HOUST":        {"name": "Housing Starts",              "category": "Growth"},
    "UNRATE":       {"name": "Unemployment Rate",           "category": "Labor"},
    "PAYEMS":       {"name": "Nonfarm Payrolls",            "category": "Labor"},
    "ICSA":         {"name": "Initial Jobless Claims",      "category": "Labor"},
    "BAMLH0A0HYM2": {"name": "HY Credit Spread",           "category": "Credit"},
    "BAMLC0A0CM":   {"name": "IG Credit Spread",           "category": "Credit"},
    "UMCSENT":      {"name": "Consumer Sentiment (UMich)",  "category": "Sentiment"},
    "VIXCLS":       {"name": "VIX (Market Volatility)",    "category": "Sentiment"},
}

SERIES_NAME_MAP: dict[str, str] = {k: v["name"] for k, v in FRED_SERIES.items()}
SERIES_CATEGORY_MAP: dict[str, str] = {k: v["category"] for k, v in FRED_SERIES.items()}


class FredClient:
    """
    Fetch and engineer FRED macro series.

    Usage:
        client = FredClient()
        panel = client.build_macro_panel(start="2000-01-01")
        client.save_panel(panel)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise ValueError("FRED_API_KEY not set. Add it to your .env file.")
        self.fred = Fred(api_key=key)
        FRED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_series(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetch a single FRED series, resample to monthly end-of-period,
        forward-fill short gaps (up to 3 periods), and return a named Series.
        """
        logger.info("[FRED] Fetching %s (%s)…", series_id, FRED_SERIES.get(series_id, {}).get("name", ""))
        raw: pd.Series = self.fred.get_series(series_id, observation_start=start,
                                               observation_end=end)
        raw.index = pd.to_datetime(raw.index)
        # Resample to monthly; use last observation in the month
        monthly = raw.resample("ME").last().ffill(limit=3)
        monthly.name = series_id
        return monthly

    def fetch_all_series(
        self,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        series_ids: Optional[list[str]] = None,
    ) -> dict[str, pd.Series]:
        """Fetch every series in FRED_SERIES (or a subset) and return as a dict."""
        ids = series_ids or list(FRED_SERIES.keys())
        result: dict[str, pd.Series] = {}
        for sid in ids:
            try:
                result[sid] = self.fetch_series(sid, start=start, end=end)
                # Save individual series as Parquet
                out = FRED_DATA_DIR / f"{sid}.parquet"
                result[sid].to_frame().to_parquet(out)
            except Exception as exc:
                logger.error("[FRED] Failed to fetch %s: %s", sid, exc)
        return result

    def build_macro_panel(
        self,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        series_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a wide-format macro panel DataFrame with engineered features.

        Engineered columns added per series:
            {sid}_yoy   : year-over-year % change
            {sid}_mom   : month-over-month % change

        Additional derived columns:
            yield_curve : GS10 - GS2 (if both are present)

        Returns:
            DataFrame indexed by monthly date with one column per series/feature.
        """
        series_dict = self.fetch_all_series(start=start, end=end, series_ids=series_ids)
        if not series_dict:
            raise RuntimeError("No FRED series fetched successfully.")

        panel = pd.DataFrame(series_dict)
        panel.index.name = "date"

        return self.compute_features(panel)

    @staticmethod
    def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer YoY, MoM, and yield-curve features from a raw panel DataFrame.

        Args:
            panel: Wide DataFrame with FRED series IDs as columns.

        Returns:
            Same DataFrame with additional _yoy, _mom columns and yield_curve.
        """
        result = panel.copy()
        for col in panel.columns:
            if col.endswith(("_yoy", "_mom", "_yield_curve")):
                continue
            result[f"{col}_yoy"] = panel[col].pct_change(12) * 100
            result[f"{col}_mom"] = panel[col].pct_change(1) * 100

        # Yield curve: 10Y minus 2Y spread
        if "GS10" in panel.columns and "GS2" in panel.columns:
            result["yield_curve"] = panel["GS10"] - panel["GS2"]

        return result

    def save_panel(self, panel: pd.DataFrame) -> Path:
        """Save the macro panel to Parquet and return the path."""
        out = FRED_DATA_DIR / "macro_panel.parquet"
        panel.to_parquet(out)
        logger.info("[FRED] Macro panel saved to %s (%d rows, %d cols)",
                    out, len(panel), len(panel.columns))
        return out

    @staticmethod
    def load_panel() -> Optional[pd.DataFrame]:
        """Load the macro panel from Parquet if it exists, else return None."""
        path = FRED_DATA_DIR / "macro_panel.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
