"""
Unit tests for ingestion/fred_client.py.

All tests use synthetic DataFrames and run entirely offline — no FRED API calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.fred_client import FredClient, FRED_SERIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_monthly_panel(start: str = "2015-01", periods: int = 60) -> pd.DataFrame:
    """Create a synthetic monthly macro panel with common FRED series columns."""
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    rng = np.random.default_rng(seed=42)

    data: dict[str, np.ndarray] = {
        "DFF":      rng.uniform(0.0, 5.5, periods),
        "GS10":     rng.uniform(1.0, 5.0, periods),
        "GS2":      rng.uniform(0.5, 4.5, periods),
        "CPIAUCSL": np.linspace(240, 310, periods) + rng.normal(0, 1, periods),
        "UNRATE":   rng.uniform(3.5, 10.0, periods),
        "VIXCLS":   rng.uniform(12.0, 80.0, periods),
        "GDP":      np.linspace(18000, 25000, periods) + rng.normal(0, 200, periods),
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# compute_features() tests
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    """Tests for FredClient.compute_features() static method."""

    def test_yoy_columns_created(self) -> None:
        """YoY columns should be created for every raw series column."""
        panel = _make_monthly_panel()
        result = FredClient.compute_features(panel)
        for col in panel.columns:
            assert f"{col}_yoy" in result.columns, f"Missing {col}_yoy"

    def test_mom_columns_created(self) -> None:
        """MoM columns should be created for every raw series column."""
        panel = _make_monthly_panel()
        result = FredClient.compute_features(panel)
        for col in panel.columns:
            assert f"{col}_mom" in result.columns, f"Missing {col}_mom"

    def test_yield_curve_column_when_gs10_gs2_present(self) -> None:
        """yield_curve = GS10 - GS2 should be added when both series are present."""
        panel = _make_monthly_panel()
        assert "GS10" in panel.columns and "GS2" in panel.columns
        result = FredClient.compute_features(panel)
        assert "yield_curve" in result.columns
        # Spot-check: yield_curve at each row should equal GS10 - GS2
        diff = (result["GS10"] - result["GS2"]).round(8)
        yc = result["yield_curve"].round(8)
        pd.testing.assert_series_equal(yc, diff, check_names=False)

    def test_no_yield_curve_without_gs10(self) -> None:
        """yield_curve should NOT be added if GS10 is missing."""
        panel = _make_monthly_panel().drop(columns=["GS10"])
        result = FredClient.compute_features(panel)
        assert "yield_curve" not in result.columns

    def test_yoy_values_are_percentage(self) -> None:
        """YoY values should be in percentage points, not fractional."""
        # CPI goes from ~240 to ~310 over 5 years, so YoY should be ~single digits
        panel = _make_monthly_panel(periods=36)
        result = FredClient.compute_features(panel)
        yoy_vals = result["CPIAUCSL_yoy"].dropna()
        # Should be expressed as percent (~2-5% for gentle trend), not 0.02-0.05
        assert yoy_vals.abs().max() > 0.5, "YoY values look fractional, expected %"

    def test_first_12_rows_yoy_are_nan(self) -> None:
        """First 12 rows of _yoy columns must be NaN (no prior year to compare)."""
        panel = _make_monthly_panel(periods=36)
        result = FredClient.compute_features(panel)
        assert result["DFF_yoy"].iloc[:12].isna().all(), (
            "Expected first 12 rows of DFF_yoy to be NaN"
        )

    def test_first_row_mom_is_nan(self) -> None:
        """First row of _mom columns must be NaN."""
        panel = _make_monthly_panel()
        result = FredClient.compute_features(panel)
        assert pd.isna(result["UNRATE_mom"].iloc[0])

    def test_original_columns_preserved(self) -> None:
        """Original column values must be unchanged after compute_features()."""
        panel = _make_monthly_panel()
        result = FredClient.compute_features(panel)
        for col in panel.columns:
            pd.testing.assert_series_equal(panel[col], result[col])

    def test_empty_dataframe_returns_empty(self) -> None:
        """compute_features on an empty DataFrame should not crash."""
        empty = pd.DataFrame({"GS10": pd.Series(dtype=float), "GS2": pd.Series(dtype=float)})
        result = FredClient.compute_features(empty)
        assert "yield_curve" in result.columns or result.empty

    def test_feature_count(self) -> None:
        """Result should have at least 3× the columns of the input (raw + yoy + mom)."""
        panel = _make_monthly_panel()
        result = FredClient.compute_features(panel)
        # Each raw col gets 2 derived cols, plus yield_curve
        min_expected = len(panel.columns) * 3
        assert len(result.columns) >= min_expected - 1  # allow for yield_curve


# ---------------------------------------------------------------------------
# FRED_SERIES catalogue tests
# ---------------------------------------------------------------------------

class TestFredSeriesCatalogue:
    """Sanity checks on the FRED_SERIES metadata dictionary."""

    def test_series_count(self) -> None:
        """Exactly 17 series should be defined."""
        assert len(FRED_SERIES) == 17, f"Expected 17 series, got {len(FRED_SERIES)}"

    def test_all_series_have_name_and_category(self) -> None:
        """Every series entry must have 'name' and 'category' keys."""
        for sid, meta in FRED_SERIES.items():
            assert "name" in meta, f"Series {sid} missing 'name'"
            assert "category" in meta, f"Series {sid} missing 'category'"

    def test_expected_series_present(self) -> None:
        """Key series should be in the catalogue."""
        required = {"DFF", "GS10", "GS2", "CPIAUCSL", "UNRATE", "VIXCLS", "GDP"}
        missing = required - set(FRED_SERIES.keys())
        assert not missing, f"Missing required series: {missing}"

    def test_categories_are_valid(self) -> None:
        """All category values should be from the expected set."""
        valid = {"Interest Rates", "Inflation", "Growth", "Labor", "Credit", "Sentiment"}
        for sid, meta in FRED_SERIES.items():
            assert meta["category"] in valid, (
                f"Series {sid} has unknown category '{meta['category']}'"
            )
