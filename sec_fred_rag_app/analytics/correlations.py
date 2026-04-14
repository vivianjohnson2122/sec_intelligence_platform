"""
Correlation analysis between SEC filing sentiment and FRED macro indicators.

Methodology:
    1. Merge sentiment scores with the macro panel using pd.merge_asof
       (nearest-date join) so each filing date is matched to the nearest
       month-end macro observation.
    2. Compute Pearson r between the sentiment column and every FRED series,
       reporting the correlation coefficient and p-value.
    3. Visualise as a horizontal bar chart coloured green/red by sign,
       sorted by absolute correlation magnitude.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def merge_sentiment_macro(
    sentiment_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_col_sentiment: str = "filing_date",
) -> pd.DataFrame:
    """
    Merge sentiment scores with the macro panel on nearest date.

    Uses pd.merge_asof (sorted left-key backward merge) so each filing
    date is matched to the most recent macro observation on or before it.

    Args:
        sentiment_df: DataFrame with at least a filing_date column and
                      compound (or other sentiment) column.
        macro_df: Wide DataFrame indexed by date with FRED series columns.
        date_col_sentiment: Name of the date column in sentiment_df.

    Returns:
        Merged DataFrame with both sentiment columns and macro columns.
    """
    sent = sentiment_df.copy()
    sent[date_col_sentiment] = pd.to_datetime(sent[date_col_sentiment], errors="coerce")
    sent = sent.dropna(subset=[date_col_sentiment]).sort_values(date_col_sentiment)

    macro = macro_df.copy()
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    macro_reset = macro.reset_index()
    macro_reset.rename(columns={"date": "macro_date"}, inplace=True)

    merged = pd.merge_asof(
        sent,
        macro_reset,
        left_on=date_col_sentiment,
        right_on="macro_date",
        direction="backward",
    )
    logger.info(
        "[Correlations] Merged %d sentiment rows with macro panel → %d rows",
        len(sent),
        len(merged),
    )
    return merged


def compute_correlations(
    merged_df: pd.DataFrame,
    sentiment_col: str = "compound",
    min_observations: int = 10,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between the sentiment column and every
    numeric FRED column in the merged DataFrame.

    Args:
        merged_df: Output of merge_sentiment_macro().
        sentiment_col: Name of the sentiment column (default: "compound").
        min_observations: Minimum non-NaN paired observations to include
                          a series in the results.

    Returns:
        DataFrame sorted by absolute correlation with columns:
            series_id, series_name, correlation, p_value, n_observations
    """
    from ingestion.fred_client import SERIES_NAME_MAP

    if sentiment_col not in merged_df.columns:
        raise ValueError(f"Column '{sentiment_col}' not found in DataFrame.")

    sentiment_vals = merged_df[sentiment_col].values

    # Only consider raw FRED series columns (not _yoy/_mom/_yield_curve)
    exclude_suffixes = ("_yoy", "_mom")
    numeric_cols = [
        c for c in merged_df.select_dtypes(include=[np.number]).columns
        if c != sentiment_col
        and not any(c.endswith(s) for s in exclude_suffixes)
        and c != "chunk_index"
    ]

    records = []
    for col in numeric_cols:
        mask = ~(np.isnan(sentiment_vals) | merged_df[col].isna().values)
        n = mask.sum()
        if n < min_observations:
            continue
        r, p = stats.pearsonr(sentiment_vals[mask], merged_df[col].values[mask])
        records.append(
            {
                "series_id": col,
                "series_name": SERIES_NAME_MAP.get(col, col),
                "correlation": round(float(r), 4),
                "p_value": round(float(p), 6),
                "n_observations": int(n),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["series_id", "series_name", "correlation", "p_value", "n_observations"]
        )

    corr_df = pd.DataFrame(records)
    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    corr_df = corr_df.sort_values("abs_correlation", ascending=False).drop(
        columns=["abs_correlation"]
    )
    return corr_df.reset_index(drop=True)


def plot_correlation_bar(
    corr_df: pd.DataFrame,
    title: str = "Sentiment–Macro Correlation (Pearson r)",
    max_series: int = 20,
) -> go.Figure:
    """
    Return a horizontal bar chart of sentiment correlations with FRED series.

    Bars are green for positive and red for negative correlations.
    Sorted by absolute value (largest effect at top).

    Args:
        corr_df: Output of compute_correlations().
        title: Chart title.
        max_series: Maximum number of series to display.

    Returns:
        Plotly Figure object.
    """
    if corr_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No correlation data available")
        return fig

    plot_df = corr_df.head(max_series).copy()
    plot_df = plot_df.sort_values("correlation", ascending=True)

    colors = ["#ef4444" if r < 0 else "#22c55e" for r in plot_df["correlation"]]

    # Label: series name + significance stars
    def significance_star(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    labels = [
        f"{name} {significance_star(p)}"
        for name, p in zip(plot_df["series_name"], plot_df["p_value"])
    ]
    hover = [
        f"r = {r:.3f}<br>p = {p:.4f}<br>n = {n}"
        for r, p, n in zip(
            plot_df["correlation"], plot_df["p_value"], plot_df["n_observations"]
        )
    ]

    fig = go.Figure(
        go.Bar(
            x=plot_df["correlation"],
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.add_vline(x=0, line_color="white", line_width=1, opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Pearson r",
        yaxis_title="",
        height=max(400, len(plot_df) * 28),
        margin={"l": 200, "r": 40, "t": 50, "b": 50},
        annotations=[
            dict(
                x=0.01,
                y=-0.12,
                xref="paper",
                yref="paper",
                text="* p<0.05  ** p<0.01  *** p<0.001",
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ],
    )
    return fig
