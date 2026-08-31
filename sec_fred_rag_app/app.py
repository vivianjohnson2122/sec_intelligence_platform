"""
SEC + FRED RAG Application — Streamlit frontend.

Four tabs:
    1. Company Explorer  — ingest filings and browse sections
    2. RAG Chat          — metadata-filtered Q&A over SEC filings
    3. Macro Dashboard   — FRED macro series with recession shading
    4. Sentiment Analysis — sentiment over time, by section, and correlated with macro
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path when running via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEC + FRED RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _page_guide(title: str, description: str, steps: list[str]) -> None:
    steps_html = "".join(f"<li>{step}</li>" for step in steps)
    st.markdown(
        f"""
        <div class="page-guide">
            <div class="page-guide-title">{title}</div>
            <p class="page-guide-desc">{description}</p>
            <ol class="page-guide-steps">{steps_html}</ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_styles()

# ---------------------------------------------------------------------------
# NBER recession periods (start, end) for grey shading on charts
# ---------------------------------------------------------------------------
RECESSIONS: list[tuple[str, str]] = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

PLOTLY_DARK = "plotly_dark"

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict = {
        "chat_history": [],
        "ingested_tickers": [],
        "rag_chain": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (cached so they're not re-created on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_rag_chain():
    """Load the RAG chain (initialises ChromaDB + HuggingFace LLM)."""
    try:
        from rag.chain import FilingRAGChain
        return FilingRAGChain()
    except Exception as exc:
        logger.warning("RAG chain unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Load the filing embedder (ChromaDB client)."""
    try:
        from ingestion.embedder import FilingEmbedder
        return FilingEmbedder()
    except Exception as exc:
        logger.warning("Embedder unavailable: %s", exc)
        return str(exc)  # return error string so UI can display it


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def run_fred_ingestion() -> tuple[bool, str]:
    """Fetch all 17 FRED macro series and build the macro panel parquet."""
    try:
        from ingestion.fred_client import FredClient
        client = FredClient()
        panel = client.build_macro_panel(start="2000-01-01")
        path = client.save_panel(panel)
        return True, f"FRED macro panel saved ({len(panel)} rows × {len(panel.columns)} columns)."
    except Exception as exc:
        logger.exception("FRED ingestion failed")
        return False, f"FRED ingestion error: {exc}"


def run_sec_ingestion(
    ticker: str,
    form_type: str,
    limit: int,
) -> tuple[bool, str]:
    """
    Fetch, parse, embed and score sentiment for `limit` filings of `ticker`.
    Returns (success, message).
    """
    try:
        from ingestion.edgar_client import EdgarClient
        from ingestion.filing_parser import FilingParser
        from ingestion.embedder import FilingEmbedder
        from analytics.sentiment import FilingSentimentAnalyzer

        edgar = EdgarClient()
        parser = FilingParser()
        embedder = FilingEmbedder()
        analyzer = FilingSentimentAnalyzer()

        cik = edgar.get_cik(ticker)
        company_info = edgar.get_company_info(cik)
        filings = edgar.get_filings(cik, form_type=form_type, limit=limit)

        if not filings:
            return False, f"No {form_type} filings found for {ticker}."

        total_chunks = 0
        all_sentiment_records: list[dict] = []

        for filing in filings:
            filing["ticker"] = ticker.upper()
            filing["company_name"] = company_info.get("name", ticker)

            html = edgar.get_filing_text(filing)
            if not html:
                st.warning(f"Could not fetch {filing['accession_number']} — skipping.")
                continue

            sections = parser.parse(
                html,
                form_type=form_type,
                accession_number=filing["accession_number"],
            )

            n = embedder.embed_filing(sections=sections, metadata=filing)
            total_chunks += n

            records = analyzer.score_sections(sections=sections, metadata=filing)
            all_sentiment_records.extend(records)

        if all_sentiment_records:
            df = analyzer.build_sentiment_series(all_sentiment_records)
            analyzer.save_scores(df)

        # Invalidate the RAG chain cache so it picks up new documents
        get_rag_chain.clear()

        return True, (
            f"Ingested {len(filings)} filings for {ticker.upper()}. "
            f"Added {total_chunks} chunks to ChromaDB."
        )
    except Exception as exc:
        logger.exception("Ingestion failed for %s", ticker)
        return False, f"Ingestion error: {exc}"


# ---------------------------------------------------------------------------
# Shared helper: recession shading for Plotly figures
# ---------------------------------------------------------------------------

def add_recession_shading(fig: go.Figure, x_min: Optional[str] = None, x_max: Optional[str] = None) -> go.Figure:
    """Add grey vertical bands for NBER recession periods."""
    for start, end in RECESSIONS:
        if x_max and start > x_max:
            continue
        if x_min and end < x_min:
            continue
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(200,200,200,0.12)",
            layer="below",
            line_width=0,
            annotation_text="Recession",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="gray",
        )
    return fig


# ===========================================================================
# TAB 1 — COMPANY EXPLORER
# ===========================================================================

_COMMON_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "V", "UNH", "XOM", "WMT", "JNJ", "PG",
]


def render_company_explorer() -> None:
    """Render the Company Explorer tab."""
    st.header("Company Explorer")
    _page_guide(
        "How to use this page",
        "Review what’s in your database and read individual filing sections "
        "pulled live from EDGAR.",
        [
            "Ingest tickers from the sidebar (Step 1) if you haven’t already.",
            "Scan the filings table for dates, form types, and parsed sections.",
            "Choose a section and filing, then click <strong>Load Section Text</strong>.",
            "Prefer sections like <strong>business</strong> or <strong>mda</strong> — "
            "re-ingest if you only see generic names like <strong>section_1</strong>.",
        ],
    )

    tickers_to_ingest: list[str] = st.session_state.get("_tickers_to_ingest", [])
    form_type: str = st.session_state.get("_form_type", "10-K")
    num_filings: int = st.session_state.get("_num_filings", 3)
    ingest_btn: bool = st.session_state.pop("_ingest_btn", False)
    fred_btn: bool = st.session_state.pop("_fred_btn", False)

    # --- Handle FRED ingestion ---
    if fred_btn:
        with st.spinner("Fetching 17 FRED macro series… this may take a minute."):
            success, msg = run_fred_ingestion()
        if success:
            st.success(msg)
        else:
            st.error(msg)

    # --- Handle SEC ingestion (one ticker at a time with per-ticker feedback) ---
    if ingest_btn and tickers_to_ingest:
        progress_bar = st.progress(0, text="Starting ingestion…")
        total = len(tickers_to_ingest)
        for i, ticker in enumerate(tickers_to_ingest):
            progress_bar.progress(i / total, text=f"Ingesting {ticker} ({i + 1}/{total})…")
            with st.spinner(f"Ingesting {num_filings} × {form_type} for {ticker}…"):
                success, msg = run_sec_ingestion(ticker, form_type, num_filings)
            if success:
                st.success(f"**{ticker}**: {msg}")
                if ticker not in st.session_state["ingested_tickers"]:
                    st.session_state["ingested_tickers"].append(ticker)
            else:
                st.error(f"**{ticker}**: {msg}")
        progress_bar.progress(1.0, text="Done!")
        get_rag_chain.clear()

    # --- Filing summary table ---
    embedder = get_embedder()
    if embedder is None or isinstance(embedder, str):
        error_detail = embedder if isinstance(embedder, str) else "Unknown error"
        st.warning(f"Embedder not available: {error_detail}")
        if st.button("Retry loading embedder", key="retry_embedder"):
            get_embedder.clear()
            st.rerun()
        return

    summary = embedder.get_filing_summary()

    if not summary:
        st.info("No filings ingested yet. Use the sidebar to select tickers and click Ingest.")
        return

    # Optional filter by ticker — empty means show all
    all_available = sorted({r["ticker"] for r in summary})
    filter_ticker = st.selectbox(
        "Filter by Ticker",
        options=["All"] + all_available,
        index=0,
        key="explorer_filter_ticker",
    )
    ticker_rows = summary if filter_ticker == "All" else [r for r in summary if r["ticker"] == filter_ticker]

    st.subheader(f"Ingested Filings — {filter_ticker}")
    display_df = pd.DataFrame(
        [
            {
                "Ticker": r["ticker"],
                "Company": r.get("company_name", ""),
                "Filing Date": r["filing_date"],
                "Form": r["form_type"],
                "Sections": ", ".join(sorted(r.get("sections", []))),
                "Chunks": r["chunk_count"],
                "Accession": r["accession_number"],
            }
            for r in ticker_rows
        ]
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Section text viewer ---
    st.subheader("Section Text Viewer")
    all_sections: list[str] = sorted({s for r in ticker_rows for s in r.get("sections", [])})
    if all_sections:
        chosen_section = st.selectbox("Select Section", all_sections, key="section_viewer_select")
        chosen_accession = st.selectbox(
            "Select Filing",
            [r["accession_number"] for r in ticker_rows],
            format_func=lambda a: next(
                (f"{r['ticker']} · {r['filing_date']} · {a}" for r in ticker_rows if r["accession_number"] == a),
                a,
            ),
            key="section_filing_select",
        )

        if st.button("Load Section Text", key="load_section_btn"):
            row = next(
                (r for r in ticker_rows if r["accession_number"] == chosen_accession),
                None,
            )
            section_text: Optional[str] = None
            source_label = "EDGAR"
            live_sections: Optional[dict[str, str]] = None

            with st.spinner("Fetching and parsing filing from EDGAR…"):
                try:
                    from ingestion.edgar_client import EdgarClient
                    from ingestion.filing_parser import FilingParser

                    collection = embedder.collection
                    meta_results = collection.get(
                        where={"accession_number": chosen_accession},
                        include=["metadatas"],
                        limit=1,
                    )
                    metas = meta_results.get("metadatas") or []
                    meta = metas[0] if metas else {}

                    edgar = EdgarClient()
                    parser = FilingParser()
                    form_type = row.get("form_type") if row else meta.get("form_type", "10-K")
                    ticker = row.get("ticker") if row else meta.get("ticker", "")

                    filing: Optional[dict] = None
                    if ticker:
                        cik = edgar.get_cik(ticker)
                        filings = edgar.get_filings(cik, form_type=form_type, limit=40)
                        filing = next(
                            (f for f in filings if f["accession_number"] == chosen_accession),
                            None,
                        )

                    if filing is None and meta.get("cik"):
                        acc_clean = chosen_accession.replace("-", "")
                        cik = meta["cik"]
                        filing = {
                            "cik": cik,
                            "accession_number": chosen_accession,
                            "form_type": form_type,
                            "primary_document": meta.get("primary_document", ""),
                            "document_url": "",
                            "index_url": (
                                f"https://www.sec.gov/Archives/edgar/data/"
                                f"{cik.lstrip('0')}/{acc_clean}/{chosen_accession}-index.htm"
                            ),
                        }

                    if filing:
                        html = edgar.get_filing_text(filing)
                        if html and not edgar._is_submission_wrapper(html):
                            live_sections = parser.parse(
                                html,
                                form_type=form_type,
                                accession_number=chosen_accession,
                            )
                            section_text = live_sections.get(chosen_section)
                except Exception as exc:
                    logger.warning("Live section fetch failed: %s", exc)

            # Only use ChromaDB when live EDGAR fetch did not succeed.
            if not section_text and live_sections is None:
                source_label = "ChromaDB (cached)"
                collection = embedder.collection
                results = collection.query(
                    query_texts=[""],
                    n_results=50,
                    where={
                        "$and": [
                            {"accession_number": {"$eq": chosen_accession}},
                            {"section": {"$eq": chosen_section}},
                        ]
                    },
                    include=["documents"],
                )
                docs = results.get("documents", [[]])[0]
                if docs:
                    from ingestion.edgar_client import EdgarClient

                    cached = "\n\n".join(docs)
                    if not EdgarClient()._is_submission_wrapper(cached):
                        section_text = cached

            if section_text:
                with st.expander(
                    f"Section: {chosen_section} · source: {source_label}",
                    expanded=True,
                ):
                    st.text_area("Section Text", section_text, height=400, key="section_text_area")
            elif live_sections is not None:
                available = ", ".join(sorted(live_sections.keys()))
                st.warning(
                    f"Section `{chosen_section}` is not in this filing's parsed sections. "
                    f"Available sections: {available}. "
                    "Pick one of those, or re-ingest the ticker from the sidebar to refresh "
                    "cached section names."
                )
            else:
                st.info(
                    "No text found for that section/filing combination. "
                    "If you previously ingested before a fix, re-ingest the ticker from the sidebar."
                )


# ===========================================================================
# TAB 2 — RAG CHAT
# ===========================================================================

_STARTER_QUESTIONS: list[str] = [
    "What are the primary risk factors mentioned in the most recent 10-K?",
    "How has revenue growth trended over the past two years?",
    "What does management say about macroeconomic headwinds?",
    "Describe the company's liquidity position and capital allocation strategy.",
    "What are the key drivers of operating margin expansion or compression?",
]


def render_rag_chat() -> None:
    """Render the RAG Chat tab."""
    st.header("RAG Chat — SEC Filings Q&A")
    _page_guide(
        "How to use this page",
        "Ask natural-language questions about ingested SEC filings. "
        "Answers are grounded in retrieved chunks with source citations.",
        [
            "Load at least one company’s filings from the sidebar first.",
            "Optionally narrow by ticker, section, or date range in <strong>Filters</strong>.",
            "Type a question below or tap a suggested starter prompt.",
            "Expand <strong>Sources</strong> on any answer to see the exact filing excerpts used.",
        ],
    )

    chain = get_rag_chain()
    if chain is None:
        st.warning(
            "RAG chain is not available. Ensure at least one filing has been ingested "
            "and the HuggingFace model loaded successfully."
        )
        return

    available_tickers = chain.get_available_tickers()
    available_sections = chain.get_available_sections()

    # --- Inline filters (collapsed by default to keep UI clean) ---
    with st.expander("Filters (optional — leave blank to search all filings)", expanded=False):
        col_t, col_s = st.columns(2)
        with col_t:
            selected_tickers = st.multiselect(
                "Ticker(s)", options=available_tickers, default=[], key="chat_tickers"
            )
        with col_s:
            selected_sections = st.multiselect(
                "Section(s)", options=available_sections, default=[], key="chat_sections"
            )
        col_start, col_end, col_clear = st.columns([2, 2, 1])
        with col_start:
            date_start = st.date_input("Date From", value=date(2018, 1, 1), key="chat_date_start")
        with col_end:
            date_end = st.date_input("Date To", value=date.today(), key="chat_date_end")
        with col_clear:
            st.write("")
            st.write("")
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    # --- Suggested starter questions ---
    st.subheader("Suggested Questions")
    cols = st.columns(len(_STARTER_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, _STARTER_QUESTIONS)):
        with col:
            if st.button(
                question[:55] + "…" if len(question) > 55 else question,
                key=f"starter_{i}",
                use_container_width=True,
            ):
                st.session_state["_pending_question"] = question

    st.divider()

    # Container declared first so it occupies space above the chat input
    chat_container = st.container()

    # --- Chat input renders below the container ---
    prompt = st.chat_input("Ask a question about the filings…")

    # Handle either typed input or starter button click
    pending = st.session_state.pop("_pending_question", None)
    if pending and not prompt:
        prompt = pending

    # Populate the container (renders in its original position, above the input)
    with chat_container:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                        for src in msg["sources"]:
                            st.markdown(
                                f"**{src.get('ticker')}** · {src.get('filing_date')} · "
                                f"*{src.get('section')}* · {src.get('form_type')}"
                            )
                            st.caption(src.get("text", "")[:300] + "…")
                            st.divider()

        if prompt:
            # Append and display user message
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Run RAG query and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching filings…"):
                    result = chain.query(
                        question=prompt,
                        tickers=selected_tickers or None,
                        sections=selected_sections or None,
                        date_start=str(date_start) if date_start else None,
                        date_end=str(date_end) if date_end else None,
                    )
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"Sources ({len(result['sources'])} chunks)"):
                        for src in result["sources"]:
                            st.markdown(
                                f"**{src.get('ticker')}** · {src.get('filing_date')} · "
                                f"*{src.get('section')}* · {src.get('form_type')}"
                            )
                            st.caption(src.get("text", "")[:300] + "…")
                            st.divider()

            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                }
            )


# ===========================================================================
# TAB 3 — MACRO DASHBOARD
# ===========================================================================

_CATEGORY_SERIES: dict[str, list[str]] = {
    "Interest Rates": ["DFF", "GS10", "GS2"],
    "Inflation":      ["CPIAUCSL", "PCEPI", "T10YIE"],
    "Growth":         ["GDP", "INDPRO", "RSAFS", "HOUST"],
    "Labor":          ["UNRATE", "PAYEMS", "ICSA"],
    "Credit":         ["BAMLH0A0HYM2", "BAMLC0A0CM"],
    "Sentiment":      ["UMCSENT", "VIXCLS"],
}

try:
    from ingestion.fred_client import SERIES_NAME_MAP
except Exception:
    SERIES_NAME_MAP = {}


def render_macro_dashboard() -> None:
    """Render the Macro Dashboard tab."""
    st.header("Macro Dashboard — FRED Economic Data")
    _page_guide(
        "How to use this page",
        "Explore 17 FRED macro indicators grouped by theme, with NBER recession "
        "periods shaded on every chart.",
        [
            "Fetch or refresh macro data from the sidebar (Step 2) if the page is empty.",
            "Pick a <strong>Category</strong> (rates, inflation, labor, etc.) and adjust the date range.",
            "Use the chart range slider to zoom into a specific period.",
            "Check <strong>Summary Statistics</strong> below for current levels and recent changes.",
        ],
    )

    macro_df = None
    macro_path = Path("./data/fred/macro_panel.parquet")
    if macro_path.exists():
        try:
            macro_df = pd.read_parquet(macro_path)
            macro_df.index = pd.to_datetime(macro_df.index)
        except Exception as exc:
            st.warning(f"Could not load macro panel: {exc}")

    if macro_df is None:
        st.info("No FRED macro data found yet.")
        if st.button("Fetch FRED Macro Data Now", type="primary", key="macro_fetch_btn"):
            with st.spinner("Fetching 17 FRED macro series… this may take a minute."):
                success, msg = run_fred_ingestion()
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        return

    # --- Inline controls ---
    date_min = macro_df.index.min().date()
    date_max = macro_df.index.max().date()
    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        category = st.selectbox(
            "Category",
            list(_CATEGORY_SERIES.keys()),
            key="macro_category",
        )
    with ctrl_col2:
        macro_date_range = st.slider(
            "Date Range",
            min_value=date_min,
            max_value=date_max,
            value=(date(2010, 1, 1), date_max),
            key="macro_date_range",
        )

    series_ids = [s for s in _CATEGORY_SERIES[category] if s in macro_df.columns]
    if not series_ids:
        st.warning(f"No data found for category '{category}' in the macro panel.")
        return

    start_str = str(macro_date_range[0])
    end_str = str(macro_date_range[1])
    filtered = macro_df.loc[start_str:end_str, series_ids].dropna(how="all")

    # --- Line chart ---
    st.subheader(f"{category} Indicators")
    fig = go.Figure()
    for sid in series_ids:
        if sid not in filtered.columns:
            continue
        series_data = filtered[sid].dropna()
        fig.add_trace(
            go.Scatter(
                x=series_data.index,
                y=series_data.values,
                name=SERIES_NAME_MAP.get(sid, sid),
                mode="lines",
                line=dict(width=2),
            )
        )
    add_recession_shading(fig, x_min=start_str, x_max=end_str)
    fig.update_layout(
        template=PLOTLY_DARK,
        height=420,
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary stats table ---
    st.subheader("Summary Statistics")
    stats_rows = []
    for sid in series_ids:
        if sid not in macro_df.columns:
            continue
        s = macro_df[sid].dropna()
        if s.empty:
            continue
        current = s.iloc[-1]
        one_month_ago = s.iloc[-2] if len(s) >= 2 else None
        one_year_ago = s.shift(12).iloc[-1] if len(s) >= 13 else None
        last_52w = s.iloc[-52:] if len(s) >= 52 else s

        stats_rows.append(
            {
                "Series": SERIES_NAME_MAP.get(sid, sid),
                "Current": f"{current:.3f}",
                "1M Change": (
                    f"{current - one_month_ago:+.3f}" if one_month_ago is not None else "—"
                ),
                "1Y Change": (
                    f"{current - one_year_ago:+.3f}" if one_year_ago is not None else "—"
                ),
                "52W High": f"{last_52w.max():.3f}",
                "52W Low": f"{last_52w.min():.3f}",
            }
        )
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)



# ===========================================================================
# MAIN — Tab layout
# ===========================================================================

def render_sidebar() -> None:
    """Render a single, step-by-step sidebar shared across all tabs."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">Setup Guide</p>
                <p class="sidebar-brand-desc">
                    Complete these steps once, then explore any tab above.
                    Ingestion runs in the background and may take a few minutes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-step">
                <div class="sidebar-step-num">1</div>
                <p class="sidebar-step-title">Load company filings</p>
                <p class="sidebar-step-desc">
                    Select tickers and choose 10-K or 10-Q. Each filing is parsed
                    into sections, embedded, and stored for search.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cb_cols = st.columns(2)
        checked_tickers: list[str] = []
        for i, ticker in enumerate(_COMMON_TICKERS):
            with cb_cols[i % 2]:
                if st.checkbox(ticker, key=f"cb_{ticker}"):
                    checked_tickers.append(ticker)

        custom_ticker = st.text_input(
            "Or enter any ticker",
            value="",
            placeholder="e.g. BABA, COIN, PLTR…",
            key="ticker_input",
        ).strip().upper()

        tickers_to_ingest: list[str] = list(
            dict.fromkeys(checked_tickers + ([custom_ticker] if custom_ticker else []))
        )
        st.session_state["_tickers_to_ingest"] = tickers_to_ingest

        col_form, col_n = st.columns(2)
        with col_form:
            form_type = st.selectbox("Form type", ["10-K", "10-Q"], key="form_type_select")
        with col_n:
            num_filings = st.number_input("# filings", min_value=1, max_value=10, value=3, key="num_filings")

        st.session_state["_form_type"] = form_type
        st.session_state["_num_filings"] = num_filings

        if st.button("Ingest Selected Filings", type="primary", key="ingest_btn", disabled=not tickers_to_ingest):
            st.session_state["_ingest_btn"] = True

        # Show which tickers are already in the DB
        try:
            from ingestion.embedder import FilingEmbedder
            _emb = FilingEmbedder()
            existing = _emb.list_tickers()
            if existing:
                st.caption(f"Already loaded: {', '.join(existing)}")
        except Exception:
            pass

        st.divider()

        st.markdown(
            """
            <div class="sidebar-step">
                <div class="sidebar-step-num">2</div>
                <p class="sidebar-step-title">Load macro data</p>
                <p class="sidebar-step-desc">
                    Pulls 17 FRED series (rates, inflation, labor, credit, sentiment)
                    into the Macro Dashboard. Safe to refresh anytime.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        macro_path = Path("./data/fred/macro_panel.parquet")
        if macro_path.exists():
            st.success("FRED data loaded")
            fred_label = "Refresh FRED Data"
        else:
            st.warning("Not yet fetched")
            fred_label = "Fetch FRED Data"

        if st.button(fred_label, key="fred_btn"):
            st.session_state["_fred_btn"] = True

        st.divider()

        st.markdown(
            """
            <div class="sidebar-step">
                <div class="sidebar-step-num">3</div>
                <p class="sidebar-step-title">Explore &amp; ask questions</p>
                <p class="sidebar-step-desc">
                    Use <strong>RAG Chat</strong> for Q&amp;A, <strong>Company Explorer</strong>
                    to browse sections, and <strong>Macro Dashboard</strong> for economic context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Entry point: renders the three-tab layout."""
    st.markdown(
        """
        <div class="app-hero">
            <p class="app-hero-title">SEC + FRED <span>RAG Platform</span></p>
            <p class="app-hero-tagline">
                Ingest SEC filings, query them with retrieval-augmented generation,
                and overlay macroeconomic context from FRED — all in one workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["RAG Chat", "Company Explorer", "Macro Dashboard"])

    with tab1:
        render_rag_chat()

    with tab2:
        render_company_explorer()

    with tab3:
        render_macro_dashboard()


if __name__ == "__main__":
    main()
