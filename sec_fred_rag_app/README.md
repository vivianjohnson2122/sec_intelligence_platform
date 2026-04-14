# SEC + FRED RAG Platform

An end-to-end data science application that ingests SEC filings and FRED macroeconomic data, provides a retrieval-augmented generation (RAG) chat interface, and surfaces sentiment and correlation analytics through a Streamlit dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit App                        │
│  ┌──────────────┐ ┌──────────────┐ ┌────────┐ ┌─────────┐  │
│  │   Explorer   │ │   RAG Chat   │ │  Macro │ │Sentiment│  │
│  └──────┬───────┘ └──────┬───────┘ └───┬────┘ └────┬────┘  │
└─────────┼────────────────┼─────────────┼────────────┼───────┘
          │                │             │            │
  ┌───────▼──────┐  ┌──────▼──────┐  ┌──▼────────────▼─────┐
  │  Ingestion   │  │  RAG Chain  │  │      Analytics       │
  │  ─────────── │  │  ─────────  │  │  ──────────────────  │
  │  EdgarClient │  │  LangChain  │  │  FilingSentiment      │
  │  FilingParser│  │  ChatOpenAI │  │  Correlations         │
  │  FilingEmbed │  │  ChromaDB   │  │  TopicModeler         │
  │  FredClient  │  │  Retriever  │  │                      │
  └──────┬───────┘  └──────┬──────┘  └──────────────────────┘
         │                 │
  ┌──────▼─────────────────▼──────┐
  │          Storage              │
  │  ─────────────────────────    │
  │  ChromaDB  (./data/chroma/)   │
  │  Parquet   (./data/fred/)     │
  │  Parquet   (sentiment_scores) │
  └───────────────────────────────┘
         │
  ┌──────▼──────────────────┐
  │   External APIs         │
  │  SEC EDGAR REST API     │
  │  FRED API               │
  │  OpenAI Embeddings/LLM  │
  └─────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)
- SEC EDGAR access (free, but requires a descriptive User-Agent string)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd sec_fred_rag_app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI secret key | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `FRED_API_KEY` | FRED API key | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `SEC_USER_AGENT` | `"AppName your@email.com"` | [www.sec.gov/developer](https://www.sec.gov/developer) |

> **SEC EDGAR note:** EDGAR rate-limits requests without a proper User-Agent. Set it to something like `"MyResearch your@email.com"`. The application enforces a 0.11-second delay between requests to stay within the 10 req/sec limit.

---

## Running the Ingestion Pipeline

### Ingest FRED macro data only

```bash
python run_ingestion.py --fred-only
```

### Ingest specific tickers (10-K, 3 most recent filings each)

```bash
python run_ingestion.py --tickers AAPL MSFT GOOGL --form 10-K --limit 3
```

### Ingest 10-Q filings

```bash
python run_ingestion.py --tickers JPM BAC --form 10-Q --limit 5
```

### Full ingestion (FRED + SEC for default tickers)

```bash
python run_ingestion.py
```

### Force re-ingest (overwrite existing ChromaDB entries)

```bash
python run_ingestion.py --tickers AAPL --force
```

---

## Launching the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
sec_fred_rag_app/
├── app.py                        # Streamlit 4-tab application
├── run_ingestion.py              # CLI ingestion pipeline
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── edgar_client.py           # SEC EDGAR REST wrapper (CIK lookup, filing fetch)
│   ├── filing_parser.py          # HTML → named sections (MD&A, Risk Factors, etc.)
│   ├── embedder.py               # Chunking + OpenAI embeddings + ChromaDB upsert
│   └── fred_client.py            # FRED API: 17 macro series, Parquet storage
│
├── rag/
│   └── chain.py                  # LangChain RAG chain with metadata-filtered retrieval
│
├── analytics/
│   ├── sentiment.py              # VADER sentiment with custom financial lexicon
│   ├── correlations.py           # Sentiment × macro Pearson correlation analysis
│   └── topics.py                 # TF-IDF + LDA topic modeling
│
├── tests/
│   ├── test_parser.py            # Unit tests: section extraction, chunking
│   ├── test_fred.py              # Unit tests: compute_features, FRED catalogue
│   └── test_sentiment.py        # Unit tests: score_text, build_sentiment_series
│
└── data/                         # Auto-created at runtime
    ├── chroma/                   # ChromaDB persistent vector store
    ├── fred/                     # Individual series Parquet + macro_panel.parquet
    └── sentiment_scores.parquet  # Filing sentiment time series
```

---

## Data Science Components

### Sentiment Analysis

We use **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based model that does not require labelled training data and performs well on short financial text. The base VADER lexicon is augmented with ~40 domain-specific financial terms (`analytics/sentiment.py::FINANCIAL_LEXICON`):

- **Strong negatives**: `impairment (-2.5)`, `going concern (-3.5)`, `restatement (-3.0)`, `fraud (-3.5)`, `layoffs (-2.0)`
- **Strong positives**: `outperform (+2.5)`, `profitability (+1.5)`, `beat (+2.0)`, `resilient (+1.5)`

Scores are computed per filing section (MD&A, Risk Factors, Financials), creating a time-series panel with VADER's four metrics: `pos`, `neu`, `neg`, and `compound` (normalised to [-1, 1]).

**Expected finding:** Risk Factors sections consistently score the most negative; MD&A sections exhibit more variance correlated with business performance.

### FRED Feature Engineering

Seventeen macro series are fetched, resampled to monthly frequency, and engineered into a wide panel (`ingestion/fred_client.py`):

- **YoY % change** (`{series}_yoy`): captures trend direction over 12 months
- **MoM % change** (`{series}_mom`): captures momentum and inflection points
- **Yield curve** (`yield_curve = GS10 − GS2`): classic recession predictor and credit stress indicator

The panel covers six categories: Interest Rates, Inflation, Growth, Labor, Credit, and Sentiment (VIX).

### Topic Modeling

`analytics/topics.py` implements a **TF-IDF + LDA** pipeline using scikit-learn. Advantages over heavier alternatives (BERTopic, GPT-based clustering):

- No GPU required; fits in seconds on 10K filings
- Fully interpretable — topics are bags of weighted words
- Deterministic with `random_state` set

Each chunk is assigned a dominant topic; results are visualised as stacked bar charts of topic proportions per company.

### Correlation Analysis

`analytics/correlations.py` merges the sentiment time series with the macro panel using `pd.merge_asof` (backward nearest-date join), then computes **Pearson r** and p-values for every FRED series. Results are ranked by absolute correlation and colour-coded by sign. Typical findings:

- Filing sentiment is most **negatively** correlated with **VIX** and **credit spreads** (companies write more cautiously during market stress)
- Filing sentiment is most **positively** correlated with **equity-linked** indicators and **GDP growth**

---

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests from the project root
pytest tests/ -v

# Run a specific test file
pytest tests/test_sentiment.py -v
```

All tests run fully offline — no API keys are required.

---

## Resume-Ready Project Description

Built a full-stack financial intelligence platform that ingests SEC 10-K/10-Q filings via the EDGAR REST API and 17 FRED macroeconomic series, stores 1,536-dimensional OpenAI embeddings in a persistent ChromaDB vector store, and exposes a LangChain GPT-4o-mini RAG chat interface with metadata-filtered retrieval scoped by ticker, section, and date. The analytics layer combines VADER sentiment scoring (augmented with a custom 40-term financial lexicon) with Pearson correlation analysis against the macro panel, quantifying how filing tone co-moves with VIX, credit spreads, and GDP growth. A four-tab Streamlit dashboard surfaces company ingestion, RAG chat, FRED visualisations with NBER recession shading, and sentiment heatmaps—all backed by a fully offline pytest suite testing the parsing, FRED feature engineering, and sentiment pipelines.
