"""Mini-Crosscheck · Engineering Document QA Interface"""

from __future__ import annotations

import html as _html
import csv
import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.ask import create_service

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config.yaml"

_EXAMPLES: list[tuple[str, str]] = [
    ("Max process temp?", "What is the maximum process temperature in Revision C of the HX-200 manual?"),
    ("P-204 seal failure?", "Why did the seal on Pump P-204 fail in November 2024?"),
    ("6061-T6 yield strength?", "What is the yield strength of 6061-T6 at 100 deg C?"),
    ("Vibration limit?", "What is the vibration limit for acceptable long-term operation?"),
]

_STAGES: list[dict[str, Any]] = [
    {"name": "Ingestion", "desc": "Parse PDFs into page-level JSONL records with pdfplumber. Extracts metadata, prose, and tables separately.", "cmd": "python -m src.ingest --input data --output outputs/parsed.jsonl", "check": "outputs/parsed.jsonl"},
    {"name": "Chunking", "desc": "Recursive text splitting with configured overlap. Tables stay intact when they fit the chunk window.", "cmd": "python -m src.chunk", "check": "outputs/chunks.jsonl"},
    {"name": "Indexing", "desc": "Sentence-transformer embeddings into ChromaDB and serialized BM25 index over the same ordered corpus.", "cmd": "python -m src.index", "check": "outputs/chroma"},
    {"name": "Retrieval", "desc": "Hybrid RRF combining dense vector similarity and BM25 lexical scores. Supports revision filtering and cross-encoder reranking.", "cmd": None, "check": None},
    {"name": "Generation", "desc": "Score-gated answer generation via Ollama or any OpenAI-compatible endpoint. Abstains below the configured threshold.", "cmd": None, "check": None},
    {"name": "Evaluation", "desc": "Recall@k, MRR, nDCG, and citation faithfulness across curated engineering questions.", "cmd": "python -m src.evaluate", "check": "outputs/eval_results.csv"},
    {"name": "Interface", "desc": "This app. Shared query service with the CLI. Full audit logging to JSONL.", "cmd": "streamlit run src/app.py", "check": "outputs/query_log.jsonl"},
]


# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────

_FONTS = """<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">"""

_CSS = """
/* ── Base typography ────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stText,
p, span, li, label, div {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    letter-spacing: -0.008em;
}

code, pre, .stCode {
    font-family: 'JetBrains Mono', 'Cascadia Code', monospace !important;
}

/* ── Layout ─────────────────────────────────────── */
.block-container {
    max-width: 1120px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── Hide Streamlit chrome ──────────────────────── */
#MainMenu, header, footer { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #11141b !important;
    border-right: 1px solid rgba(255,255,255,0.035) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
}

/* ── Form inputs ────────────────────────────────── */
.stTextInput > div > div > input {
    background: #161a23 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 7px !important;
    color: #e0dbd4 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 0.8rem !important;
    transition: border-color 0.15s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c97d38 !important;
    box-shadow: 0 0 0 1px rgba(201,125,56,0.2) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4a473f !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #161a23 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 7px !important;
    color: #e0dbd4 !important;
}

/* ── Labels ─────────────────────────────────────── */
.stSelectbox label, .stNumberInput label,
.stTextInput label, .stCheckbox label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #7d7872 !important;
}

/* ── Primary button / form submit ───────────────── */
.stFormSubmitButton > button,
button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background: #c97d38 !important;
    color: #0e1117 !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.015em !important;
    padding: 0.52rem 1.5rem !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
}
.stFormSubmitButton > button:hover,
button[kind="primary"]:hover {
    background: #b5702e !important;
    transform: translateY(-1px) !important;
}
.stFormSubmitButton > button:active { transform: translateY(0) !important; }

/* ── Secondary buttons ──────────────────────────── */
.stButton > button {
    background: transparent !important;
    color: #7d7872 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.3rem 0.65rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    color: #e0dbd4 !important;
    border-color: rgba(201,125,56,0.35) !important;
    background: rgba(201,125,56,0.05) !important;
}

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #4e4a44 !important;
    padding: 0.6rem 1.1rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #7d7872 !important; }
.stTabs [aria-selected="true"] {
    color: #e0dbd4 !important;
    border-bottom-color: #c97d38 !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: #c97d38 !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Metrics ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161a23 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 9px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #4e4a44 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    color: #e0dbd4 !important;
}

/* ── Spinner ────────────────────────────────────── */
.stSpinner > div > div { border-top-color: #c97d38 !important; }

/* ── Expander ───────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    background: #161a23 !important;
    border-radius: 7px !important;
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.13); }

/* ── Custom HTML classes ────────────────────────── */

.brand-wrap { margin-bottom: 1.4rem; }
.brand-row { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.25rem; }
.brand-mark {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #c97d38 0%, #9e5a1d 100%);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; color: #fff; flex-shrink: 0;
}
.brand-title {
    font-size: 1.05rem; font-weight: 700; color: #e0dbd4;
    letter-spacing: -0.025em;
}
.brand-sub {
    font-size: 0.72rem; color: #4e4a44; letter-spacing: 0.04em;
    padding-left: 0.1rem;
}
.brand-rule {
    width: 36px; height: 2px; background: #c97d38;
    border-radius: 1px; margin-top: 0.7rem;
}

.sidebar-label {
    font-size: 0.68rem; font-weight: 600; color: #4e4a44;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.2rem 0 0.45rem; padding-bottom: 0.35rem;
    border-bottom: 1px solid rgba(255,255,255,0.035);
}

.corpus-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #6b665f;
    padding: 0.22rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.02);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.page-title {
    font-size: 1.5rem; font-weight: 700; color: #e0dbd4;
    letter-spacing: -0.03em; margin-bottom: 0.15rem;
}
.page-accent-bar {
    width: 42px; height: 2.5px; background: #c97d38;
    border-radius: 2px; margin-bottom: 0.35rem;
}
.page-desc {
    font-size: 0.88rem; color: #5a5650; line-height: 1.55;
    margin-bottom: 1.2rem;
}

/* ── Answer card ── */
.answer-wrap {
    background: #161a23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin: 0.8rem 0 1.2rem;
}
.answer-body {
    font-size: 0.96rem; line-height: 1.75; color: #e0dbd4;
}
.answer-footer {
    display: flex; align-items: center; gap: 0.7rem;
    margin-top: 0.9rem; padding-top: 0.7rem;
    border-top: 1px solid rgba(255,255,255,0.035);
}
.conf-label {
    font-size: 0.72rem; color: #4e4a44;
    text-transform: uppercase; letter-spacing: 0.06em;
    flex-shrink: 0;
}
.conf-track {
    flex: 1; height: 5px;
    background: rgba(255,255,255,0.035);
    border-radius: 3px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.35s ease;
}
.conf-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; font-weight: 500;
    flex-shrink: 0; min-width: 3rem; text-align: right;
}

/* ── Evidence card ── */
.ev-card {
    background: #161a23;
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 3px solid rgba(201,125,56,0.35);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    transition: border-left-color 0.15s ease;
}
.ev-card:hover { border-left-color: #c97d38; }
.ev-top {
    display: flex; align-items: center;
    justify-content: space-between;
    flex-wrap: wrap; gap: 0.35rem;
    margin-bottom: 0.45rem;
}
.ev-src {
    font-size: 0.82rem; font-weight: 600; color: #c97d38;
}
.ev-tags { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.ev-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 500; color: #4e4a44;
    background: rgba(255,255,255,0.03);
    padding: 0.12rem 0.45rem; border-radius: 3px;
}
.ev-body {
    font-size: 0.84rem; line-height: 1.6; color: #7d7872;
    margin-top: 0.3rem;
    max-height: 200px; overflow-y: auto;
}
.ev-scores {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; color: #4e4a44;
    margin-top: 0.4rem;
}

/* ── Section heading ── */
.sec-label {
    font-size: 0.72rem; font-weight: 600; color: #4e4a44;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 1.4rem 0 0.5rem;
}

/* ── Revision warning ── */
.rev-warn {
    background: rgba(212,165,70,0.06);
    border: 1px solid rgba(212,165,70,0.18);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin: 0.7rem 0;
    font-size: 0.84rem; color: #d4a546;
    display: flex; align-items: flex-start; gap: 0.45rem;
    line-height: 1.5;
}

/* ── Document card ── */
.doc-card {
    background: #161a23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 9px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.15s ease;
}
.doc-card:hover { border-color: rgba(255,255,255,0.1); }
.doc-name {
    font-size: 0.88rem; font-weight: 600; color: #e0dbd4;
    margin-bottom: 0.2rem;
}
.doc-meta {
    font-size: 0.76rem; color: #4e4a44; line-height: 1.5;
}

/* ── Pipeline stage ── */
.stage-row {
    background: #161a23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 9px;
    padding: 1rem 1.2rem;
    margin: 0.45rem 0;
    display: flex; align-items: flex-start; gap: 0.9rem;
}
.stage-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 500; color: #c97d38;
    background: rgba(201,125,56,0.08);
    width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 5px; flex-shrink: 0;
}
.stage-body { flex: 1; }
.stage-name {
    font-size: 0.88rem; font-weight: 600; color: #e0dbd4;
    margin-bottom: 0.15rem;
}
.stage-desc {
    font-size: 0.8rem; color: #5a5650; line-height: 1.5;
}
.stage-cmd {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #6b665f;
    background: rgba(255,255,255,0.025);
    padding: 0.25rem 0.5rem; border-radius: 4px;
    margin-top: 0.4rem; display: inline-block;
}
.stage-dot {
    width: 2px; height: 12px;
    background: rgba(255,255,255,0.04);
    margin: 0 0 0 12px;
}

/* ── Empty state ── */
.empty-box {
    text-align: center; padding: 2.5rem 1rem; color: #4e4a44;
}
.empty-icon { font-size: 1.8rem; margin-bottom: 0.3rem; opacity: 0.35; }
.empty-text { font-size: 0.88rem; }

/* ── Technical workbench theme ──────────────────── */
:root {
    --paper: #f3f6f2;
    --surface: #ffffff;
    --ink: #1b2928;
    --muted: #647370;
    --line: #dce5e0;
    --teal: #176b63;
    --coral: #b8503d;
    --gold: #a8761f;
}
html, body, [class*="css"], .stMarkdown, .stText,
p, span, li, label, div { letter-spacing: 0 !important; }
.stApp, [data-testid="stAppViewContainer"] { background: var(--paper) !important; color: var(--ink) !important; }
.block-container { max-width: 1220px !important; padding-top: 1.7rem !important; }
section[data-testid="stSidebar"] { background: #e8efeb !important; border-right: 1px solid var(--line) !important; }
section[data-testid="stSidebar"] * { color: var(--ink); }
.brand-title, .page-title, .doc-name, .stage-name { color: var(--ink) !important; }
.brand-sub, .page-desc, .stage-desc, .doc-meta, .sec-label { color: var(--muted) !important; }
.brand-mark { background: var(--teal) !important; border-radius: 5px !important; }
.brand-rule, .page-accent-bar { background: var(--coral) !important; }
.sidebar-label { color: var(--teal) !important; border-color: var(--line) !important; }
.corpus-item { color: var(--muted) !important; border-color: var(--line) !important; }
.stTextInput > div > div > input, .stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--surface) !important; color: var(--ink) !important;
    border-color: var(--line) !important; border-radius: 5px !important;
}
.stTextInput > div > div > input:focus { border-color: var(--teal) !important; box-shadow: 0 0 0 1px rgba(23,107,99,.16) !important; }
.stTextInput > div > div > input::placeholder { color: #899691 !important; }
.stSelectbox label, .stNumberInput label, .stTextInput label, .stCheckbox label { color: var(--muted) !important; }
.stFormSubmitButton > button, button[kind="primary"], [data-testid="stFormSubmitButton"] > button {
    background: var(--teal) !important; color: white !important; border-radius: 5px !important;
}
.stFormSubmitButton > button:hover, button[kind="primary"]:hover { background: #10584f !important; }
.stButton > button { color: var(--muted) !important; border-color: var(--line) !important; border-radius: 5px !important; }
.stButton > button:hover { color: var(--teal) !important; border-color: var(--teal) !important; background: #edf5f1 !important; }
.stTabs [data-baseweb="tab-list"] { border-color: var(--line) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; border-bottom-color: var(--teal) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal) !important; }
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 5px !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--ink) !important; }
.answer-wrap, .ev-card, .doc-card, .stage-row { background: var(--surface) !important; border-color: var(--line) !important; border-radius: 6px !important; }
.answer-body { color: var(--ink) !important; }
.ev-src { color: var(--teal) !important; }
.ev-body, .ev-scores, .ev-tag { color: var(--muted) !important; }
.ev-tag { background: #eff3f0 !important; }
.stage-num { color: var(--teal) !important; background: #e9f3ef !important; }
.stage-cmd { color: #52625e !important; background: #f1f4f1 !important; }
.rev-warn { color: #805d17 !important; background: #fbf5e6 !important; border-color: #ead8a7 !important; }
.empty-box { color: var(--muted) !important; }
.empty-icon { opacity: .65; }
.status-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .55rem; border:1px solid var(--line); border-radius:4px; background:#fff; color:var(--muted); font-size:.74rem; font-weight:600; }
.status-dot { width:7px; height:7px; border-radius:50%; background:var(--gold); }
.status-dot.ready { background:var(--teal); }
.status-dot.pending { background:var(--coral); }
.stage-state { color:var(--muted); font-size:.72rem; font-weight:600; white-space:nowrap; }
"""


# ──────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def _get_service(config_path: str) -> Any:
    """Load and cache the shared pipeline service."""
    return create_service(Path(config_path))


@st.cache_data
def _scan_corpus(data_dir: str, parsed_file: str) -> tuple[list[dict], list[dict]]:
    """Return (pdf_files, parsed_docs) for the corpus overview."""
    pdfs: list[dict[str, Any]] = []
    for path in sorted(Path(data_dir).glob("*.pdf")):
        pdfs.append({"name": path.name, "kb": path.stat().st_size / 1024})

    docs: dict[str, dict[str, Any]] = {}
    pf = Path(parsed_file)
    if pf.exists():
        for line in pf.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record["metadata"]
            entry = docs.setdefault(meta["doc_name"], {
                "name": meta["doc_name"],
                "number": meta["doc_number"],
                "rev": meta["revision"],
                "date": meta["date"],
                "pages": 0,
                "tables": 0,
            })
            entry["pages"] += 1
            entry["tables"] += len(record.get("tables", []))

    return pdfs, list(docs.values())


def _count_records(path: str) -> int:
    """Count non-empty JSONL records without loading the corpus into memory."""
    source = Path(path)
    if not source.exists():
        return 0
    with source.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _read_evaluation(path: str = "outputs/eval_results.csv") -> list[dict[str, Any]]:
    """Read saved evaluation metrics for the quality overview."""
    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, Any]] = []
        for raw_row in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if key == "mode" or value is None:
                    row[key] = value
                    continue
                try:
                    row[key] = int(value)
                except ValueError:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
        return rows


# ──────────────────────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    """Map retrieval confidence to an appropriate indicator colour."""
    if score >= 0.6:
        return "#5b9e7a"
    if score >= 0.3:
        return "#d4a546"
    return "#c25550"


def _answer_html(result: dict[str, Any]) -> str:
    """Build the styled answer card with a confidence bar."""
    score = result["score"]
    colour = _score_color(score)
    pct = max(2, min(100, int(score * 100)))
    body = _html.escape(result["answer"]).replace("\n", "<br>")
    return (
        '<div class="answer-wrap">'
        f'<div class="answer-body">{body}</div>'
        '<div class="answer-footer">'
        '<span class="conf-label">Confidence</span>'
        '<div class="conf-track">'
        f'<div class="conf-fill" style="width:{pct}%;background:{colour};"></div>'
        '</div>'
        f'<span class="conf-val" style="color:{colour};">{score:.1%}</span>'
        '</div></div>'
    )


def _evidence_block(chunks: list[dict[str, Any]]) -> str:
    """Render all evidence cards as a single HTML string."""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc = _html.escape(str(meta.get("doc_name", "Unknown")))
        rev = _html.escape(str(meta.get("revision", "?")))
        page = meta.get("page", "?")
        sc = chunk.get("score", 0.0)
        text = _html.escape(chunk.get("text", "")).replace("\n", "<br>")

        scores_dict = chunk.get("scores", {})
        scores_str = ""
        if scores_dict:
            scores_str = (
                '<div class="ev-scores">'
                + " · ".join(f"{k}: {v:.4f}" for k, v in scores_dict.items())
                + '</div>'
            )

        parts.append(
            '<div class="ev-card">'
            '<div class="ev-top">'
            f'<span class="ev-src">{idx}. {doc}</span>'
            '<div class="ev-tags">'
            f'<span class="ev-tag">Rev {rev}</span>'
            f'<span class="ev-tag">p. {page}</span>'
            f'<span class="ev-tag">{sc:.4f}</span>'
            '</div></div>'
            f'<div class="ev-body">{text}</div>'
            f'{scores_str}'
            '</div>'
        )
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Render the full Mini-Crosscheck interface."""

    st.set_page_config(
        page_title="Mini-Crosscheck · Engineering QA",
        page_icon="⚙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_FONTS, unsafe_allow_html=True)
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────

    with st.sidebar:
        st.markdown(
            '<div class="brand-wrap">'
            '<div class="brand-row">'
            '<div class="brand-mark">⚙</div>'
            '<div class="brand-title">Mini-Crosscheck</div>'
            '</div>'
            '<div class="brand-sub">Engineering Document QA</div>'
            '<div class="brand-rule"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Retrieval</div>', unsafe_allow_html=True)
        mode = st.selectbox("Search mode", ("hybrid", "vector", "bm25"), index=0)
        top_k = st.number_input("Results to fetch", min_value=1, max_value=20, value=5)
        revision = st.text_input("Revision filter", placeholder="e.g. C")
        c1, c2 = st.columns(2)
        with c1:
            latest_only = st.checkbox("Latest only")
        with c2:
            rerank = st.checkbox("Rerank")

        config_path = st.text_input("Config file", value=_DEFAULT_CONFIG)

        st.markdown('<div class="sidebar-label">Corpus</div>', unsafe_allow_html=True)
        pdfs, docs = _scan_corpus("data", "outputs/parsed.jsonl")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Documents", len(pdfs))
        with m2:
            total_pages = sum(d["pages"] for d in docs) if docs else 0
            st.metric("Pages", total_pages if total_pages else "—")

        for pdf in pdfs:
            st.markdown(
                f'<div class="corpus-item">{_html.escape(pdf["name"])}</div>',
                unsafe_allow_html=True,
            )

    # ── Page header ─────────────────────────────────────────

    st.markdown(
        '<div class="page-title">Engineering answers, grounded in source pages.</div>'
        '<div class="page-accent-bar"></div>'
        '<div class="page-desc">'
        'Search technical documents with page-level citations, revision awareness, and measurable retrieval quality.'
        '</div>',
        unsafe_allow_html=True,
    )

    index_ready = Path("outputs/chroma").exists() and Path("outputs/bm25.pkl").exists()
    eval_rows = _read_evaluation()
    st.markdown(
        '<div class="status-pill"><span class="status-dot '
        f'{"ready" if index_ready else "pending"}'
        '"></span>'
        f'{"Hybrid index ready" if index_ready else "Index not built yet"}'
        '</div>',
        unsafe_allow_html=True,
    )
    chunks_total = _count_records("outputs/chunks.jsonl")
    total_pages = sum(doc["pages"] for doc in docs)
    eval_question_count = eval_rows[0].get("questions", 0) if eval_rows else 0
    metric_cols = st.columns(4)
    for column, label, value in zip(
        metric_cols,
        ("Source PDFs", "Pages parsed", "Retrieval chunks", "Questions evaluated"),
        (len(pdfs), total_pages, chunks_total, eval_question_count),
    ):
        with column:
            st.metric(label, value)

    # ── Tabs ────────────────────────────────────────────────

    tab_query, tab_docs, tab_quality, tab_pipeline = st.tabs(
        ["Ask", "Corpus", "Evaluation", "Pipeline"]
    )

    # ── QUERY TAB ───────────────────────────────────────────

    with tab_query:
        service = None
        service_err = None
        try:
            service = _get_service(config_path)
        except Exception as exc:
            service_err = str(exc)

        if service_err:
            st.markdown(
                '<div class="rev-warn">'
                f'⚠&nbsp;&nbsp;Pipeline not ready — {_html.escape(service_err)}<br>'
                'Run the indexing stage first. See the <b>Pipeline</b> tab for commands.'
                '</div>',
                unsafe_allow_html=True,
            )

        with st.form("query_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="Ask about specifications, failures, materials, or maintenance…",
            )
            submitted = st.form_submit_button("Search documents")

        # Example queries
        st.caption("Try an example →")
        example_query: str | None = None
        ex_cols = st.columns(len(_EXAMPLES))
        for i, (label, full) in enumerate(_EXAMPLES):
            with ex_cols[i]:
                if st.button(label, key=f"ex_{i}", use_container_width=True):
                    example_query = full

        # Decide what to run
        run_query: str | None = None
        if submitted and question.strip():
            run_query = question.strip()
        elif example_query:
            run_query = example_query

        if run_query:
            if service:
                with st.spinner("Retrieving evidence and generating answer…"):
                    try:
                        result = service.ask(
                            run_query,
                            mode=mode,
                            top_k=int(top_k),
                            revision=revision.strip() or None,
                            latest_only=latest_only,
                            rerank=rerank,
                        )
                        st.session_state["last_result"] = result
                        st.session_state["last_question"] = run_query
                    except Exception as exc:
                        st.error(f"Query failed: {type(exc).__name__}: {exc}")
            else:
                st.error("Service unavailable. Complete the pipeline setup first.")
        elif submitted:
            st.warning("Enter a question to search the documents.")

        # Display results
        result = st.session_state.get("last_result")
        if result:
            last_q = st.session_state.get("last_question", "")
            if last_q:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#5a5650;margin-top:0.4rem;">'
                    f'<b>Q:</b> {_html.escape(last_q)}</div>',
                    unsafe_allow_html=True,
                )

            if result.get("revision_variants"):
                details = "; ".join(
                    f"{v['doc_number']}: revisions {', '.join(v['revisions'])}"
                    for v in result["revision_variants"]
                )
                st.markdown(
                    f'<div class="rev-warn">⚠&nbsp;&nbsp;Multiple revisions retrieved '
                    f'— compare cited values carefully. {_html.escape(details)}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(_answer_html(result), unsafe_allow_html=True)

            chunks = result.get("chunks", [])
            if chunks:
                st.markdown(
                    f'<div class="sec-label">Retrieved evidence ({len(chunks)})</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_evidence_block(chunks), unsafe_allow_html=True)

        elif not submitted and not example_query:
            st.markdown(
                '<div class="empty-box">'
                '<div class="empty-icon">🔍</div>'
                '<div class="empty-text">Ask a question to search across your engineering documents.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── DOCUMENTS TAB ───────────────────────────────────────

    with tab_docs:
        if docs:
            doc_cols = st.columns(3)
            for i, doc in enumerate(docs):
                with doc_cols[i % 3]:
                    table_note = f" · {doc['tables']} tables" if doc["tables"] else ""
                    st.markdown(
                        '<div class="doc-card">'
                        f'<div class="doc-name">{_html.escape(doc["name"])}</div>'
                        '<div class="doc-meta">'
                        f'Doc {_html.escape(doc["number"])} · Rev {_html.escape(doc["rev"])}<br>'
                        f'{_html.escape(doc["date"])} · {doc["pages"]} pages{table_note}'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Run the ingestion stage to populate document metadata here.")

    # ── EVALUATION TAB ──────────────────────────────────────

    with tab_quality:
        st.markdown('<div class="sec-label">Retrieval quality</div>', unsafe_allow_html=True)
        if eval_rows:
            modes = [row for row in eval_rows if row.get("mode")]
            best_mode = max(modes, key=lambda row: float(row.get("ndcg", 0)), default={})
            q1, q2, q3 = st.columns(3)
            q1.metric("Questions", eval_question_count)
            q2.metric("Modes compared", len(modes))
            q3.metric(
                "Best nDCG",
                f'{best_mode.get("mode", "—")} · {float(best_mode.get("ndcg", 0)):.3f}'
                if best_mode else "—",
            )

            chart_rows = [
                {
                    "Mode": row.get("mode", ""),
                    "Recall@k": float(row.get("recall_at_k", 0)),
                    "MRR": float(row.get("mrr", 0)),
                    "nDCG": float(row.get("ndcg", 0)),
                }
                for row in modes
            ]
            st.bar_chart(chart_rows, x="Mode", y=["Recall@k", "MRR", "nDCG"], height=280)
            st.dataframe(eval_rows, use_container_width=True, hide_index=True)
            st.caption(
                "Retrieval metrics are source-file level. Faithfulness checks verify citation provenance "
                "and numeric claims against the cited evidence."
            )
        else:
            st.markdown(
                '<div class="empty-box"><div class="empty-text">'
                'No evaluation results have been saved yet.'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Run the benchmark to compare vector, BM25, hybrid, and reranked retrieval.")
            st.code("python -m src.evaluate", language="powershell")

    # ── PIPELINE TAB ────────────────────────────────────────

    with tab_pipeline:
        st.markdown(
            '<div class="page-desc" style="margin-top:0.2rem;">'
            'Seven-stage pipeline from raw PDFs to cited, revision-aware answers.'
            '</div>',
            unsafe_allow_html=True,
        )

        for i, stage in enumerate(_STAGES):
            check_path = stage["check"]
            if stage["name"] == "Interface":
                status = "Available"
            elif stage["name"] == "Indexing":
                status = "Ready" if index_ready else "Pending"
            elif stage["name"] in {"Retrieval", "Generation"}:
                status = "Ready" if index_ready else "Waiting for index"
            elif check_path:
                status = "Ready" if Path(check_path).exists() else "Pending"
            else:
                status = "Not run"

            cmd_html = ""
            if stage["cmd"]:
                cmd_html = f'<div class="stage-cmd">{_html.escape(stage["cmd"])}</div>'

            st.markdown(
                '<div class="stage-row">'
                f'<div class="stage-num">{i + 1}</div>'
                '<div class="stage-body">'
                f'<div class="stage-name">{_html.escape(stage["name"])}</div>'
                f'<div class="stage-desc">{_html.escape(stage["desc"])}</div>'
                f'{cmd_html}'
                '</div>'
                f'<div class="stage-state">{_html.escape(status)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if i < len(_STAGES) - 1:
                st.markdown('<div class="stage-dot"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()