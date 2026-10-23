# Mini-Crosscheck

Mini-Crosscheck is a local, plain-Python RAG pipeline for engineering documents. It is built in stages so each transformation can be inspected before moving on. Source PDFs in `data/` are treated as read-only.

## Project layout

```text
src/                 Pipeline modules and CLI
data/                Source PDFs and sample_questions.json (read-only inputs)
tests/               Focused unit tests
outputs/              Generated parsed data, indexes, logs, and evaluations
config.yaml           Shared pipeline settings
requirements.txt      Python dependencies
```

## Dependencies

Python 3.11 or newer is required. The full pipeline uses `pdfplumber` for PDF text and tables, `sentence-transformers` with `BAAI/bge-small-en-v1.5` for embeddings, ChromaDB for vector storage, `rank-bm25` for lexical retrieval, `requests` for local Ollama and OpenAI-compatible HTTP APIs, and Streamlit for the interface. `pytest` runs the tests.

Create and activate a virtual environment, then install the dependencies:

```powershell
C:/msys2/ucrt64/bin/python.exe -m venv .venv
./.venv/bin/python.exe -m pip install -r requirements.txt
```

## Stage 1: Ingestion

The ingestion module emits one JSON Lines record per PDF page. Each record contains page-level document metadata, prose text, and separately extracted tables whose rows repeat their column labels. Header metadata is preferred; the revision falls back to a `RevX` token in the filename.

```powershell
./.venv/bin/python.exe -m src.ingest --input data --output outputs/parsed.jsonl
```

Run the ingestion unit tests with `./.venv/bin/python.exe -m pytest tests/test_ingest.py`.

## Stage 2: Chunking

Chunking reads the page records from Stage 1. Prose is split recursively at paragraph, line, sentence, and word boundaries with configured overlap. Tables remain complete when they fit; larger tables are grouped by labeled rows. Each output chunk carries the source metadata and a deterministic ID.

```powershell
./.venv/bin/python.exe -m src.chunk
```

Override settings for an experiment with `--chunk-size` and `--overlap`. The command prints text/table counts and chunk lengths. Run the focused tests with `./.venv/bin/python.exe -m pytest tests/test_chunk.py`.

## Stage 3: Indexing

The indexing step reads one validated chunk list and builds both a normalized sentence-transformer embedding collection in Chroma and a serialized BM25 index over the same ordered chunks. Re-indexing clears the configured collection before adding the current corpus, preventing stale chunk IDs from earlier runs.

```powershell
./.venv/bin/python.exe -m src.index
```

The configured index locations are `outputs/chroma/` and `outputs/bm25.pkl`. Run the focused tests with `./.venv/bin/python.exe -m pytest tests/test_index.py`.

## Stage 4: Retrieval

The retriever supports vector-only, BM25-only, and hybrid Reciprocal Rank Fusion. Use `--revision` for an exact revision filter or `--latest-only` to retain the highest observed revision per document number. Optional cross-encoder reranking is enabled with `--rerank` or the config setting.

```powershell
./.venv/bin/python.exe -m src.retrieve "What is the maximum process temperature?" --mode hybrid --revision C
./.venv/bin/python.exe -m src.retrieve "What is the maximum process temperature?" --mode hybrid --latest-only --rerank
```

Results include chunk text, source metadata, and retrieval scores. Run the focused tests with `./.venv/bin/python.exe -m pytest tests/test_retrieve.py`.

## Stage 5: Generation

Generation checks the normalized retrieval abstention score before contacting an LLM. If evidence is empty or below `generation.abstention_threshold`, it returns exactly `INSUFFICIENT EVIDENCE`. The prompt requires evidence-only answers, citations on factual claims, revision identification, and disclosure of disagreements between revisions with both citations.

Ollama is the default backend. Set `MINI_CROSSCHECK_LLM_BACKEND=openai_compatible` and provide `OPENAI_BASE_URL` and `OPENAI_MODEL` to select an OpenAI-compatible endpoint; `OPENAI_API_KEY` is optional for local servers. Run the focused tests with `./.venv/bin/python.exe -m pytest tests/test_generate.py`.

## Stage 6: Evaluation

The supplied ten questions remain unchanged. Ten PDF-grounded proposals for review are in `data/proposed_questions.json`; the evaluation includes them by default. It compares vector, BM25, hybrid, and hybrid-plus-reranker retrieval using source-file-level Recall@k, MRR, and nDCG, then reports correct abstention and a simple citation/value faithfulness rate.

```powershell
./.venv/bin/python.exe -m src.evaluate --limit 2
./.venv/bin/python.exe -m src.evaluate
```

The full evaluation writes `outputs/eval_results.csv` and makes model calls for questions that pass the score threshold. Use `--exclude-proposed` to run only the supplied questions. Run the focused tests with `./.venv/bin/python.exe -m pytest tests/test_evaluate.py`.

## Stage 7: Interface

The CLI and Streamlit app share a query service. Each question appends the query settings, answer, retrieved chunk text, metadata, and scores to `outputs/query_log.jsonl`.

```powershell
./.venv/bin/python.exe -m src.ask "What is the maximum process temperature?" --mode hybrid --revision C
./.venv/bin/python.exe -m src.ask "What is the maximum process temperature?" --mode hybrid --latest-only
./.venv/bin/python.exe -m streamlit run src/app.py
```

The app shows the answer, retrieved evidence, retrieval scores, and a warning when multiple revisions of one document are present. Run the service tests with `./.venv/bin/python.exe -m pytest tests/test_ask.py`.