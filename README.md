# LexMacedonica

<p align="center"><img src="static/img/logo-top-nobg.jpg" alt="LexMacedonica" width="300"></p>

**An AI legal assistant for Macedonian law** — a Retrieval-Augmented Generation (RAG) web app
built as a student project. You describe a real-life situation in Macedonian; it answers with
the most likely outcome, an honest **probability computed from real court decisions** (not
invented by the model), and citations of the most similar past cases.

> ⚖️ Informational only — nothing this app produces is legal advice.

## The four interfaces

| Tab | What it does |
|---|---|
| **Правен асистент** | Describe your situation → likely outcome + probability % derived from similar past cases, with cited decisions (court, case number, date, summary, full text on click) |
| **Симулација** | Four AI agents (judge, both lawyers, narrator) play out your scenario as a live courtroom — every line **streams token by token** |
| **Адвокати** | Law-first answers for professionals with exact article citations (закон + член), case law as support, and the model's **reasoning visible** in a collapsible panel |
| **Администрација** | Document anonymization (names, ЕМБГ, addresses, accounts…) — deterministic regex rules first, LLM for the rest, with a diff table explaining every replacement |

**Corpus**: 581 labor-dispute decisions (РО) from 19 basic courts scraped from the official
portal [sud.mk], with per-case metadata (court, date, outcome, judge, legal area, case
type/subtype, legal basis) + two laws (Закон за работните односи 2023, Закон за
облигационите односи) chunked **by article** so citations are exact.

## How answering works

```
question → deterministic Cyrillic checks → cheap Macedonian/meaning classifier
         (+ browser-side history → condensed only when history exists)
   │
   ▼
semantic cache ──hit──► reuse answer (skips retrieval, reranker and answer LLM)
   │ miss
   ▼
corpus-wide hybrid search: BM25 keywords + vector search over all chunks,
                           fused with Reciprocal Rank Fusion → top 30
   ▼
local cross-encoder: rerank the first 10 candidates → top-3 passages
   ▼
probability = deduplicate cases represented in the reranked 10, then take the
              relevance-weighted share of real outcomes among up to 5 cases
              (merits decisions only; < 3 comparable → no number)
   ▼
ONE LLM call with self-check folded into the prompt («САМОПРОВЕРКА» rule).
Too low similarity? → honest «Не знам», the model is never even called.
```

Why hybrid: vectors understand paraphrases («ме отпуштија» ≈ «престанок на работен однос»),
BM25 nails exact terms like «член 101». RRF cheaply merges both rankings, then the
multilingual cross-encoder reads each of the best ten question/passage pairs together.

## Stack

- **FastAPI** + Jinja2 + plain HTML/CSS/JS (no build step, no frontend framework)
- **ChromaDB** — one persistent store for vectors + texts + metadata
  (collections: case chunks, case summaries, law articles, semantic cache);
  metadata filtering at query time and metadata updates without re-embedding
- **BM25** (`rank_bm25`) built at startup from the same store — keyword leg of the hybrid search
- **Sentence Transformers** + `BAAI/bge-reranker-v2-m3` — local multilingual cross-encoder;
  loaded once per process and used only on ten candidates (RRF remains the safe fallback)
- **SQLite** (SQLAlchemy) — structured case metadata: outcome (regex-extracted from the
  dispositive), judge, legal area, case type/subtype, legal basis, LLM summary
- **OpenAI** `gpt-4o-mini` + `text-embedding-3-small` via LangChain — cheapest tier on
  purpose; embeddings are cached in `.npz` ledgers so re-runs never re-pay
- **slowapi** per-IP rate limiting on every LLM-calling endpoint (budget protection)
- **NDJSON streaming** for the courtroom simulation (plain `fetch` + reader, no SSE machinery)
- **PyMuPDF** for PDF text extraction (decisions are text-based, pre-anonymized by the court)

Total OpenAI spend for the whole project: **a few dollars** — tracked on the OpenAI dashboard.

## Run it locally

```bash
# 1. dependencies (Python 3.12+)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt          # Windows
# source .venv/bin/activate && pip install -r ...      # Linux/macOS

# The first accepted, non-cached question downloads the ~2.3 GB reranker.
# Later requests reuse both the downloaded files and one in-process model instance.

# 2. secrets
copy other\.env.example .env       # then put your OPENAI_API_KEY inside

# 3. data pipeline (one-time; skip if data/ already exists)
python -m app.scraper.run_scrape --prefix РО --limit 30   # download decisions from sud.mk
python -m app.ingest.build_index                          # PDF → text → SQLite → chunks → vectors
python -m app.ingest.summarize                            # per-case summaries (situation search)
python -m app.lawyer.download_laws                        # download the laws
python -m app.lawyer.build_laws                           # article-level law index

# 4. start
uvicorn main:app --reload
# app       → http://127.0.0.1:8000
# API docs  → http://127.0.0.1:8000/docs   (only /api/* endpoints — pages are excluded)
```

Or with Docker: `docker compose up --build` (needs `.env` and an already-built `data/`).

## API

HTML pages live at `/`, `/simulacija`, `/advokati`, `/administracija` and are deliberately
kept **out of the OpenAPI schema**; everything under `/api/*` is JSON and testable from
`/docs`:

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | main assistant (question + browser-kept history) |
| `GET  /api/case/{id}/text` | full decision text for a citation card |
| `POST /api/lawyer` | law-first professional answer with reasoning |
| `POST /api/simulate/turn` | one streamed courtroom turn (NDJSON: meta → token… → final) |
| `POST /api/anonymize` | document anonymization with replacement table |

## Scraping sud.mk (the fun part)

The court portal is an IBM WebSphere Portal: session-encoded URLs, sticky server-side search
state, and relative links that must be resolved against `<base href>` — naive scraping gets
silently empty pages. The scraper mimics a real browser: one `requests.Session` per filter
set, harvests *every* form field with its defaults, overrides only the filters, and parses
each result box including the collapsed «Повеќе податоци» section (judge, legal area, case
type, subtype, legal basis). Politeness delay of 2.5 s between requests, resume support at
every step, raw HTML saved for debugging.

## Project structure

```
main.py            app setup, HTML page routes, router registration
app/
  config.py        pydantic-settings tunables + paths (env-overridable, LEX_ prefix)
  schemas.py       Pydantic request models
  limits.py        shared slowapi limiter
  vectorstore.py   ChromaDB client, cosine similarity helper, metadata hygiene
  routers/         one APIRouter per tab (chat, lawyer, simulation, admin)
  scraper/         sud.mk scraper + CLI
  ingest/          PDF → text → outcomes (regex) → SQLite → chunks → vectors; summaries
  rag/             two-stage retriever, probability, answer chain + self-check, semantic cache
  agents/          courtroom simulation (4 roles, 6 scripted turns, streaming)
  lawyer/          laws: download, article-level index, law-first answers
  admin/           anonymization (regex rules + LLM pass)
ideas/             documented experiments NOT wired into the app (e.g. bge-reranker pipeline)
other/             previous implementations & unused files — the "roads not taken" archive
static/ templates/ frontend (vanilla JS, one small shared common.js)
```

Two conventions worth knowing when reading the code:

- **Replaced approaches stay visible**: superseded code is either kept as a labeled
  commented block («ПРЕТХОДЕН ПРИСТАП») next to its replacement, or preserved as a whole
  file under `other/` — the project's history of alternatives is part of the learning.
- **Honesty over impressiveness**: the probability is framed as statistics over past cases,
  the model self-checks against its sources, and low similarity produces «Не знам» instead
  of a hallucination.

## Author

**Bojan Sevdinov** — LexMacedonica © 2026

[sud.mk]: http://www.sud.mk/wps/portal/central/sud/odluki
