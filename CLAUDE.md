# LexMacedonica

RAG web app for Macedonian law, built as a **student learning project**. Styled after factum.mk.
Hard deadline: **presentation at the end of August 2026**. Everything user-facing is in **Macedonian**.

## The four interfaces (top nav bar)

1. **Почетна / Правен асистент** (PRIORITY — the presentation centerpiece): citizen describes a situation →
   answer with the most likely outcome + a **probability %** computed from outcomes of the most similar past
   cases, citing the top 3 similar cases (case number, court, date, one-line summary). No chain-of-thought shown.
2. **Симулација**: 4 LLM agents — судија, адвокат на тужителот, адвокат на тужениот, наратор — play out the
   user's scenario in turns (capped ~8–10 turns). No chain-of-thought shown.
3. **Адвокати**: specialized assistant for lawyers, retrieval **law-first** (laws chunked by член so citations
   are exact), cases secondary. **Reasoning/chain-of-thought IS displayed** (collapsible panel).
4. **Администрација**: admin tasks, first tool = anonymization (ЕМБГ, names, addresses; regex + LLM assist)
   with diff-style highlights. **Reasoning IS displayed.**

## Stack

- Backend: **FastAPI** (`main.py`), Python, **LangChain** for RAG.
- Frontend: plain **HTML/CSS/JS** served by FastAPI (Jinja2 templates + `static/`). No build step.
- Vector search: **FAISS** (`IndexHNSWFlat`, cosine via normalized inner product; index at `data/faiss.index`,
  embeddings cached in `data/embeddings.npz` so re-runs don't re-pay OpenAI). NOTE: chromadb 1.x Rust core
  crashes (access violation) on Python 3.14/Windows — do not switch back without testing.
  Keyword: **BM25** (`rank_bm25`). Hybrid via RRF fusion.
- Metadata: **SQLite** (`data/lex.db`) via SQLAlchemy — court, case number, date, legal area, **outcome** per case.
- PDF: **PyMuPDF**. Embeddings: OpenAI `text-embedding-3-small`. LLM: `gpt-4o-mini` tier **everywhere** (cheapest).
- Semantic cache: numpy matrix of (query embedding → answer) at `data/semantic_cache.pkl`, cosine ≥ ~0.95 =
  cache hit. Redis only as a later Docker learning exercise.
- Logo: `static/img/logo.png` (source `logo.png` in root). Palette from logo: navy `#1e3a5f`, gold `#d4a017`,
  red accent, white background.

## Budget rule (IMPORTANT)

Total OpenAI spend until the presentation must stay **under $20** (realistic estimate: $2–5).
- Cheapest model tier only; never upgrade models without asking the user.
- Every OpenAI call goes through the cost logger (`app/costs.py`); warn loudly at $15 cumulative.
- Semantic cache stays in front of the chat chain. Batch embeddings. Top-3 chunks max as LLM context.

## Data source: sud.mk (vsrm.mk) — scraper findings

The court portal is an **IBM WebSphere Portal** (session-based, that's why naive scraping fails). Findings from
`all.devtools` (saved copy of the odluki search page — keep this file, it's the reference):
- The advanced search form POSTs to a `javax.portlet.action/searchAction` URL (session-encoded, must be parsed
  from the live page each session). Fields: `court`, `casenumber`, `dateVerifyFrom`, `dateVerifyTo`, `judgename`,
  `legalarea`, `typeofcase`, `subtypeofcase`, `irrevocable`, `guilty`, `currentPage`, `query`, …
- Each result row has an **anonymized text preview** and a download link:
  `...documentDownload=/?caseId=<GUID>&connected=false` (relative portlet URL — resolve against current page URL).
- Recipe: `requests.Session()` → GET the odluki page (cookies + fresh form action URL) → POST search → parse rows
  (BeautifulSoup) → paginate with `currentPage` → download PDFs (append `.pdf` if missing).
- Politeness: 2–3 s delay between requests, resume support, cache raw HTML in `data/raw_html/` for debugging.
- Fallbacks in order: Playwright automation → user downloads a few hundred PDFs manually into `data/raw_pdfs/`
  (the rest of the pipeline is unchanged).
- Downloaded PDFs are **text-based** (wPDF generator, embedded Cyrillic — no OCR needed) and **pre-anonymized**
  (initials like «З.С.»).
- Court decisions are formulaic: outcome extraction is **regex-first** on the dispositive
  (СЕ УСВОЈУВА / СЕ ОДБИВА / ДЕЛУМНО СЕ УСВОЈУВА / СЕ ОСУДУВА…), LLM fallback only for misses.

## Corpus decision

Focus on **one civil area** (labor disputes or debt/damages — whichever yields more documents when first
queried), target **200–500 cases**. Probability = share of outcomes among top-N similar cases, weighted by
similarity, framed honestly as statistics over past cases, **not legal advice**.

## RAG conventions

- Chunking: RecursiveCharacterTextSplitter ~800 tokens, ~15 % overlap (semantic chunking = later experiment).
- **Two-stage retrieval**: stage 1 searches per-case LLM summaries ("what is the case about",
  `app/ingest/summarize.py`, index `data/summaries.index`) → top-12 candidate cases; stage 2 = BM25 + vectors
  → RRF fusion restricted to those cases → **top-3 chunks** to the LLM. Probability + citations use the
  stage-1 (situation-level) ranking and the case summaries.
- **Conversation memory**: no server sessions — the browser keeps the chat history and sends it along;
  the backend condenses follow-ups into standalone questions (cheap LLM call) before retrieval/caching.
- Self-check pass (second cheap call) before output; if max summary similarity below threshold → honest
  **"Не знам"** answer, never hallucinate.
- All prompts and UI text in Macedonian.

## Working style (user is a student)

- **Explain everything while building** — what each concept is and why it's used (RAG, hybrid search, HNSW,
  chunking, sessions/cookies, Docker…). Keep code small, simple, and readable over clever.
- Each phase must end with the app runnable (`uvicorn main:app --reload`).

## Roadmap after August (do NOT build now)

User accounts/login → paid plans, Redis cache in production, deployment (not Vercel — stateful Python+Chroma
needs a VM or Railway/Fly.io), broader corpus across all legal areas.
