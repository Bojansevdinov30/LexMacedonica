# Learning Roadmap — understand every piece of LexMacedonica

Rule #1: **every topic below lives in a file of this project.** Don't study
abstractly — open the file, read it, break it, fix it. The fastest way to learn
is: change something → predict what happens → run it → be surprised → understand.

Rough effort: ~1–1.5 h/day for 6 weeks gets you through everything before the
presentation. Levels build on each other; don't skip Level 3 prep.

---

## Level 0 — How the web app talks (week 1)

**1. HTTP: request → response.**
Open DevTools (F12) → Network tab → ask a question in the chat. Watch the
`POST /api/chat` request: headers, JSON body, JSON response. That request/response
pair IS the whole frontend↔backend relationship.
- *In this repo:* `static/js/chat.js` (the `fetch(...)` call) ↔ `main.py` (`@app.post("/api/chat")`).
- *Learn:* MDN "HTTP Overview". Know: GET vs POST, status codes (200/404/500), headers, JSON.

**2. HTML / CSS / JS just enough.**
You don't need to be a frontend dev. Understand: the DOM is a tree, JS finds
nodes (`getElementById`) and changes them, CSS selects and styles them.
- *In this repo:* read `templates/index.html` + `static/js/chat.js` top to bottom;
  in `static/css/style.css` find `.prob-bar-fill` and make the bar green.
- *Exercise:* add a "Исчисти разговор" button that empties the chat window and the `history` array.

**3. Jinja2 templates.**
Server-side HTML assembly: `base.html` holds the nav/footer, each page fills `{% block content %}`.
- *Exercise:* add a fifth nav tab "За нас" with a static page.

## Level 1 — The Python backend (week 1–2)

**4. venv + pip + requirements.txt.** Why every project gets its own bubble of
packages. Delete `.venv`, recreate it from `requirements.txt` — now you know it's reproducible.

**5. FastAPI.**
Routes, Pydantic models (`ChatRequest` — automatic JSON validation!), why our
endpoints are `def` not `async def` (they do blocking LLM calls; FastAPI moves
them to a thread pool). Run `uvicorn main:app --reload` and visit `/docs` —
FastAPI auto-generates interactive API documentation. Play with it.
- *Learn:* the official FastAPI tutorial, first ~6 chapters. It's excellent.

**6. SQLite + SQLAlchemy.**
One file = a real database. ORM = Python classes ↔ table rows.
- *In this repo:* `app/ingest/structure.py` (the `Case` model), then open
  `data/lex.db` with "DB Browser for SQLite" and look at your 370 cases.
- *Exercise:* write a 5-line script that prints outcome counts per court.

**7. Regex.**
The outcome extractor is regex-first because court decisions are formulaic.
- *In this repo:* `OUTCOME_PATTERNS` in `structure.py`, the ЕМБГ/phone rules in
  `app/admin/anonymize.py`. Paste them into regex101.com and poke at them.

## Level 2 — The data pipeline (week 2)

**8. Scraping a session-based site.**
The best war story in this project. Read the docstring at the top of
`app/scraper/sudmk.py` — it documents the three traps we hit for real:
`<base href>` resolution, sticky server sessions, dead search filters.
Concepts: cookies, sessions, form POSTs, why `requests.Session()` ≠ `requests.get()`.
- *Learn:* MDN "HTTP cookies", then reread the scraper — it will suddenly make sense.

**9. BeautifulSoup.** HTML → Python objects. See `_parse_results()` — find the
result boxes, pull out values. *Exercise:* print the preview text of 5 results
from a saved page in `data/raw_html/`.

**10. PDF extraction.** PDFs are containers of drawing commands, not text files;
PyMuPDF walks them and returns text. Digital PDFs (ours) vs scanned ones (need OCR — we dodged that).

## Level 3 — The AI core (weeks 3–4, the most important level)

**11. What an LLM actually is.**
Next-token prediction, tokens (Cyrillic is token-expensive — that's why our
chunks are counted in tokens!), context windows, temperature, cost per token.
- *Learn:* 3Blue1Brown "But what is a GPT?" (visual, superb), then Karpathy's
  "Intro to Large Language Models" talk.
- *In this repo:* `app/costs.py` — you literally track tokens→dollars.

**12. Embeddings — the single most important concept in this project.**
Text → vector of 1536 numbers where *similar meaning = nearby vectors*.
Cosine similarity measures that closeness. Everything else (search, cache,
the "Не знам" gate) is built on this.
- *Learn:* Jay Alammar's illustrated guides; Simon Willison's "Embeddings" explainer.
- *Exercise (do this one!):* embed «отказ», «ме отпуштија од работа» and
  «рецепт за ајвар» with the OpenAI API, compute cosine similarities with numpy
  (3 lines), and see the numbers agree with intuition.

**13. Vector search + FAISS.**
Brute-force cosine over 6K vectors is instant (`IndexFlatIP` — what we use);
HNSW is the approximate graph trick for millions of vectors (what we planned
until it crashed — read the comment in `build_index.py:build_faiss`). Know why
normalizing vectors turns inner product into cosine similarity.

**14. BM25 + hybrid search + RRF.**
BM25 = word-frequency ranking (exact terms: «член 101»); vectors = meaning
(paraphrases). RRF fuses the two rankings using only positions: Σ 1/(60+rank).
- *In this repo:* `app/rag/retriever.py` — read it slowly, it's the heart.
- *Exercise:* in `eval_questions.py` output, find a question where BM25 and
  vectors disagree; check which chunks made the final top 3.

**15. Chunking.** Why documents are split (~800 tokens) with overlap (15%):
one idea per vector, no sentence cut in half. `app/ingest/chunking.py`.

**16. RAG end-to-end.** Now assemble the picture: retrieve → build prompt with
context → generate → verify. Read `app/rag/chains.py` top to bottom and match
every step to the pipeline comment at the top of the file.

**17. Two-stage retrieval (your own idea!).** Why chunk-level search matches
words, not situations, and how per-case summaries fix it. `app/ingest/summarize.py`
+ stage 1 in `retriever.py`. This pattern is also called hierarchical retrieval.

**18. Prompt engineering patterns in this repo:**
- system prompts as role contracts (`ANSWER_PROMPT`, the 4 roles in `agents/simulation.py`)
- self-check pass (`SELF_CHECK_PROMPT`) — model verifies model
- query condensation for follow-ups (`CONDENSE_PROMPT`)
- forcing JSON output (`response_format` in `lawyer/rag.py`, `admin/anonymize.py`)
- honest refusal: the similarity threshold in `config.py` — *we measured* real
  questions (0.49–0.58) vs off-topic (0.25–0.36) and put the gate at 0.42.
  Grounding beats trusting the model.

**19. Semantic cache.** Exact-match caches miss paraphrases; embedding-similarity
caches don't. `app/rag/cache.py` — 40 lines, fully understandable.

**20. Multi-agent = prompts + a loop.** `agents/simulation.py` demystifies the
buzzword: four system prompts, a fixed turn script, shared transcript. That's it.

## Level 4 — Engineering craft (weeks 5–6)

**21. Git.** Run `git log --oneline` — the project's history is a story you
lived. For each commit, can you explain what changed and why? Learn: commit,
diff, branch, merge; try making a branch and merging it.

**22. Docker.** Image vs container, layers & caching, volumes, env vars at
runtime vs baked in. Our `Dockerfile` and `docker-compose.yml` are commented as
a lesson — read them, install Docker Desktop, run `docker compose up --build`.
- *Learn:* Docker's official "Get started" (parts 1–5).

**23. The war stories (interview gold).** This project hit four real production
problems — be able to tell each as: symptom → diagnosis → fix:
- Antivirus TLS interception broke Python HTTPS → `truststore` (in `app/config.py`)
- chromadb Rust core crashed on Python 3.14/Windows → switched to FAISS
- FAISS HNSW crashed (OpenMP) → exact flat index, better at our scale anyway
- WebSphere portal scraping → sessions, `<base href>`, sticky state

## Test yourself (before the presentation)

Explain out loud, without notes, in 2–3 sentences each:
1. What happens, step by step, between pressing «Прашај» and seeing the answer?
2. Why hybrid search instead of just vectors?
3. Where does the probability % come from? (The one answer you MUST nail —
   it's a weighted statistic over real outcomes, not model output.)
4. Why does the app say «Не знам» for a dog bite, even though 3 documents contain «куче»?
5. Why is the same repeated question instant and free the second time?
6. What would break if we deleted the overlap in chunking?
7. Why don't we store chat history on the server?

## Suggested calendar (presentation end of August)

- **Week 1 (Jul 7–13):** Level 0 + start FastAPI
- **Week 2 (Jul 14–20):** finish Level 1 + Level 2
- **Week 3 (Jul 21–27):** LLMs + embeddings + the numpy exercise
- **Week 4 (Jul 28–Aug 3):** FAISS, BM25/RRF, chunking, chains.py deep-read
- **Week 5 (Aug 4–10):** two-stage retrieval, prompts, cache, agents + Git
- **Week 6 (Aug 11–17):** Docker + war stories + self-test questions
- **Week 7+ (Aug 18–):** rehearse DEMO.md; fix whatever the rehearsal exposes
