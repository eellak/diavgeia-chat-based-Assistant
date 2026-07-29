# Diavgeia Chat-Based Assistant

A Greek-language RAG assistant for **Διαύγεια** (Diavgeia) government documents.
Ask natural-language questions; the assistant retrieves relevant decisions from
Elasticsearch and answers with **Google Gemini (Vertex AI)**, citing the ΑΔΑ
number of each document it used.

This repo is the cleaned starting point for an **EELLAK-supported** extension of
the original prototype. The underlying corpus, extraction pipeline, boilerplate
analysis, and RAG benchmark are described in our paper — see [Research](#research).

## Demo

**🔗 Live demo:** http://35.224.220.36:8501 — *DiavgeiaAssistant*, a hosted
instance answering natural-language questions over ~32K sampled Diavgeia
decisions (BM25 retrieval + **Gemini 2.5 Flash** via Vertex AI), with multi-turn
chat and ΑΔΑ citations.

![Demo: asking the assistant two questions about Diavgeia decisions](docs/demo.gif)

*The recording above shows the previous interface.*

> **Note:** the interface has been redesigned — a new chat layout with
> right-aligned user messages and a navigation sidebar. The **live demo above
> already runs this updated UI**; the refreshed Streamlit code and an updated demo
> recording will be added to this repo shortly. The live demo is a community
> deployment and may not always be online.

---

## Repo layout

```
.
├── streamlit_ui/
│   ├── streamlit_app_v2.py       Streamlit chat UI — current interface (entry point)
│   ├── streamlit_app_demo.py     Earlier UI (kept for reference)
│   └── assets/                   Bot/user avatars
│
├── diaygeia/                     Assistant Python package
│   ├── bot/response.py           DiaygeiaBot — retrieval + Gemini generation
│   ├── domain.py                 Conversation dataclass / DataFrame wrapper
│   ├── session/manage_session.py Redis-backed per-user chat history
│   ├── data_io/redis/engine.py   Redis connection factory
│   ├── txt_to_index.py           Bulk-index local .txt files into Elasticsearch
│   ├── index_diavgeia_dataset.py Stream the public glossAPI/diavgeia dataset into ES
│   ├── aggregation/              Structured metadata queries + Diavgeia OpenData name lookups
│   └── grounding.py              Source links, ΑΔΑ citations, insufficient-evidence reply
│
├── Diaygeia.ipynb                Data-processing notebook
├── Dockerfile                    Image used by docker-compose
├── docker-compose.yml            redis + elasticsearch + kibana + streamlit
├── requirements.txt              Python deps
└── .gitignore
```

A `secrets/gcp-sa.json` file (gitignored) must hold your GCP service-account
key — `docker-compose.yml` mounts it read-only into the container.

---

## How it works

1. User asks a question in the Streamlit UI.
2. **Structured path first** (`DiaygeiaBot.try_structured`): if the question is
   *countable* ("how many / by type / by org / in period Z") it is answered by an
   exact Elasticsearch **aggregation** over the metadata — not by the LLM — and
   returned immediately (see [Structured queries](#structured-queries-counts--aggregations)).
   Otherwise it falls through to retrieval:
3. `DiaygeiaBot.get_context_multi` runs a multi-field BM25 query — over the decision
   text and its **boosted `subject`** (the high-signal title) — against the
   `diaygeia` index and pulls the top-k matching decisions. (The single-field
   `get_context` is kept as a baseline.)
4. Recent conversation history is fetched from Redis (compressed JSON keyed by
   `session_id`).
5. The retrieved context, history, and user question are sent to **Gemini
   (Vertex AI)**, instructed to answer **only from that context** (or say it doesn't
   have enough information) and to cite the ΑΔΑ of each decision it used — those ΑΔΑs
   become clickable links to diavgeia.gov.gr (see [Grounding & citations](#grounding--citations)).
6. The new turn is appended to Redis history.

---

## Structured queries (counts & aggregations)

RAG is great at *"what does a decision say"* but weak at *"how many"* — the LLM only
sees the top-k snippets and can't count across the corpus. So **countable questions
are routed to Elasticsearch instead of the LLM**:

- `diaygeia/aggregation/structured_query.py` — `build_spec` (Gemini, temperature 0)
  turns a Greek question into a query spec, run as an ES **`count`** or **`terms`
  (group-by)** with filters on decision type, organisation, thematic category and
  **issue-date range**. Non-countable questions return `None` → RAG.
- `diaygeia/aggregation/diavgeia_lookup.py` — resolves the opaque metadata IDs to
  human names via the public **Diavgeia OpenData API** (`/opendata/types`,
  `/opendata/organizations/{id}`), cached on disk, so answers read *"98 αποφάσεις του
  ΕΚΕΤΑ"* and a user can filter by an organisation's name.

Examples that now get exact, grounded answers:

```
πόσες αποφάσεις εξέδωσε το ΕΚΕΤΑ;             → 98
ποια είναι η κατανομή των αποφάσεων ανά τύπο;  → Γ.3.2: 1.544 · Ε.4: 456
πόσες αποφάσεις τον Σεπτέμβριο 2025;           → 432
```

**Name resolution needs a warm cache.** After indexing, resolve the organisation ids
present in your index once (idempotent; cached to a gitignored `.diavgeia_cache/`):

```python
from elasticsearch import Elasticsearch
from diaygeia.aggregation import diavgeia_lookup as lk
lk.warm_org_cache(Elasticsearch("http://localhost:9200"), "diaygeia")
```

Only counting/filtering is supported — the dataset metadata has no monetary amounts,
so "total spent" sums are out of scope.

---

## Grounding & citations

To keep answers trustworthy, the RAG path (`diaygeia/grounding.py`) adds three things:

- **Clickable sources** — every ΑΔΑ the model cites becomes a link to the official
  decision (`https://diavgeia.gov.gr/doc/{ada}`), but **only for ΑΔΑs that were
  actually retrieved** — a hallucinated id is never linked.
- **"I don't know" instead of guessing** — the generation prompt tells Gemini to
  answer *only* from the retrieved context and otherwise say it lacks the
  information; if retrieval returns nothing, a fixed insufficient-evidence reply is
  used and the LLM isn't called at all.
- **Evidence panel** — toggle "Εμφάνιση context (πηγών)" in the sidebar to see the
  exact decisions behind each answer (ΑΔΑ links + snippets).

---

## Evaluation

The upgrades were measured **head-to-head against the original prototype on the same
index** — only the *system* changes (retrieval field, generation prompt, structured
routing), never the data or the model (`gemini-2.5-flash` for both), so any difference is
attributable to the upgrades. Ground truth for the count questions comes straight from
Elasticsearch; refusals are checked against a fixed off-topic set; the retrieval set is 50
auto-generated natural-language questions, each paraphrasing a decision's title with that
decision as the gold answer.

On a **50,000-document** index:

| Metric | Old (prototype) | New (upgraded) |
|---|---|---|
| Retrieval recall@5 | 46% | **66%** |
| Retrieval MRR | 0.37 | **0.59** |
| Quantitative exact-match | 0% | **92%** |
| Refusal on off-topic questions | 0% | **88%** |

- **Quantitative** is the headline: the prototype cannot count across the corpus (0 of 13
  count questions correct); the structured path answers exactly (12 of 13) — the concrete
  fix for the quantitative weakness.
- **Retrieval**: boosting the high-signal title (`subject`) field recovers about one in
  five more correct documents in the top-5, and ranks them higher.
- **Grounding**: the prototype answered every off-topic question from its own knowledge
  (0 refusals); the new system declines 7 of 8.

*(The evaluation harness itself is kept out of the repo — this section reports its
results.)*

---

## Optional: cross-encoder reranking (Tier 1)

Retrieval is multi-field BM25 by default. For higher precision on vague or
paraphrased questions, you can optionally re-rank the top BM25 candidates with a
multilingual **cross-encoder**:

```bash
export USE_RERANK=1                             # off by default
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3   # Greek-capable (default)
pip install sentence-transformers               # extra dep, kept out of the base image
```

`get_context_multi` then over-fetches a pool of BM25 hits and re-orders them with
the cross-encoder, keeping the true top-k. It is **heavy** (a large model plus
RAM/GPU) and intended for a resourced host — **not** the default laptop container;
if the model cannot load, retrieval falls back to plain BM25 order. In our testing
`bge-reranker-v2-m3` handled Greek well, while smaller/base rerankers hurt results.

---

## Running

### 1. Drop your GCP service-account JSON into `secrets/`

```bash
mkdir -p secrets
cp /path/to/your-service-account.json secrets/gcp-sa.json
```

`docker-compose.yml` mounts this at `/app/secrets/gcp-sa.json` inside the
container. The GCP project ID and Gemini model are already set in the compose
file — edit them there if you need different values.

### 2. Build and start the stack

```bash
docker compose up --build
```

This brings up Redis, Elasticsearch, Kibana, and the Streamlit UI.
UI → http://localhost:8501.

### 3. Populate Elasticsearch

The bot has nothing to retrieve until the `diaygeia` index is filled. Two ways:

**Option A — load the public Diavgeia dataset (recommended).**
Stream documents directly from the open
[`glossAPI/diavgeia`](https://huggingface.co/datasets/glossAPI/diavgeia) dataset
(CC BY 4.0) — no local files needed. The dataset is *gated* (auto-approval): open
the page once and accept the terms, then authenticate with a Hugging Face token.
With the stack running (so Elasticsearch is reachable on `localhost:9200`):

```bash
pip install datasets                 # already in requirements.txt / the image — needed only when running on the host
huggingface-cli login                # or: export HF_TOKEN=hf_xxx

# stream + bulk-index the first N documents (start small, grow later)
python diaygeia/index_diavgeia_dataset.py --limit 2000 --recreate
python diaygeia/index_diavgeia_dataset.py --limit 50000
```

The loader indexes each decision's Markdown text (with a Greek analyzer) plus
structured metadata — ΑΔΑ, organisation, decision type, dates, thematic
categories. Use `--es <url>` (or `ELASTICSEARCH_URL`) to target a different
Elasticsearch, and `--help` for all options.

**Option B — index your own `.txt` files.**
Mount a folder of `.txt` decisions into the running container and run the
original indexer (adjust the hardcoded `path` at the top of the script first):

```bash
docker compose exec streamlit python diaygeia/txt_to_index.py
```

---

## Data-processing notebook

`Diaygeia.ipynb` contains exploration and preprocessing for Diavgeia data.
Open it locally with Jupyter, or run a one-off Jupyter container against the
project deps.

---

## Research

This assistant accompanies our dataset-and-RAG paper, which introduces a
**1-million-document** open corpus of Greek government decisions (normalized
metadata + Markdown text), a reproducible extraction pipeline with a comparative
OCR/VLM benchmark, a Greek **boilerplate-extraction** method, and the
evidence-grounded **RAG task** this assistant implements.

> G. Antoniou, G. Filandrianos, A. Vlachos, G. Stamou, L. Kollimenos,
> K. Skianis, M. Vazirgiannis.
> *A Greek Government Decisions Dataset for Public-Sector Analysis and Insight.*
> arXiv:2512.05647, 2025. — https://arxiv.org/abs/2512.05647

> **Note:** an extended version of the paper (with the full extraction benchmark
> and an online-demo appendix) is currently under review; this section will be
> updated when it becomes public.

```bibtex
@article{antoniou2025diavgeia,
  title   = {A Greek Government Decisions Dataset for Public-Sector Analysis and Insight},
  author  = {Antoniou, Giorgos and Filandrianos, Giorgos and Vlachos, Aggelos and Stamou, Giorgos and Kollimenos, Lampros and Skianis, Konstantinos and Vazirgiannis, Michalis},
  journal = {arXiv preprint arXiv:2512.05647},
  year    = {2025}
}
```

---

## Acknowledgements

- [Διαύγεια / Diavgeia](https://diavgeia.gov.gr) — built and operated by **OTS (Open Technology Services)**
- [Google Gemini](https://ai.google.dev/) · [Vertex AI](https://cloud.google.com/vertex-ai)
- [Elasticsearch](https://www.elastic.co) · [Streamlit](https://streamlit.io)
- [EELLAK](https://eellak.gr) 
