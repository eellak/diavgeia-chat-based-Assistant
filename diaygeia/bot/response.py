import logging
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, ConnectionError

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from diaygeia.domain import Conversation
from diaygeia.session.manage_session import SessionManager
from diaygeia.aggregation import structured_query as sq

load_dotenv()

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GCP_PROJECT:
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    gemini_model = GenerativeModel(GEMINI_MODEL)
else:
    gemini_model = None


def _gemini_generate(prompt):
    """Deterministic plain-text generation via Gemini — used for structured-query
    intent detection (temperature 0 for stable JSON specs)."""
    if gemini_model is None:
        raise RuntimeError("Gemini not configured")
    resp = gemini_model.generate_content(
        prompt, generation_config=GenerationConfig(temperature=0))
    return (resp.text or "").strip()


NUM_RESULTS = 3

# --- Optional cross-encoder reranking (Tier 1) ---
# OFF by default. When USE_RERANK=1, get_context_multi over-fetches a Tier-0 pool
# and re-orders it with a cross-encoder. This needs a Greek-capable model
# (bge-reranker-v2-m3) plus real RAM/GPU and the `sentence-transformers` package —
# kept out of requirements.txt to keep the base image lean — so it is meant for a
# resourced host, not the default laptop container. If the model cannot load,
# retrieval degrades gracefully to plain Tier-0 order.
RERANK_ENABLED = os.getenv("USE_RERANK", "0").lower() not in ("0", "false", "no", "")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))    # Tier-0 pool for rerank
RERANK_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CHAR_LIMIT = int(os.getenv("RERANK_CHAR_LIMIT", "2000"))  # chars fed to reranker

_reranker = None


def _get_reranker():
    """Lazily load the cross-encoder so importing this module never fails when the
    reranker is unused (or sentence-transformers isn't installed)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


def get_user_history(user_history_data: Conversation, limit=5):
    history = []
    if len(user_history_data) > 0:
        for i, u in user_history_data.tail(limit).iterrows():
            role = "user" if u['user'] != 'BOT' else "assistant"
            history.append({"role": role, "content": u['utterance']})
    return history


class DiaygeiaBot:
    def __init__(self, name, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.session_manager = SessionManager(logger=self.logger)
        self.index_name = name

        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        try:
            self.client = Elasticsearch(
                hosts=[{"host": es_host, "port": 9200}],
                timeout=30,
                max_retries=10,
                retry_on_timeout=True,
            )
            if not self.client.ping():
                if es_host == "localhost":
                    self.logger.warning("Failed to connect to localhost:9200, trying elasticsearch:9200...")
                    self.client = Elasticsearch(
                        hosts=[{"host": "elasticsearch", "port": 9200}],
                        timeout=30,
                        max_retries=10,
                        retry_on_timeout=True,
                    )
                    if not self.client.ping():
                        raise ValueError("Connection to Elasticsearch failed")
                else:
                    raise ValueError("Connection to Elasticsearch failed")
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def get_context(self, question: str, k: int = NUM_RESULTS):
        search_query = {"query": {"match": {"content": question}}}
        resp = self.client.search(index=self.index_name, body=search_query, size=k)
        return [{"id": hit['_id'], "content": hit['_source']['content']} for hit in resp['hits']['hits']]

    def get_context_multi(self, question: str, k: int = NUM_RESULTS,
                          rerank: bool = RERANK_ENABLED,
                          candidates: int = RERANK_CANDIDATES):
        """Tier 0 retrieval: multi-field BM25 over content + subject^2.

        The subject/title is very high-signal for Diavgeia decisions, so boosting it
        markedly improves retrieval over the single-field `get_context` (kept above as
        the baseline). Returns the same shape as `get_context`.

        Tier 1 (optional, `rerank=True` / USE_RERANK=1): over-fetch `candidates` and
        re-order them with a cross-encoder, keeping the true top-k. Opt-in and heavy;
        if the reranker cannot load, it degrades gracefully to Tier-0 order.
        """
        size = candidates if rerank else k
        search_query = {"query": {"multi_match": {"query": question,
                                                  "fields": ["content", "subject^2"]}}}
        resp = self.client.search(index=self.index_name, body=search_query, size=size)
        hits = resp['hits']['hits']
        if not hits:
            return []

        docs = [{"id": h['_id'], "content": h['_source'].get('content', '')} for h in hits]
        if rerank:
            try:
                reranker = _get_reranker()
                pairs = []
                for h in hits:
                    subject = h['_source'].get('subject') or ''
                    body = h['_source'].get('content') or ''
                    pairs.append((question, (subject + "\n" + body)[:RERANK_CHAR_LIMIT]))
                scores = reranker.predict(pairs)
                order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
                docs = [docs[i] for i in order]
            except Exception as e:  # noqa: BLE001 — never let reranking break retrieval
                self.logger.warning(f"Reranker unavailable ({e}); using Tier-0 order")

        return docs[:k]

    def get_llm_response(self, question, history, context):
        self.logger.warning(f"History: {history}")

        if gemini_model is None:
            raise RuntimeError("Gemini not configured. Set GOOGLE_CLOUD_PROJECT and credentials.")

        system_prompt = (
            "You are a helpful assistant for answering questions about Diavgeia Greek government documents. "
            "Given a question, provide the answer along with the ADA number of the document in this form ΑΔΑ:XXXXXXXXXXXXXXX."
        )
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history]) if history else ""
        prompt = f"""
        {system_prompt}

        Conversation history:
        {history_text}

        Retrieved context:
        {context}

        User question:
        {question}
        """.strip()

        response = gemini_model.generate_content(prompt)
        return (response.text or "").strip()

    def try_structured(self, question):
        """Return an exact structured answer for a countable question, else None.

        Detects a countable/aggregation question (LLM), runs an Elasticsearch
        aggregation over the metadata, and formats a grounded Greek answer with
        names resolved. None means "not a structured question" → caller uses RAG.
        Any failure returns None (safe fallback).
        """
        try:
            spec = sq.build_spec(question, _gemini_generate)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"structured intent detection failed: {e}")
            return None
        if not spec:
            return None
        try:
            return sq.answer(self.client, self.index_name, spec)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"structured query failed ({e}); using RAG")
            return None

    def get_bot_response(self, question, session_id):
        user_history_data = self.session_manager.get_user_session(session_id)
        history = get_user_history(user_history_data)

        # Structured path first: countable questions get an exact ES aggregation
        # (names resolved) instead of the LLM guessing numbers; None -> RAG below.
        structured = self.try_structured(question)
        if structured is not None:
            self.session_manager.update_user_session(
                session_id, user_history_data, question, structured)
            return structured

        # Tier 0 retrieval (multi-field + subject boost). Search on the question alone —
        # prepending history pollutes BM25; history is still used for generation below.
        context_results = self.get_context_multi(question)
        context = "\n\n".join([str(item) for sublist in context_results for item in sublist.values()])

        self.logger.warning(f"History: {history}")
        completion = None
        try:
            completion = self.get_llm_response(question, history, context)
        except Exception as e:
            self.logger.error(f"LLM error: {e}")

        self.session_manager.update_user_session(
            session_id, user_history_data, question, completion
        )
        return completion
