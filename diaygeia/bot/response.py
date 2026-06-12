import logging
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, ConnectionError

import vertexai
from vertexai.generative_models import GenerativeModel

from diaygeia.domain import Conversation
from diaygeia.session.manage_session import SessionManager

load_dotenv()

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GCP_PROJECT:
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    gemini_model = GenerativeModel(GEMINI_MODEL)
else:
    gemini_model = None

NUM_RESULTS = 3


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

    def get_bot_response(self, question, session_id):
        user_history_data = self.session_manager.get_user_session(session_id)
        history = get_user_history(user_history_data)
        str_history = "\n".join([item['content'] for item in history])
        context_results = self.get_context(str_history + " " + question)
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
