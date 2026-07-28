"""
Exercise the structured-query engine live against the index.

Each case is a natural question + the spec it should map to. This validates the
deterministic core (ES aggregation + ID→name resolution + Greek formatting)
independently of the LLM intent detection (`build_spec`, tested with the bot).

    python -m diaygeia.aggregation.test_structured
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elasticsearch import Elasticsearch

from diaygeia.aggregation import structured_query as sq

CASES = [
    ("Πόσες αποφάσεις υπάρχουν συνολικά;",
     {"op": "count"}),
    ("Πόσες αποφάσεις είναι πίνακες κατάταξης (τύπος Γ.3.2);",
     {"op": "count", "filters": {"decision_type_id": "Γ.3.2"}}),
    ("Πόσες αποφάσεις εξέδωσε το ΕΚΕΤΑ;",
     {"op": "count", "filters": {"organization_id": "99220974"}}),
    ("Πόσες αποφάσεις εκδόθηκαν τον Σεπτέμβριο 2025;",
     {"op": "count", "filters": {"date_from": "2025-09-01", "date_to": "2025-09-30"}}),
    ("Ποια είναι η κατανομή των αποφάσεων ανά τύπο;",
     {"op": "group_by", "group_field": "decision_type_id"}),
    ("Ποιοι φορείς εξέδωσαν τις περισσότερες αποφάσεις; (top 5)",
     {"op": "group_by", "group_field": "organization_id", "top": 5}),
]


def main():
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    if not es.ping():
        raise SystemExit("Cannot reach Elasticsearch at localhost:9200 — is the stack up?")
    index = os.getenv("INDEX", "diaygeia")

    for question, spec in CASES:
        print("\n" + "=" * 92)
        print("Q:", question)
        print("  spec:", spec)
        print("-" * 92)
        print(sq.answer(es, index, spec))


if __name__ == "__main__":
    main()
