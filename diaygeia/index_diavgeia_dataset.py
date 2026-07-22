"""
Index documents from the public Diavgeia dataset into Elasticsearch.

Source dataset: https://huggingface.co/datasets/glossAPI/diavgeia  (CC BY 4.0)
The dataset is *gated* (auto-approval): open the page once, accept the terms,
then authenticate with a Hugging Face token before running this script:

    huggingface-cli login              # or:  export HF_TOKEN=hf_xxx

The dataset is streamed, so only the first `--limit` documents are downloaded —
you can start with a few thousand and grow the index later.

Each row provides:  id (ADA) · markdown_text (content) · metadata_json (fields).

Usage:
    python diaygeia/index_diavgeia_dataset.py --limit 50000
    python diaygeia/index_diavgeia_dataset.py --limit 200000 --recreate
"""

import argparse
import json
import os
import sys

from elasticsearch import Elasticsearch, helpers

DATASET = "glossAPI/diavgeia"
INDEX_NAME = "diaygeia"

# Greek analyzer on the free-text fields (built-in); structured fields for
# future metadata-aware filtering. Kept backward-compatible: retrieval still
# reads the `content` field exactly as before.
INDEX_CONFIG = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "greek"},
            "subject": {"type": "text", "analyzer": "greek"},
            "ada": {"type": "keyword"},
            "organization_id": {"type": "keyword"},
            "organization_name": {"type": "keyword"},
            "decision_type_id": {"type": "keyword"},
            "signer_ids": {"type": "keyword"},
            "unit_ids": {"type": "keyword"},
            "thematic_category_ids": {"type": "keyword"},
            "issue_date": {
                "type": "date",
                "format": "epoch_millis||strict_date_optional_time",
            },
        }
    },
}


def build_doc(row):
    """Map one dataset row to an Elasticsearch document.

    Defensive: a missing or malformed metadata field never drops the document —
    the `content` field (what retrieval uses) is always populated.
    """
    ada = row.get("id")
    content = row.get("markdown_text") or ""

    meta = {}
    raw = row.get("metadata_json")
    if raw:
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            meta = {}

    org_name = None
    extra = meta.get("extraFieldValues")
    if isinstance(extra, dict):
        org = extra.get("org")
        if isinstance(org, dict):
            org_name = org.get("name")

    # `content` is always kept (even if empty) so the schema stays consistent;
    # optional metadata fields are pruned when empty.
    doc = {"content": content}
    if ada is not None:
        doc["ada"] = ada
    optional = {
        "subject": meta.get("subject"),
        "organization_id": meta.get("organizationId"),
        "organization_name": org_name,
        "decision_type_id": meta.get("decisionTypeId"),
        "signer_ids": meta.get("signerIds"),
        "unit_ids": meta.get("unitIds"),
        "thematic_category_ids": meta.get("thematicCategoryIds"),
        "issue_date": meta.get("issueDate"),
    }
    doc.update({k: v for k, v in optional.items() if v not in (None, "", [], {})})
    return ada, doc


def actions(dataset, index_name, limit):
    """Yield bulk-index actions for up to `limit` rows, logging progress."""
    for i, row in enumerate(dataset):
        if i >= limit:
            break
        ada, doc = build_doc(row)
        yield {"_index": index_name, "_id": ada, "_source": doc}
        if (i + 1) % 5000 == 0:
            print(f"  prepared {i + 1:,} documents...", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=50000,
                        help="number of documents to index (default: 50000)")
    parser.add_argument("--index", default=INDEX_NAME, help="Elasticsearch index name")
    parser.add_argument("--es", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
                        help="Elasticsearch URL")
    parser.add_argument("--dataset", default=DATASET, help="Hugging Face dataset id")
    parser.add_argument("--split", default="train", help="dataset split")
    parser.add_argument("--batch-size", type=int, default=500, help="bulk batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="delete and recreate the index before indexing")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: pip install datasets  (see requirements.txt)")

    es = Elasticsearch(args.es)
    if not es.ping():
        sys.exit(f"Cannot reach Elasticsearch at {args.es} — is the stack up?")

    if args.recreate and es.indices.exists(index=args.index):
        print(f"Deleting existing index '{args.index}'...")
        es.indices.delete(index=args.index)
    if not es.indices.exists(index=args.index):
        print(f"Creating index '{args.index}' (Greek analyzer + metadata fields)...")
        es.indices.create(index=args.index, body=INDEX_CONFIG)

    print(f"Streaming '{args.dataset}' [{args.split}] — indexing up to {args.limit:,} docs...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True,
                               token=os.getenv("HF_TOKEN"))
    except Exception as e:  # noqa: BLE001 — surface the gating hint clearly
        sys.exit(f"Failed to load dataset: {e}\n"
                 "If this is an access error: open the dataset page, accept the "
                 "terms, then `huggingface-cli login` or set HF_TOKEN.")

    indexed, errors = helpers.bulk(
        es, actions(dataset, args.index, args.limit),
        chunk_size=args.batch_size, raise_on_error=False,
    )
    es.indices.refresh(index=args.index)
    total = es.count(index=args.index).get("count")
    print(f"\nDone. Indexed {indexed:,} documents"
          + (f" ({len(errors)} errors)" if errors else "")
          + f". Index '{args.index}' now holds {total:,} documents.")


if __name__ == "__main__":
    main()
