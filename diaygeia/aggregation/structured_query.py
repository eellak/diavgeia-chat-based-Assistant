"""
Structured-query path for countable questions ("how many...", "count by...").

Instead of asking the LLM to count / do arithmetic over retrieved text (the
measured weak point), a countable question is (1) detected and translated into a
*spec* (LLM, `build_spec`), then (2) run as an Elasticsearch aggregation over the
indexed metadata and answered exactly and grounded (`answer`).

A *spec*:
    {"op": "count" | "group_by",
     "group_field": "decision_type_id" | "organization_id" | "thematic_category_ids",
     "top": 10,
     "filters": {"decision_type_id": "Γ.3.2",
                 "organization_id": "99220974",
                 "thematic_category_ids": "44",
                 "date_from": "2025-09-01", "date_to": "2025-09-30"}}

Feasible on the current index (all metadata are keyword/date fields). Monetary
sums are NOT possible (no amount field in the dataset metadata).
"""
import json
from datetime import datetime, timedelta, timezone

from . import diavgeia_lookup as lookup

TERM_FILTERS = ("decision_type_id", "organization_id", "thematic_category_ids",
                "signer_ids", "unit_ids")


# --- ES execution ---------------------------------------------------------------
def _to_millis(datestr, end_of_day=False):
    dt = datetime.strptime(datestr, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _build_query(filters):
    must = []
    for f in TERM_FILTERS:
        if filters.get(f):
            must.append({"term": {f: filters[f]}})
    if filters.get("date_from") or filters.get("date_to"):
        rng = {}
        if filters.get("date_from"):
            rng["gte"] = _to_millis(filters["date_from"])
        if filters.get("date_to"):
            rng["lte"] = _to_millis(filters["date_to"], end_of_day=True)
        must.append({"range": {"issue_date": rng}})
    return {"bool": {"filter": must}} if must else {"match_all": {}}


def execute(es, index, spec):
    query = _build_query(spec.get("filters", {}))
    if spec.get("op") == "group_by":
        field = spec["group_field"]
        body = {"size": 0, "query": query,
                "aggs": {"g": {"terms": {"field": field, "size": spec.get("top", 10)}}}}
        buckets = es.search(index=index, body=body)["aggregations"]["g"]["buckets"]
        return {"op": "group_by", "group_field": field,
                "buckets": [(b["key"], b["doc_count"]) for b in buckets]}
    n = es.count(index=index, body={"query": query})["count"]
    return {"op": "count", "count": n}


# --- Greek answer formatting ----------------------------------------------------
def _num(n):
    return f"{n:,}".replace(",", ".")


def _label_for(field, key):
    if field == "decision_type_id":
        return f"{lookup.type_label(key)} ({key})"
    if field == "organization_id":
        return f"{lookup.organization_label(key)}"
    return str(key)


def _filters_phrase(filters):
    parts = []
    if filters.get("decision_type_id"):
        parts.append(f"τύπου «{lookup.type_label(filters['decision_type_id'])}»")
    if filters.get("organization_id"):
        parts.append(f"του φορέα «{lookup.organization_label(filters['organization_id'])}»")
    if filters.get("thematic_category_ids"):
        parts.append(f"θεματικής κατηγορίας {filters['thematic_category_ids']}")
    lo, hi = filters.get("date_from"), filters.get("date_to")
    if lo and hi:
        parts.append(f"από {lo} έως {hi}")
    elif lo:
        parts.append(f"από {lo}")
    elif hi:
        parts.append(f"έως {hi}")
    return (" " + ", ".join(parts)) if parts else ""


def format_greek(spec, result):
    filters = spec.get("filters", {})
    if result["op"] == "count":
        return f"Βρέθηκαν συνολικά {_num(result['count'])} αποφάσεις{_filters_phrase(filters)}."
    field = result["group_field"]
    if not result["buckets"]:
        return "Δεν βρέθηκαν αποφάσεις για το ερώτημα αυτό."
    lines = [f"• {_label_for(field, key)}: {_num(cnt)}" for key, cnt in result["buckets"]]
    unit = {"decision_type_id": "τύπο", "organization_id": "φορέα",
            "thematic_category_ids": "θεματική κατηγορία"}.get(field, field)
    return (f"Κατανομή αποφάσεων{_filters_phrase(filters)} ανά {unit}:\n" + "\n".join(lines))


def answer(es, index, spec):
    return format_greek(spec, execute(es, index, spec))


# --- Intent detection: question -> spec (LLM) -----------------------------------
_COUNT_TRIGGERS = ("ΠΟΣ", "ΠΛΗΘ", "ΑΡΙΘΜ", "ΚΑΤΑΝΟΜ", " ΑΝΑ ", "ΣΥΝΟΛΙΚ", "ΠΟΣΟΣΤ",
                   "ΠΕΡΙΣΣΟΤΕΡ", "ΛΙΓΟΤΕΡ", "ΣΥΧΝΟΤΕΡ", "HOW MANY", "COUNT",
                   "NUMBER OF", "DISTRIBUTION", "TOP ")
_GROUP_FIELD_MAP = {"decision_type": "decision_type_id",
                    "organization": "organization_id",
                    "thematic_category": "thematic_category_ids"}


def looks_countable(question):
    """Cheap gate so we only spend an LLM call on plausibly-countable questions."""
    n = lookup._norm(question)
    return any(t in n for t in _COUNT_TRIGGERS)


def _spec_prompt(question, today):
    types = lookup.decision_type_labels()
    tax = "\n".join(f"  {tid}: {lab}" for tid, lab in types.items())
    return f"""You translate a Greek question about Διαύγεια government decisions into a JSON query spec, but ONLY when it asks for a COUNT or a DISTRIBUTION/breakdown. For anything else (find / show / what does a decision say), return {{"structured": false}}.

Today is {today}.

Output STRICT JSON only (no markdown), exactly one of:
  {{"structured": false}}
  {{"structured": true, "op": "count", "filters": {{...}}}}
  {{"structured": true, "op": "group_by", "group_field": "<field>", "top": <int>, "filters": {{...}}}}

group_field is one of: "decision_type", "organization", "thematic_category".
filter keys (all optional):
  "decision_type": the closest decision-type id from the taxonomy below (use the id).
  "organization": the organisation name/abbreviation exactly as the user wrote it.
  "thematic_category": a thematic-category id, if the user gives one.
  "date_from","date_to": YYYY-MM-DD inclusive; infer from "τον Σεπτέμβριο 2025", "το 2025", "πέρσι/φέτος" relative to today.

Use op "count" for "πόσες/πόσα/πλήθος/αριθμός/how many". Use op "group_by" for distributions ("κατανομή", "ανά τύπο/φορέα", "ποιοι φορείς έβγαλαν τις περισσότερες", "top N"); set top to N or 10.

Decision-type taxonomy (id: label):
{tax}

Question: {question}
JSON:"""


def _parse_json(text):
    if not text:
        return None
    t = text.strip().lstrip("`")
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(t[i:j + 1])
    except ValueError:
        return None


def _finalize(spec):
    op = spec.get("op")
    if op not in ("count", "group_by"):
        return None
    out = {"op": op, "filters": {}}
    f = spec.get("filters") or {}
    if f.get("decision_type"):
        out["filters"]["decision_type_id"] = lookup.resolve_type(f["decision_type"]) or f["decision_type"]
    if f.get("organization"):
        oid = lookup.resolve_org(f["organization"])
        if not oid:
            return None  # named an org we can't resolve → don't risk a wrong count
        out["filters"]["organization_id"] = oid
    if f.get("thematic_category"):
        out["filters"]["thematic_category_ids"] = str(f["thematic_category"])
    for dk in ("date_from", "date_to"):
        if f.get(dk):
            out["filters"][dk] = f[dk]
    if op == "group_by":
        gf = _GROUP_FIELD_MAP.get(spec.get("group_field"))
        if not gf:
            return None
        out["group_field"] = gf
        out["top"] = int(spec.get("top", 10) or 10)
    return out


def build_spec(question, generate_fn, today=None):
    """Return a structured spec for a countable question, else None (→ use RAG).

    `generate_fn(prompt:str)->str` is the LLM call, injected so this module stays
    decoupled from the generation backend. Any failure returns None (safe → RAG).
    """
    if not looks_countable(question):
        return None
    today = today or datetime.now(timezone.utc).date().isoformat()
    try:
        raw = generate_fn(_spec_prompt(question, today))
    except Exception:  # noqa: BLE001 — never let intent detection break the bot
        return None
    spec = _parse_json(raw)
    if not spec or not spec.get("structured"):
        return None
    return _finalize(spec)
