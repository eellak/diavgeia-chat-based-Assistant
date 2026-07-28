"""
Resolve Diavgeia metadata IDs to human-readable names via the public OpenData API,
with a small on-disk cache — and match a user-mentioned name back to an id.

The index stores opaque foreign keys (organizationId, decisionTypeId, ...). This
module turns them into labels for readable, grounded answers, and resolves a name
the user typed ("ΕΚΕΤΑ") back to an organisation id for filtering. Cheap
canonicalisation over the metadata we already have — not LLM entity extraction.

Public, read-only endpoints:
  GET /opendata/types                -> full decision-type taxonomy (id -> label)
  GET /opendata/organizations/{uid}  -> one organisation (uid -> label, abbreviation)
"""
import json
import os
import unicodedata
import urllib.request

BASE = "https://diavgeia.gov.gr/opendata"
CACHE_DIR = os.getenv(
    "DIAVGEIA_CACHE", os.path.join(os.path.dirname(__file__), ".diavgeia_cache"))
_UA = {"User-Agent": "diavgeia-assistant/1.0", "Accept": "application/json"}


def _get_json(url, timeout=15):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _load_cache(name):
    try:
        with open(_cache_path(name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_cache(name, data):
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _norm(s):
    """Uppercase + strip accents, for accent-insensitive name matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.upper().strip()


# --- Decision-type taxonomy (small, fixed set — fetched once) --------------------
_types = None


def decision_type_labels(refresh=False):
    """Return {type_id: label} for the whole decision-type taxonomy."""
    global _types
    if _types is not None and not refresh:
        return _types
    if not refresh:
        cached = _load_cache("types.json")
        if cached:
            _types = cached
            return _types
    data = _get_json(f"{BASE}/types")
    items = data.get("decisionTypes") or data.get("types") or []
    _types = {(t.get("uid") or t.get("code")): (t.get("label") or t.get("name"))
              for t in items if (t.get("uid") or t.get("code"))}
    _save_cache("types.json", _types)
    return _types


def type_label(type_id):
    """type_id -> label, falling back to the id itself if unknown."""
    try:
        return decision_type_labels().get(type_id, type_id)
    except Exception:  # noqa: BLE001 — never let a lookup break the caller
        return type_id


def resolve_type(name):
    """Match a user phrase to a decision-type id (exact id, else label substring)."""
    if not name:
        return None
    try:
        labels = decision_type_labels()
    except Exception:  # noqa: BLE001
        return None
    if name in labels:                       # already an id, e.g. "Γ.3.2"
        return name
    n = _norm(name)
    matches = [tid for tid, lab in labels.items() if n and n in _norm(lab)]
    return matches[0] if matches else None


# --- Organisations (id <-> name), cached as {id: {label, abbreviation}} ----------
_orgs = None


def _orgs_cache():
    global _orgs
    if _orgs is None:
        _orgs = _load_cache("organizations.json") or {}
    return _orgs


def _resolve_org_raw(uid):
    try:
        d = _get_json(f"{BASE}/organizations/{uid}")
    except Exception:  # noqa: BLE001 — offline / unknown id
        return None
    return {"label": d.get("label") or d.get("name") or str(uid),
            "abbreviation": d.get("abbreviation")}


def organization_label(uid, refresh=False):
    """Resolve one organization id -> label (cached), falling back to the id."""
    if not uid:
        return None
    uid = str(uid)
    cache = _orgs_cache()
    if uid in cache and not refresh:
        return cache[uid].get("label") or uid
    entry = _resolve_org_raw(uid)
    if entry:
        cache[uid] = entry
        _save_cache("organizations.json", cache)
        return entry.get("label") or uid
    return uid


def warm_org_cache(es, index, size=1000):
    """Resolve every organisation id present in the index into the name cache.

    Run once (it hits the OpenData API per new id); afterwards name<->id lookups
    are instant from disk. Returns a small summary dict.
    """
    resp = es.search(index=index, body={
        "size": 0, "aggs": {"o": {"terms": {"field": "organization_id", "size": size}}}})
    ids = [b["key"] for b in resp["aggregations"]["o"]["buckets"]]
    cache = _orgs_cache()
    new = 0
    for uid in ids:
        uid = str(uid)
        if uid not in cache:
            entry = _resolve_org_raw(uid)
            if entry:
                cache[uid] = entry
                new += 1
    _save_cache("organizations.json", cache)
    return {"in_index": len(ids), "cached": len(cache), "newly_resolved": new}


# Leading Greek articles/particles to ignore when matching an org name.
_ORG_STOPWORDS = {"Ο", "Η", "ΤΟ", "ΟΙ", "ΤΑ", "ΤΟΥ", "ΤΗΣ", "ΤΩΝ",
                  "ΣΤΟ", "ΣΤΗ", "ΣΤΗΝ", "ΣΤΟΝ", "ΑΠΟ", "ΓΙΑ"}


def resolve_org(name):
    """Match a user-typed organisation name/abbreviation to an id, or None.

    Robust to Greek articles ("ο/του ΕΚΕΤΑ"): matches an abbreviation against the
    whole name or any of its tokens, then falls back to a label substring. Reads
    only the cache — warm it first.
    """
    n = _norm(name)
    if not n:
        return None
    tokens = [t for t in n.split() if t not in _ORG_STOPWORDS]
    tokenset = set(tokens)
    cache = _orgs_cache()
    for uid, e in cache.items():
        ab = _norm(e.get("abbreviation"))
        if ab and (ab == n or ab in tokenset):
            return uid
    stripped = " ".join(tokens) or n
    matches = [uid for uid, e in cache.items()
               if stripped in _norm(e.get("label")) or n in _norm(e.get("label"))]
    if matches:
        return min(matches, key=lambda u: len(cache[u].get("label") or ""))
    return None
