"""
Grounding helpers for the RAG answers: link each cited decision to its official
page on diavgeia.gov.gr, build an evidence/sources list for the UI, and provide
the "insufficient evidence" reply used when retrieval finds nothing.

The document URL is derived from the ΑΔΑ (which is also the Elasticsearch id):
    https://diavgeia.gov.gr/doc/{ada}
"""
DOC_URL = "https://diavgeia.gov.gr/doc/{}"

# Returned (instead of calling the LLM) when retrieval yields no documents — so the
# assistant says it doesn't know rather than hallucinating an answer.
INSUFFICIENT = ("Δεν εντόπισα σχετικές αποφάσεις στα διαθέσιμα έγγραφα, "
                "οπότε δεν μπορώ να απαντήσω με βεβαιότητα σε αυτό το ερώτημα.")


def doc_url(ada):
    return DOC_URL.format(ada)


def linkify_adas(text, allowed):
    """Turn ΑΔΑ mentions in `text` into markdown links — but only for ΑΔΑs in
    `allowed` (the actually-retrieved documents), so a hallucinated id is never
    linked. Longest ids first to avoid partial-overlap issues; idempotent.
    """
    if not text or not allowed:
        return text
    for ada in sorted({a for a in allowed if a}, key=len, reverse=True):
        if f"]({doc_url(ada)})" in text:  # already linked
            continue
        text = text.replace(ada, f"[{ada}]({doc_url(ada)})")
    return text


def sources_markdown(context_results, max_snippet=160):
    """Markdown bullet list of the retrieved decisions (ΑΔΑ link + snippet) for the
    evidence panel."""
    lines = []
    for r in context_results or []:
        ada = r.get("id")
        snippet = (r.get("content") or "").replace("\n", " ").strip()[:max_snippet]
        lines.append(f"- [{ada}]({doc_url(ada)}) — {snippet}{'…' if snippet else ''}")
    return "\n".join(lines) if lines else "_(καμία πηγή)_"
